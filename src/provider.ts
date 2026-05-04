// SPDX-License-Identifier: AGPL-3.0-or-later

import type { OpenAIRequest, Chunk } from './types.ts';
import type { Model } from 'openai/resources/models';

export abstract class BaseProvider {
  abstract listModels(): Promise<string[]>;
  webSocketEndpoints?(): string[];
  async onWebSocketUpgrade?(request: Request, endpoint: string): Promise<Response | void>;
  async handleWebSocketMessage?(
    sessionId: string,
    message: unknown,
    send: (event: unknown) => void,
  ): Promise<void>;
  async onWebSocketClose?(sessionId: string, code: number, reason: string): Promise<void>;
  abstract run(request: OpenAIRequest, signal?: AbortSignal): AsyncIterable<Chunk>;
}

export interface UpstreamOptions {
  baseUrl: string;
  apiKey?: string;
  headers?: Record<string, string>;
}

export class UpstreamProvider extends BaseProvider {
  private baseUrl: string;
  private apiKey?: string;
  private headers: Record<string, string>;
  private cachedModels: string[] | null = null;

  constructor(options: UpstreamOptions) {
    super();
    this.baseUrl = options.baseUrl.replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.headers = options.headers || {};
  }

  override async listModels(): Promise<string[]> {
    if (this.cachedModels) return this.cachedModels;
    const url = `${this.baseUrl}/models`;
    const headers: HeadersInit = {
      ...this.headers,
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    try {
      const response = await fetch(url, { headers });
      if (!response.ok) {
        console.warn(`Failed to fetch models from ${url}: ${response.status}`);
        return [];
      }
      const data = await response.json() as { data: Model[] };
      const models = data.data.map(m => m.id);
      this.cachedModels = models;
      return models;
    } catch (err) {
      console.warn(`Error fetching models from ${url}:`, err);
      return [];
    }
  }

  async *run(request: OpenAIRequest, signal?: AbortSignal): AsyncIterable<Chunk> {
    const url = `${this.baseUrl}/chat/completions`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...this.headers,
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Upstream error ${response.status}: ${text}`);
    }

    const isStreaming = request.stream === true;

    if (!isStreaming) {
      const data = await response.json();
      const content = data.choices?.[0]?.message?.content ?? '';
      yield { type: 'text', content };
      yield {
        type: 'final',
        finishReason: data.choices?.[0]?.finish_reason,
        usage: data.usage,
      };
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        if (signal?.aborted) break;
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr === '[DONE]') continue;
            try {
              const chunk = JSON.parse(dataStr);
              const delta = chunk.choices?.[0]?.delta;
              if (delta?.content) {
                yield { type: 'text', content: delta.content };
              }
              if (chunk.choices?.[0]?.finish_reason) {
                yield { type: 'final', finishReason: chunk.choices[0].finish_reason };
              }
            } catch {
              // ignore
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// provider.ts
export abstract class StatefulProvider<TThread = any> extends BaseProvider {
  /**
   * Provider-specific thread creation process.
   * @param systemPrompts System prompt array
   * @returns Provider-side thread ID
   */
  protected abstract createThread(userId: string, systemPrompts?: string[]): Promise<TThread>;

  /**
   * Send a message to thread, return the response in chunks.
   * @param threadId Provider-side thread ID
   * @param message OpenAI message object (role, content)
   * @returns Async iterator
   */
  protected abstract sendMessage(
    thread: TThread,
    message: OpenAIRequest['messages'][0],
  ): AsyncIterable<Chunk>;

  /** Update system prompts of an existing thread (if supported). */
  protected updateSystemPrompt?(thread: TThread, systemPrompts: string[]): Promise<void>;

  protected getThreadId?(thread: TThread): string; // option, for debug

  #threads = new Map<string, TThread>();
  #threadSystemPrompts = new Map<TThread, string[]>();

  /** Extract user id from request (can be overridden). */
  protected getUserId(request: OpenAIRequest): string {
    if (request.user) return request.user;
    if (request.metadata?.user_id) return request.metadata.user_id;
    return 'default';
  }

  /** Extract system prompts from messages array (developer/system roles). */
  private extractSystemPrompts(messages: OpenAIRequest['messages']): string[] {
    const prompts: string[] = [];
    for (const msg of messages) {
      if (msg.role === 'developer' || msg.role === 'system') {
        const content = msg.content;
        if (Array.isArray(content)) {
          for (const part of content as any[]) {
            if (part?.type === 'input_text' && typeof part.text === 'string') {
              prompts.push(part.text);
            }
          }
        } else if (typeof content === 'string') {
          prompts.push(content);
        }
      }
    }
    return prompts;
  }

  /** Extract the last user message from messages array. */
  private extractLastUserMessage(messages: OpenAIRequest['messages']): any {
    const userMsgs = messages.filter(m => m.role === 'user');
    const last = userMsgs[userMsgs.length - 1];
    if (!last) throw new Error('No user message found');
    return last;
  }

  private arraysEqual(a: string[], b: string[]): boolean {
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }

  async *run(request: OpenAIRequest, _signal?: AbortSignal): AsyncIterable<Chunk> {
    const userId = this.getUserId(request);
    const systemPrompts = this.extractSystemPrompts(request.messages);
    const lastUserMsg = this.extractLastUserMessage(request.messages);

    let thread = this.#threads.get(userId);
    const oldPrompts = thread ? this.#threadSystemPrompts.get(thread) : undefined;

    if (!thread) {
      thread = await this.createThread(userId, systemPrompts.length ? systemPrompts : undefined);
      this.#threads.set(userId, thread);
      if (systemPrompts.length) this.#threadSystemPrompts.set(thread, systemPrompts);
    } else if (systemPrompts.length && !this.arraysEqual(oldPrompts || [], systemPrompts)) {
      if (this.updateSystemPrompt) {
        await this.updateSystemPrompt(thread, systemPrompts);
        this.#threadSystemPrompts.set(thread, systemPrompts);
      } else {
        console.warn(`System prompts changed for thread but provider does not support update.`);
      }
    }

    yield* this.sendMessage(thread, lastUserMsg);
  }
}
