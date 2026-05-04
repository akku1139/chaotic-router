// SPDX-License-Identifier: AGPL-3.0-or-later

import type { Chunk, ChaoticRouterRequest } from './types.ts';
import type { Model } from 'openai/resources/models';
import type { Tool } from 'openai/resources/responses/responses.mjs';

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
  abstract run(request: ChaoticRouterRequest, signal?: AbortSignal, headers?: Headers): AsyncIterable<Chunk>;
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
    const headers: HeadersInit = { ...this.headers };
    if (this.apiKey) headers['Authorization'] = `Bearer ${this.apiKey}`;
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

  async *run(
    request: ChaoticRouterRequest,
    signal?: AbortSignal,
    _headers?: Headers,
  ): AsyncIterable<Chunk> {
    const url = `${this.baseUrl}/responses`;
    const fetchHeaders: HeadersInit = {
      'Content-Type': 'application/json',
      ...this.headers,
    };
    if (this.apiKey) fetchHeaders['Authorization'] = `Bearer ${this.apiKey}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: fetchHeaders,
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Upstream error ${response.status}: ${text}`);
    }

    const isStreaming = request.stream === true;

    if (!isStreaming) {
      const data = await response.json() as any;
      const output = data.output?.[0];
      let fullText = '';
      if (output?.type === 'message' && Array.isArray(output.content)) {
        fullText = output.content
          .filter((part: any) => part.type === 'output_text')
          .map((part: any) => part.text)
          .join('\n');
      } else if (data.output_text) {
        fullText = data.output_text;
      }
      yield { type: 'text', content: fullText };
      yield {
        type: 'final',
        finishReason: data.status === 'completed' ? 'stop' : 'error',
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
          if (line.startsWith('event:')) {
            // ignore
            continue;
          }
          if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim();
            if (dataStr === '[DONE]') continue;
            try {
              const event = JSON.parse(dataStr);
              if (event.type === 'response.output_text.delta') {
                yield { type: 'text', content: event.delta };
              } else if (event.type === 'response.completed') {
                yield { type: 'final', finishReason: 'stop', usage: event.response?.usage };
              } else if (event.type === 'response.failed') {
                yield { type: 'final', finishReason: 'error' };
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

export abstract class StatefulProvider<TThread = any> extends BaseProvider {
  /**
   * Provider-specific thread creation process.
   * @param systemPrompts System prompt array
   * @returns Provider-side thread ID
   */
  protected abstract createThread(userId: string, systemPrompts?: string[], tools?: Tool[]): Promise<TThread>;

  /**
   * Send a message to thread, return the response in chunks.
   * @param threadId Provider-side thread ID
   * @param newMessages OpenAI message objects (role, content)
   * @param fullRequest OpenAI request object
   * @returns Async iterator
   */
   protected abstract sendMessage(
     thread: TThread,
     newMessages: ChaoticRouterRequest['input'][number],
     fullRequest: ChaoticRouterRequest,
   ): AsyncIterable<Chunk>;

  /** Update system prompts of an existing thread (if supported). */
  protected updateSystemPrompt?(thread: TThread, systemPrompts: string[]): Promise<void>;
  protected updateTools?(thread: TThread, tools: any[]): Promise<void>;

  protected getThreadId?(thread: TThread): string; // option, for debug

  #threads = new Map<string, TThread>();
  #threadSystemPrompts = new Map<TThread, string[]>();
  #threadTools = new Map<TThread, any[]>();

  /** Extract user id from request (can be overridden). */
  protected getUserId(request: ChaoticRouterRequest, headers?: Headers): string {
    if (headers) {
      const windowId = headers.get('x-codex-window-id');
      if (windowId) return windowId;
      const sessionId = headers.get('session_id');
      if (sessionId) return sessionId;
    }
    if (request.metadata?.user_id) return request.metadata.user_id;
    if (request.metadata?.session_id) return request.metadata.session_id;
    return 'default';
  }

  /** Extract system prompts from messages array (developer/system roles). */
  private extractSystemPrompts(messages: ChaoticRouterRequest['input']): string[] {
    const prompts: string[] = [];
    for (const msg of messages) {
      if (msg.role === 'developer' || msg.role === 'system') {
        const content = msg.content;
        if (Array.isArray(content)) {
          for (const part of content) {
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
  private extractLastUserMessage(messages: ChaoticRouterRequest['input']): ChaoticRouterRequest['input'][number] {
    const userMsgs = messages.filter(m => m.role === 'user');
    const last = userMsgs[userMsgs.length - 1];
    if (!last) throw new Error('No user message found');
    return last;
  }

  private arraysEqual(a: any[], b: any[]): boolean {
    return a.length === b.length && a.every((v, i) => JSON.stringify(v) === JSON.stringify(b[i]));
  }

  async *run(request: ChaoticRouterRequest, _signal?: AbortSignal, headers?: Headers): AsyncIterable<Chunk> {
    const userId = this.getUserId(request, headers);
    const systemPrompts = this.extractSystemPrompts(request.input);
    const tools = request.tools;

    let thread = this.#threads.get(userId);
    const oldSystemPrompts = thread ? this.#threadSystemPrompts.get(thread) : undefined;
    const oldTools = thread ? this.#threadTools.get(thread) : undefined;

    if (!thread) {
      thread = await this.createThread(
        userId,
        systemPrompts.length ? systemPrompts : undefined,
        tools,
      );
      this.#threads.set(userId, thread);
      if (systemPrompts.length) this.#threadSystemPrompts.set(thread, systemPrompts);
      if (tools) this.#threadTools.set(thread, tools);
    } else {
      if (systemPrompts.length && !this.arraysEqual(oldSystemPrompts || [], systemPrompts)) {
        if (this.updateSystemPrompt) {
          await this.updateSystemPrompt(thread, systemPrompts);
          this.#threadSystemPrompts.set(thread, systemPrompts);
        } else {
          console.warn(`System prompts changed but provider does not support update.`);
        }
      }
      if (tools && !this.arraysEqual(oldTools || [], tools)) {
        if (this.updateTools) {
          await this.updateTools(thread, tools);
          this.#threadTools.set(thread, tools);
        } else {
          console.warn(`Tools changed but provider does not support update.`);
        }
      }
    }

    yield* this.sendMessage(thread, this.extractLastUserMessage(request.input), request);
  }
}
