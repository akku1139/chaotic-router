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

export abstract class StatefulProvider extends BaseProvider {
  private threads = new Map<string, string>(); // userId → providerThreadId

  /**
   * Provider-specific thread creation process.
   * @param userId User identifier (obtained from request.user)
   * @returns Provider-side thread ID
   */
  protected abstract createThread(userId: string): Promise<string>;

  /**
   * Send a message to thread, return the response in chunks.
   * @param threadId Provider-side thread ID
   * @param message OpenAI message object (role, content)
   * @returns Async iterator
   */
  protected abstract sendMessage(
    threadId: string,
    message: OpenAIRequest['messages'][0],
  ): AsyncIterable<Chunk>;

  /**
   * Resets (forces the creation of a new) the thread for the specified user.
   * @param userId User ID (default: "default")
   */
  async resetThread(userId: string = 'default'): Promise<void> {
    this.threads.delete(userId);
  }

  /**
   * Processes OpenAI API requests.
   * - Uses request.user as the thread identifier.
   * - Automatically creates a thread if one does not exist.
   * - Sends the latest message to the thread and streams the response.
   */
  async *run(request: OpenAIRequest): AsyncIterable<Chunk> {
    const userId = request.user ?? 'default';
    let threadId = this.threads.get(userId);

    if (!threadId) {
      threadId = await this.createThread(userId);
      this.threads.set(userId, threadId);
    }

    const lastMessage = request.messages[request.messages.length - 1];
    if (!lastMessage) {
      throw new Error('No messages provided');
    }

    yield* this.sendMessage(threadId, lastMessage);
  }
}
