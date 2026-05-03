// SPDX-License-Identifier: AGPL-3.0-or-later

import { Hono } from 'hono';
import type { UpgradeWebSocket } from 'hono/ws';
import type { OpenAIRequest, OpenAIResponse, OpenAIStreamChunk } from './types.ts';
import { BaseProvider, UpstreamProvider } from './provider.ts';

export type ProviderConfig =
  | { prefix: string; provider: BaseProvider }
  | { prefix: string; upstream: string; apiKey?: string; headers?: Record<string, string> };

export interface ChaoticRouterOptions {
  providers: ProviderConfig[];
  upgradeWebSocket: UpgradeWebSocket;
}

export class ChaoticRouter {
  private app: Hono;
  private providers: { prefix: string; provider: BaseProvider }[];
  fetch: Hono['fetch'];

  constructor(options: ChaoticRouterOptions) {
    this.app = new Hono();
    this.providers = options.providers.map((cfg) => {
      if ('provider' in cfg) {
        return { prefix: cfg.prefix, provider: cfg.provider };
      } else {
        const upstreamProvider = new UpstreamProvider({
          baseUrl: cfg.upstream,
          apiKey: cfg.apiKey,
          headers: cfg.headers,
        });
        return { prefix: cfg.prefix, provider: upstreamProvider };
      }
    });

    this.setupRoutes(options.upgradeWebSocket);
    this.fetch = this.app.fetch;
  }

  private setupRoutes(upgradeWebSocket: UpgradeWebSocket) {
    this.app.get('/v1/models', async (c) => {
      const models: any[] = [];
      for (const { prefix, provider } of this.providers) {
        const modelNames = provider.listModels ? await provider.listModels() : [];
        for (const name of modelNames) {
          models.push({
            id: `${prefix}/${name}`,
            object: 'model',
            created: Math.floor(Date.now() / 1000),
            owned_by: prefix,
          });
        }
      }
      return c.json({ object: 'list', data: models });
    });

    this.app.post('/v1/chat/completions', async (c) => {
      const req = await c.req.json<OpenAIRequest>();
      const fullModel = req.model;
      const slashIndex = fullModel.indexOf('/');
      if (slashIndex === -1) {
        return c.json(
          {
            error: {
              message: `Model must be in format 'provider/model', got '${fullModel}'`,
              type: 'invalid_request_error',
            },
          },
          400,
        );
      }
      const prefix = fullModel.substring(0, slashIndex);
      const modelName = fullModel.substring(slashIndex + 1);
      const entry = this.providers.find((p) => p.prefix === prefix);
      if (!entry) {
        return c.json(
          { error: { message: `Unknown provider: ${prefix}`, type: 'invalid_request_error' } },
          404,
        );
      }

      const internalReq = { ...req, model: modelName };
      const streamMode = req.stream === true;

      try {
        const chunkIter = entry.provider.run(internalReq, c.req.raw.signal);
        if (!streamMode) {
          let fullContent = '';
          let finishReason = '';
          let usage: any = undefined;
          for await (const chunk of chunkIter) {
            if (chunk.type === 'text') fullContent += chunk.content;
            else if (chunk.type === 'final') {
              finishReason = chunk.finishReason ?? 'stop';
              usage = chunk.usage;
            }
          }
          const response: OpenAIResponse = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: fullModel,
            choices: [
              {
                index: 0,
                message: {
                  role: 'assistant',
                  content: fullContent,
                  tool_calls: undefined,
                  refusal: null,
                },
                finish_reason: finishReason as any,
                logprobs: null, // TODO
              },
            ],
            usage: usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          };
          return c.json(response);
        } else {
          const encoder = new TextEncoder();
          const readable = new ReadableStream({
            async start(controller) {
              try {
                for await (const chunk of chunkIter) {
                  if (chunk.type === 'text' && chunk.content) {
                    const sseChunk: OpenAIStreamChunk = {
                      id: `chatcmpl-${Date.now()}`,
                      object: 'chat.completion.chunk',
                      created: Math.floor(Date.now() / 1000),
                      model: fullModel,
                      choices: [
                        {
                          index: 0,
                          delta: { content: chunk.content },
                          finish_reason: null,
                        },
                      ],
                    };
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify(sseChunk)}\n\n`));
                  } else if (chunk.type === 'final') {
                    const finalChunk: OpenAIStreamChunk = {
                      id: `chatcmpl-${Date.now()}`,
                      object: 'chat.completion.chunk',
                      created: Math.floor(Date.now() / 1000),
                      model: fullModel,
                      choices: [
                        {
                          index: 0,
                          delta: {},
                          finish_reason: chunk.finishReason as any,
                        },
                      ],
                    };
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`));
                  }
                }
                controller.enqueue(encoder.encode('data: [DONE]\n\n'));
                controller.close();
              } catch (err) {
                controller.error(err);
              }
            },
          });
          return new Response(readable, {
            headers: { 'Content-Type': 'text/event-stream' },
          });
        }
      } catch (err) {
        console.error(err);
        return c.json({ error: { message: String(err), type: 'api_error' } }, 500);
      }
    });

    const wsEndpoints = new Map<string, BaseProvider>();
    for (const { provider } of this.providers) {
      for (const ep of provider.webSocketEndpoints?.() ?? []) {
        wsEndpoints.set(ep, provider);
      }
    }

    for (const [path, provider] of wsEndpoints.entries()) {
      this.app.get(
        path,
        upgradeWebSocket(c => {
          const sessionId = crypto.randomUUID();

          return {
            onOpen: async (_event, _ws) => {
              await provider.onWebSocketUpgrade?.(c.req.raw, path);
            },
            onMessage: async (event, ws) => {
              if (!provider.handleWebSocketMessage) return;
              const send = (data: unknown) => ws.send(JSON.stringify(data));
              try {
                const msg = JSON.parse(event.data as string);
                await provider.handleWebSocketMessage(sessionId, msg, send);
              } catch (err) {
                send({ type: 'error', error: { message: String(err) } });
              }
            },
            onClose: async (event, _ws) => {
              await provider.onWebSocketClose?.(sessionId, event.code, event.reason);
            },
          };
        }),
      );
    }
  }
}
