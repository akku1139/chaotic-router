// SPDX-License-Identifier: AGPL-3.0-or-later

import { Hono } from 'hono';
import type { UpgradeWebSocket } from 'hono/ws';
import type { ResponsesRequest, ChaoticRouterRequest } from './types.ts';
import { BaseProvider, UpstreamProvider } from './provider.ts';
import type { Response as ResponsesResponse, ResponseStreamEvent } from 'openai/resources/responses/responses.mjs';

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

    this.app.post('/v1/responses', async (c) => {
      const req = await c.req.json<ResponsesRequest>();
      const fullModel = req.model;
      if (!fullModel) {
        return c.json({ error: { message: 'Model is required', type: 'invalid_request_error' } }, 400);
      }
      const slashIndex = fullModel.indexOf('/');
      if (slashIndex === -1) {
        return c.json({
          error: { message: `Model must be in format 'provider/model', got '${fullModel}'`, type: 'invalid_request_error' },
        }, 400);
      }
      const prefix = fullModel.substring(0, slashIndex);
      const modelName = fullModel.substring(slashIndex + 1);
      const entry = this.providers.find(p => p.prefix === prefix);
      if (!entry) {
        return c.json({ error: { message: `Unknown provider: ${prefix}` } }, 404);
      }

      const input: ChaoticRouterRequest['input'] = [];
      if (req.instructions) {
        input.push({ role: 'developer', content: [{ type: 'input_text', text: req.instructions }]});
      }
      if (typeof req.input === 'string') {
        input.push({ role: 'user', content: [{ type: 'input_text', text: req.input }]});
      } else if (Array.isArray(req.input)) {
        for (const item of req.input) {
          if (item.type !== 'message') continue; // TODO
          let role = item.role;
          let content = item.content;
          if (!role) continue;
          if (Array.isArray(content)) {
            const texts = content
              .filter((part: any) => part.type === 'input_text' && part.text)
              .map((part: any) => part.text);
            content = texts.join('\n');
          } else if (typeof content !== 'string') {
            content = JSON.stringify(content);
          }
          if (content) {
            if (role === 'developer') {
              input.push({ role: 'developer', content: [{ type: 'input_text', text: content }] });
            } else {
              input.push({ role: 'user', content: [{ type: 'input_text', text: content }] });
            }
          }
        }
      }

      const openaiReq: ChaoticRouterRequest = {
        ...req,
        model: modelName,
        input,
      };

      const streamMode = req.stream === true;

      try {
        const chunkIter = entry.provider.run(openaiReq, c.req.raw.signal);
        if (!streamMode) {
          let fullContent = '';
          for await (const chunk of chunkIter) {
            if (chunk.type === 'text') fullContent += chunk.content;
          }
          const response: ResponsesResponse = {
            id: `resp-${Date.now()}`,
            object: 'response',
            created_at: Math.floor(Date.now() / 1000),
            error: null,
            instructions: null,
            incomplete_details: null,
            model: fullModel,
            status: 'completed',
            output: [{
              type: 'message',
              id: `msg-${Date.now()}`,
              status: 'completed',
              role: 'assistant',
              content: [{
                type: 'output_text',
                text: fullContent,
                annotations: [],
              }],
            }],
            output_text: fullContent,
            parallel_tool_calls: true,
            previous_response_id: null,
            reasoning: {
              effort: null,
              summary: null
            },
            temperature: null,
            text: {
              format: {
                type: "text"
              }
            },
            tool_choice: "auto",
            tools: [],
            top_p: null,
            // FIXME
            usage: {
              input_tokens: 0,
              output_tokens: 0,
              total_tokens: 0,
              input_tokens_details: {
                cached_tokens: 0
              },
              output_tokens_details: {
                reasoning_tokens: 0
              },
            },
            metadata: {}
          };

          return c.json(response);
        } else {
          const encoder = new TextEncoder();
          const responseId = `resp_${Date.now()}_${crypto.randomUUID()}`;
          const itemId = `msg_${Date.now()}_${crypto.randomUUID()}`;
          const created_at = Math.floor(Date.now() / 1000);

          const responseBase = {
            id: responseId,
            object: 'response' as const,
            created_at,
            model: fullModel,
            error: null,
            incomplete_details: null,
            instructions: '',
            max_output_tokens: req.max_output_tokens ?? null,
            parallel_tool_calls: true,
            previous_response_id: null,
            reasoning: { effort: null, summary: null },
            store: true,
            temperature: req.temperature ?? 1.0,
            text: { format: { type: 'text' as const } },
            tool_choice: 'auto' as ResponsesResponse['tool_choice'],
            tools: [],
            top_p: 1.0,
            truncation: 'disabled' as const,
            metadata: {},

            output_text: '',
          } ;

          const readable = new ReadableStream({
            async start(controller) {
              let seq = 0;
              const event = <T extends ResponseStreamEvent>(data: T extends any ? Omit<T, 'sequence_number'> : never): void => {
                controller.enqueue(encoder.encode(
                  `event: ${data.type}\ndata: ${JSON.stringify({ ...data, sequence_number: seq++ } satisfies ResponseStreamEvent)}\n\n`
                ))
              };

              try {
                event({
                  type: 'response.created',
                  response: { ...responseBase, status: 'in_progress', output: [] },
                });

                event({
                  type: 'response.in_progress',
                  response: { ...responseBase, status: 'in_progress', output: [] },
                });

                event({
                  type: 'response.output_item.added',
                  output_index: 0,
                  item: {
                    id: itemId,
                    type: 'message',
                    status: 'in_progress',
                    role: 'assistant',
                    content: [],
                  },
                });

                event({
                  type: 'response.content_part.added',
                  item_id: itemId,
                  output_index: 0,
                  content_index: 0,
                  part: { type: 'output_text', text: '', annotations: [] },
                });

                let fullText = '';
                for await (const chunk of chunkIter) {
                  if (chunk.type === 'text' && chunk.content) {
                    fullText += chunk.content;
                    event({
                      type: 'response.output_text.delta',
                      item_id: itemId,
                      output_index: 0,
                      content_index: 0,
                      delta: chunk.content,
                      logprobs: [],
                    });
                  }
                }

                event({
                  type: 'response.output_text.done',
                  item_id: itemId,
                  output_index: 0,
                  content_index: 0,
                  text: fullText,
                  logprobs: [],
                });

                event({
                  type: 'response.content_part.done',
                  item_id: itemId,
                  output_index: 0,
                  content_index: 0,
                  part: { type: 'output_text', text: fullText, annotations: [] },
                });

                event({
                  type: 'response.output_item.done',
                  output_index: 0,
                  item: {
                    id: itemId,
                    type: 'message',
                    status: 'completed',
                    role: 'assistant',
                    content: [{ type: 'output_text', text: fullText, annotations: [] }],
                  },
                });

                event({
                  type: 'response.completed',
                  response: {
                    ...responseBase,
                    status: 'completed',
                    output: [{
                      id: itemId,
                      type: 'message',
                      status: 'completed',
                      role: 'assistant',
                      content: [{ type: 'output_text', text: fullText, annotations: [] }],
                    }],
                    usage: {
                      input_tokens: 0,
                      output_tokens: 0,
                      total_tokens: 0,
                      input_tokens_details: { cached_tokens: 0 },
                      output_tokens_details: { reasoning_tokens: 0 },
                    },
                  },
                });

                controller.close();
              } catch (err) {
                console.error('Stream error:', err);
                controller.error(err);
              }
            },
          });

          return new Response(readable, {
            headers: {
              'Content-Type': 'text/event-stream; charset=utf-8',
              'Cache-Control': 'no-cache',
              'Connection': 'keep-alive',
            },
          });
        }
      } catch (err) {
        console.error(err);
        return c.json({ error: { message: String(err) } }, 500);
      }
    });

    // FIXME: dummy
    this.app.get('/v1/responses/:id', async (c) => {
      const id = c.req.param('id');
      const dummyResponse: ResponsesResponse = {
        id,
        object: 'response',
        created_at: Math.floor(Date.now() / 1000),
        model: 'dummy',
        status: 'completed',
        output: [],
      } as any;
      return c.json(dummyResponse);
    });

    // FIXME: dummy
    this.app.get('/v1/responses', (c) => c.json({ data: [] }));

    // FIXME: dummy
    this.app.post('/v1/responses/:id/cancel', (c) => c.json({ success: true }));

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
