// SPDX-License-Identifier: AGPL-3.0-or-later

import type {
  ResponseCreateParamsNonStreaming,
  ResponseCreateParamsStreaming,
} from 'openai/resources/responses/responses';

export interface Chunk {
  type: 'text' | 'tool_call' | 'final';
  content?: string;
  toolCall?: { name: string; arguments: string };
  finishReason?: string;
  usage?: { promptTokens: number; completionTokens: number };
}

export type ResponsesRequest = ResponseCreateParamsNonStreaming | ResponseCreateParamsStreaming;
export type ChaoticRouterRequest = Omit<ResponsesRequest, 'input'> & {
  input: Extract<NonNullable<ResponsesRequest['input']>[number], { content: any[] }>[];
};
