// SPDX-License-Identifier: AGPL-3.0-or-later

import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletion,
  ChatCompletionChunk,
} from 'openai/resources/chat/completions';

export type OpenAIRequest =
  | ChatCompletionCreateParamsNonStreaming
  | ChatCompletionCreateParamsStreaming;

export type OpenAIResponse = ChatCompletion;
export type OpenAIStreamChunk = ChatCompletionChunk;

export interface Chunk {
  type: 'text' | 'tool_call' | 'final';
  content?: string;
  toolCall?: { name: string; arguments: string };
  finishReason?: string;
  usage?: { promptTokens: number; completionTokens: number };
}
