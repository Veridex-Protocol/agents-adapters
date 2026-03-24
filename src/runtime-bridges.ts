/**
 * @module RuntimeBridges
 * @description
 * Live execution bridges for external agent runtimes.
 *
 * These bridges let developers invoke existing agents without rewriting them
 * into Veridex definitions first. Each bridge can be used as a tool or as a
 * live handoff target inside the Veridex runtime.
 */

import type {
  HandoffHandler,
  ToolAttachment,
  ToolContract,
  ToolMetrics,
  ToolResult,
  ToolSafetyClass,
} from '@veridex/agents';
import { A2ATransport, type A2ATaskResult, type TransportAuthConfig } from '@veridex/agents';

type FetchLike = typeof fetch;

export interface BridgeInvocation {
  prompt: string;
  history?: unknown[];
  context?: Record<string, unknown>;
  threadId?: string;
  signal?: AbortSignal;
}

export interface BridgeInvocationResult {
  success: boolean;
  output: string;
  uiOutput?: string;
  attachments?: ToolAttachment[];
  history?: unknown[];
  metadata?: Record<string, unknown>;
  error?: string;
  metrics?: ToolMetrics;
  raw?: unknown;
}

export interface ExecutableAgentBridge {
  readonly name: string;
  invoke(input: BridgeInvocation): Promise<BridgeInvocationResult>;
  asTool(options?: BridgeToolOptions): ToolContract;
  asHandoffHandler(): HandoffHandler;
}

export interface BridgeToolOptions {
  name?: string;
  description?: string;
  safetyClass?: ToolSafetyClass;
  timeoutMs?: number;
}

function bridgeResultToToolResult(result: BridgeInvocationResult): ToolResult {
  return {
    success: result.success,
    llmOutput: result.output,
    uiOutput: result.uiOutput,
    attachments: result.attachments,
    error: result.error,
    metrics: result.metrics,
  };
}

function createBridgeTool(bridge: ExecutableAgentBridge, options?: BridgeToolOptions): ToolContract {
  return {
    name: options?.name ?? bridge.name,
    description: options?.description ?? `Invoke external agent runtime "${bridge.name}"`,
    input: {
      type: 'object',
      properties: {
        prompt: { type: 'string' },
        history: { type: 'array', items: {} },
        context: { type: 'object', additionalProperties: true },
        threadId: { type: 'string' },
      },
      required: ['prompt'],
    },
    safetyClass: options?.safetyClass ?? 'network',
    timeoutMs: options?.timeoutMs ?? 60_000,
    execute: async ({ input, signal }) =>
      bridgeResultToToolResult(await bridge.invoke({
        prompt: String((input as Record<string, unknown>).prompt ?? ''),
        history: (input as Record<string, unknown>).history as unknown[] | undefined,
        context: (input as Record<string, unknown>).context as Record<string, unknown> | undefined,
        threadId: (input as Record<string, unknown>).threadId as string | undefined,
        signal,
      })),
    metadata: {
      bridge: bridge.name,
      liveExecution: true,
    },
  };
}

function createBridgeHandoffHandler(bridge: ExecutableAgentBridge): HandoffHandler {
  return async (context) => {
    const result = await bridge.invoke({
      prompt: context.context
        ? `${context.reason}\n\n${context.context}`
        : context.reason,
      context: {
        runId: context.runId,
        turnIndex: context.turnIndex,
        sourceAgentId: context.agentId,
        targetAgentId: context.targetAgentId,
      },
    });

    return {
      success: result.success,
      output: result.output,
      uiOutput: result.uiOutput,
      attachments: result.attachments,
      metadata: result.metadata,
      error: result.error,
    };
  };
}

function textFromUnknown(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (value === null || value === undefined) {
    return '';
  }
  return JSON.stringify(value);
}

function extractMessageText(payload: unknown): string | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined;
  }

  const record = payload as Record<string, unknown>;
  if (typeof record.text === 'string') {
    return record.text;
  }
  if (typeof record.content === 'string') {
    return record.content;
  }
  if (Array.isArray(record.content)) {
    const textParts = record.content
      .map((item) => {
        if (typeof item === 'string') {
          return item;
        }
        if (item && typeof item === 'object' && 'text' in item && typeof (item as { text?: unknown }).text === 'string') {
          return (item as { text: string }).text;
        }
        return '';
      })
      .filter(Boolean);
    if (textParts.length) {
      return textParts.join('\n');
    }
  }
  return undefined;
}

function extractLangGraphOutput(payload: unknown): string {
  if (!payload || typeof payload !== 'object') {
    return textFromUnknown(payload);
  }

  const record = payload as Record<string, unknown>;
  if (typeof record.output === 'string') {
    return record.output;
  }
  if (record.output && typeof record.output === 'object') {
    const outputText = extractMessageText(record.output);
    if (outputText) {
      return outputText;
    }
  }
  if (record.values && typeof record.values === 'object') {
    const values = record.values as Record<string, unknown>;
    const messages = values.messages;
    if (Array.isArray(messages) && messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      const messageText = extractMessageText(lastMessage);
      if (messageText) {
        return messageText;
      }
    }
  }
  if (Array.isArray(record.messages) && record.messages.length > 0) {
    const lastMessage = record.messages[record.messages.length - 1];
    const messageText = extractMessageText(lastMessage);
    if (messageText) {
      return messageText;
    }
  }
  return textFromUnknown(payload);
}

export interface OpenAIRuntimeBridgeConfig {
  name: string;
  agent: unknown;
  description?: string;
  runner:
    | ((agent: unknown, input: string | unknown[], options?: Record<string, unknown>) => Promise<unknown>)
    | {
        run(agent: unknown, input: string | unknown[], options?: Record<string, unknown>): Promise<unknown>;
      };
  defaultContext?: Record<string, unknown>;
  maxTurns?: number;
}

export class OpenAIRuntimeBridge implements ExecutableAgentBridge {
  readonly name: string;
  private readonly agent: unknown;
  private readonly description?: string;
  private readonly runner: OpenAIRuntimeBridgeConfig['runner'];
  private readonly defaultContext?: Record<string, unknown>;
  private readonly maxTurns?: number;

  constructor(config: OpenAIRuntimeBridgeConfig) {
    this.name = config.name;
    this.agent = config.agent;
    this.description = config.description;
    this.runner = config.runner;
    this.defaultContext = config.defaultContext;
    this.maxTurns = config.maxTurns;
  }

  private async runOpenAI(input: string | unknown[], options?: Record<string, unknown>) {
    if (typeof this.runner === 'function') {
      return this.runner(this.agent, input, options);
    }
    return this.runner.run(this.agent, input, options);
  }

  async invoke(input: BridgeInvocation): Promise<BridgeInvocationResult> {
    const startedAt = Date.now();
    const runInput = input.history && input.history.length > 0
      ? input.history.concat([{ role: 'user', content: input.prompt }])
      : input.prompt;
    const raw = await this.runOpenAI(runInput, {
      context: { ...(this.defaultContext ?? {}), ...(input.context ?? {}) },
      maxTurns: this.maxTurns,
      signal: input.signal,
    });
    const record = raw as Record<string, unknown>;
    const output = typeof record.finalOutput === 'string'
      ? record.finalOutput
      : typeof record.output === 'string'
        ? record.output
        : textFromUnknown(record.finalOutput ?? record.output ?? raw);

    return {
      success: true,
      output,
      history: Array.isArray(record.history) ? record.history : undefined,
      metadata: {
        runtime: 'openai-agents-sdk',
        lastAgent: record.lastAgent,
        description: this.description,
      },
      metrics: {
        durationMs: Date.now() - startedAt,
      },
      raw,
    };
  }

  asTool(options?: BridgeToolOptions): ToolContract {
    return createBridgeTool(this, {
      description: options?.description ?? this.description,
      ...options,
    });
  }

  asHandoffHandler(): HandoffHandler {
    return createBridgeHandoffHandler(this);
  }
}

export interface LangGraphRuntimeBridgeConfig {
  name: string;
  endpoint: string;
  assistantId: string;
  description?: string;
  fetch?: FetchLike;
  auth?: TransportAuthConfig;
  headers?: Record<string, string>;
  defaultThreadId?: string;
}

export class LangGraphRuntimeBridge implements ExecutableAgentBridge {
  readonly name: string;
  private readonly endpoint: string;
  private readonly assistantId: string;
  private readonly description?: string;
  private readonly fetchImpl: FetchLike;
  private readonly auth?: TransportAuthConfig;
  private readonly headers?: Record<string, string>;
  private threadId?: string;

  constructor(config: LangGraphRuntimeBridgeConfig) {
    this.name = config.name;
    this.endpoint = config.endpoint;
    this.assistantId = config.assistantId;
    this.description = config.description;
    this.fetchImpl = config.fetch ?? globalThis.fetch.bind(globalThis);
    this.auth = config.auth;
    this.headers = config.headers;
    this.threadId = config.defaultThreadId;
  }

  private buildHeaders(extra?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...(this.headers ?? {}),
      ...(extra ?? {}),
    };
    if (this.auth?.type === 'bearer') {
      headers.Authorization = `Bearer ${this.auth.token}`;
    } else if (this.auth?.type === 'api-key') {
      headers[this.auth.headerName ?? 'X-Api-Key'] = this.auth.token;
    }
    return headers;
  }

  private async request<T>(pathname: string, init?: RequestInit): Promise<T> {
    const response = await this.fetchImpl(new URL(pathname, this.endpoint).toString(), {
      ...init,
      headers: this.buildHeaders(init?.headers as Record<string, string> | undefined),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(
        typeof (body as { detail?: unknown }).detail === 'string'
          ? (body as { detail: string }).detail
          : `LangGraph request failed with HTTP ${response.status}`,
      );
    }
    return body as T;
  }

  async createThread(): Promise<string> {
    const result = await this.request<{ thread_id: string }>('/threads', {
      method: 'POST',
      body: JSON.stringify({}),
    });
    this.threadId = result.thread_id;
    return result.thread_id;
  }

  async invoke(input: BridgeInvocation): Promise<BridgeInvocationResult> {
    const startedAt = Date.now();
    const threadId = input.threadId ?? this.threadId ?? await this.createThread();
    const historyMessages = Array.isArray(input.history)
      ? input.history.filter((item) => item && typeof item === 'object')
      : [];
    const payload = await this.request<Record<string, unknown>>(`/threads/${threadId}/runs/wait`, {
      method: 'POST',
      body: JSON.stringify({
        assistant_id: this.assistantId,
        input: {
          messages: historyMessages.concat([{ role: 'user', content: input.prompt }]),
        },
        context: input.context,
      }),
      signal: input.signal,
    });

    return {
      success: true,
      output: extractLangGraphOutput(payload),
      metadata: {
        runtime: 'langgraph',
        threadId,
        assistantId: this.assistantId,
        description: this.description,
      },
      metrics: {
        durationMs: Date.now() - startedAt,
      },
      raw: payload,
    };
  }

  asTool(options?: BridgeToolOptions): ToolContract {
    return createBridgeTool(this, {
      description: options?.description ?? this.description,
      ...options,
    });
  }

  asHandoffHandler(): HandoffHandler {
    return createBridgeHandoffHandler(this);
  }
}

export interface PydanticAIRuntimeBridgeConfig {
  name: string;
  endpoint: string;
  description?: string;
  fetch?: FetchLike;
  auth?: TransportAuthConfig;
  headers?: Record<string, string>;
  timeoutMs?: number;
}

function normalizeA2AOutput(result: A2ATaskResult): BridgeInvocationResult {
  return {
    success: result.status === 'completed',
    output: result.output?.text ?? '',
    attachments: result.output?.attachments,
    error: result.error,
    metadata: {
      runtime: 'pydanticai-a2a',
      taskId: result.id,
      status: result.status,
    },
    raw: result,
  };
}

export class PydanticAIRuntimeBridge implements ExecutableAgentBridge {
  readonly name: string;
  private readonly endpoint: string;
  private readonly description?: string;
  private readonly transport: A2ATransport;

  constructor(config: PydanticAIRuntimeBridgeConfig) {
    this.name = config.name;
    this.endpoint = config.endpoint;
    this.description = config.description;
    this.transport = new A2ATransport({
      fetch: config.fetch,
      auth: config.auth,
      headers: config.headers,
      timeoutMs: config.timeoutMs,
    });
  }

  async invoke(input: BridgeInvocation): Promise<BridgeInvocationResult> {
    const startedAt = Date.now();
    if (this.transport.state !== 'connected') {
      await this.transport.connect();
    }

    const result = await this.transport.sendTask(this.endpoint, {
      id: `a2a_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      input: {
        text: input.prompt,
      },
      metadata: input.context,
    }, {
      signal: input.signal,
    });

    const normalized = normalizeA2AOutput(result);
    return {
      ...normalized,
      metrics: {
        durationMs: Date.now() - startedAt,
      },
      metadata: {
        ...(normalized.metadata ?? {}),
        description: this.description,
      },
    };
  }

  asTool(options?: BridgeToolOptions): ToolContract {
    return createBridgeTool(this, {
      description: options?.description ?? this.description,
      ...options,
    });
  }

  asHandoffHandler(): HandoffHandler {
    return createBridgeHandoffHandler(this);
  }
}
