/**
 * @packageDocumentation
 * @module CapabilityBinding
 * @description
 * Capability binding contracts for adapter interop.
 *
 * Converts inspect-only imported tools into live-executable capabilities
 * by binding them to runtime bridges or custom executors.
 */

import type {
  ToolContract,
  ToolResult,
  ToolSafetyClass,
  ToolExecuteParams,
} from '@veridex/agents';
import type { ImportResult, AdapterConfig } from './base';
import type { ExecutableAgentBridge, BridgeInvocationResult } from './runtime-bridges';

// ── Adapter Mode ──

/** Determines how imported capabilities are used. */
export type AdapterMode = 'inspect' | 'execute';

// ── Capability Binding ──

/** Binding contract: maps a tool name to its live executor. */
export interface CapabilityBinding {
  /** The adapted tool name to bind. */
  toolName: string;
  /** The execution function that replaces the stub. */
  executor: (params: ToolExecuteParams) => Promise<ToolResult>;
  /** Override safety class if the binding has different risk. */
  safetyClassOverride?: ToolSafetyClass;
  /** Override timeout for bound execution. */
  timeoutMs?: number;
}

/** Result of a bind operation. */
export interface BindResult {
  /** Agent definition with bound tools. */
  definition: ImportResult['definition'];
  /** Tools that were successfully bound. */
  boundTools: string[];
  /** Tools that remained as stubs (no binding provided). */
  unboundTools: string[];
  /** Warnings from the binding process. */
  warnings: string[];
}

/** Options for the binder. */
export interface BinderOptions {
  /** Fail if any tool has no binding (default: false = leave as stub). */
  requireAllBound?: boolean;
  /** Adapter config for tool name transforms. */
  adapterConfig?: AdapterConfig;
}

// ── Agent Capability Binder ──

/**
 * Binds imported agent tools to live executors.
 *
 * Takes an ImportResult (from any FrameworkAdapter) and replaces
 * stub executors with real execution functions—either from explicit
 * bindings or from an ExecutableAgentBridge.
 */
export class AgentCapabilityBinder {
  /**
   * Bind tools using explicit per-tool bindings.
   */
  static bind(
    imported: ImportResult,
    bindings: CapabilityBinding[],
    options?: BinderOptions,
  ): BindResult {
    const bindingMap = new Map(bindings.map((b) => [b.toolName, b]));
    const boundTools: string[] = [];
    const unboundTools: string[] = [];
    const warnings: string[] = [...imported.warnings];

    const tools = (imported.definition.tools ?? []).map((tool) => {
      const binding = bindingMap.get(tool.name);
      if (!binding) {
        unboundTools.push(tool.name);
        return tool;
      }
      boundTools.push(tool.name);
      return {
        ...tool,
        execute: binding.executor,
        safetyClass: binding.safetyClassOverride ?? tool.safetyClass,
        timeoutMs: binding.timeoutMs ?? tool.timeoutMs,
        metadata: {
          ...tool.metadata,
          boundExecution: true,
          adapterMode: 'execute' as const,
        },
      };
    });

    if (options?.requireAllBound && unboundTools.length > 0) {
      throw new Error(
        `Required all tools to be bound, but these remain unbound: ${unboundTools.join(', ')}`,
      );
    }

    if (unboundTools.length > 0) {
      warnings.push(
        `${unboundTools.length} tool(s) remain as inspect-only stubs: ${unboundTools.join(', ')}`,
      );
    }

    return {
      definition: { ...imported.definition, tools },
      boundTools,
      unboundTools,
      warnings,
    };
  }

  /**
   * Bind all tools via an ExecutableAgentBridge.
   *
   * Each tool call is forwarded to the bridge's invoke method
   * with the tool name and arguments as context.
   */
  static bindViaBridge(
    imported: ImportResult,
    bridge: ExecutableAgentBridge,
    options?: BinderOptions & { defaultSafetyClass?: ToolSafetyClass },
  ): BindResult {
    const boundTools: string[] = [];
    const warnings: string[] = [...imported.warnings];

    const tools = (imported.definition.tools ?? []).map((tool) => {
      boundTools.push(tool.name);

      const bridgedExecutor = async (params: ToolExecuteParams): Promise<ToolResult> => {
        const result: BridgeInvocationResult = await bridge.invoke({
          prompt: JSON.stringify({
            tool: tool.name,
            arguments: params.input,
          }),
          context: {
            runId: params.context.runId,
            agentId: params.context.agentId,
            turnIndex: params.context.turnIndex,
            originalTool: tool.name,
            ...(params.context as Record<string, unknown>),
          },
          signal: params.signal,
        });

        return {
          success: result.success,
          llmOutput: result.output,
          uiOutput: result.uiOutput,
          attachments: result.attachments,
          error: result.error,
          metrics: result.metrics,
        };
      };

      return {
        ...tool,
        execute: bridgedExecutor,
        safetyClass: options?.defaultSafetyClass ?? tool.safetyClass,
        metadata: {
          ...tool.metadata,
          boundExecution: true,
          adapterMode: 'execute' as const,
          bridgeName: bridge.name,
        },
      };
    });

    return {
      definition: { ...imported.definition, tools },
      boundTools,
      unboundTools: [],
      warnings,
    };
  }

  /**
   * Check whether an ImportResult contains only stub executors (inspect mode)
   * or has any bound executors (execute mode).
   */
  static detectMode(imported: ImportResult): AdapterMode {
    const tools = imported.definition.tools ?? [];
    if (tools.length === 0) return 'inspect';

    const hasBound = tools.some(
      (t) => t.metadata && (t.metadata as Record<string, unknown>).boundExecution === true,
    );
    return hasBound ? 'execute' : 'inspect';
  }

  /**
   * Extract capability metadata for registration in ExecutionSurfaceRegistry.
   * Returns per-tool metadata needed for registration with source='adapter'.
   */
  static toCapabilities(
    imported: ImportResult,
    adapterHint: string,
  ): Array<{
    name: string;
    source: 'adapter';
    description: string;
    safetyClass: ToolSafetyClass;
    tool: ToolContract;
    adapterHint: string;
    mode: AdapterMode;
  }> {
    const mode = AgentCapabilityBinder.detectMode(imported);
    return (imported.definition.tools ?? []).map((tool) => ({
      name: tool.name,
      source: 'adapter' as const,
      description: tool.description,
      safetyClass: tool.safetyClass,
      tool,
      adapterHint,
      mode,
    }));
  }
}
