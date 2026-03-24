/**
 * @module LangGraphAdapter
 * @description
 * Import/export adapter for LangGraph/LangChain agent definitions.
 *
 * LangGraph agents are defined as state graphs with nodes and edges.
 * This adapter maps the declarative parts (tools, prompts) into
 * Veridex AgentDefinition format.
 */

import type { AgentDefinition, ToolContract } from '@veridex/agents';
import type {
  FrameworkAdapter,
  AdapterConfig,
  ImportResult,
  ExportResult,
} from './base';
import {
  inferSafetyClass,
  createStubExecutor,
  jsonSchemaToZod,
} from './base';

// ── LangGraph Types ──

export interface LangGraphConfig {
  /** Agent name */
  name?: string;
  /** System message / prompt */
  systemMessage?: string;
  /** LLM configuration */
  llm?: {
    provider?: string;
    model?: string;
    temperature?: number;
  };
  /** Tool definitions */
  tools?: LangGraphToolDef[];
  /** Graph nodes (for reference/metadata only) */
  nodes?: LangGraphNode[];
  /** Graph edges (for reference/metadata only) */
  edges?: LangGraphEdge[];
  /** Checkpointer configuration */
  checkpointer?: unknown;
  metadata?: Record<string, unknown>;
}

export interface LangGraphToolDef {
  name: string;
  description?: string;
  schema?: Record<string, unknown>;
  returnDirect?: boolean;
}

export interface LangGraphNode {
  id: string;
  type: string;
  data?: Record<string, unknown>;
}

export interface LangGraphEdge {
  source: string;
  target: string;
  condition?: string;
}

export interface LangGraphExport {
  name: string;
  systemMessage: string;
  llm: { provider: string; model: string; temperature?: number };
  tools: LangGraphToolDef[];
  nodes: Array<{ id: string; type: string }>;
  edges: Array<{ source: string; target: string }>;
}

// ── Adapter ──

export class LangGraphAdapter
  implements FrameworkAdapter<LangGraphConfig, LangGraphExport>
{
  readonly name = 'langgraph';
  readonly framework = 'LangGraph';
  readonly frameworkVersion = '>=0.2.0';
  private config: AdapterConfig;

  constructor(config?: AdapterConfig) {
    this.config = config ?? {};
  }

  importAgent(source: LangGraphConfig): ImportResult {
    const warnings: string[] = [];
    const unsupportedFeatures: string[] = [];

    const tools: ToolContract[] = (source.tools ?? []).map((t) =>
      this.importTool(t),
    );

    if (source.nodes?.length) {
      unsupportedFeatures.push(
        `graph nodes (${source.nodes.length}) — LangGraph state graphs are not directly mappable to Veridex linear turns`,
      );
    }
    if (source.edges?.length) {
      unsupportedFeatures.push(
        `graph edges (${source.edges.length}) — conditional routing is handled via hooks/policies in Veridex`,
      );
    }
    if (source.checkpointer) {
      warnings.push(
        'LangGraph checkpointer config detected — Veridex uses its own CheckpointManager',
      );
    }

    const provider = source.llm?.provider ?? 'openai';
    const model = source.llm?.model ?? 'gpt-4o';

    const definition: AgentDefinition = {
      id: `langgraph-import-${(source.name ?? 'agent').toLowerCase().replace(/\s+/g, '-')}`,
      name: source.name ?? 'LangGraph Agent',
      model: {
        provider,
        model,
        temperature: source.llm?.temperature,
      },
      instructions: source.systemMessage ?? '',
      tools,
      metadata: {
        ...(source.metadata ?? {}),
        importedFrom: 'langgraph',
        importedAt: new Date().toISOString(),
        originalNodes: source.nodes?.length ?? 0,
        originalEdges: source.edges?.length ?? 0,
      },
    };

    return { definition, warnings, unsupportedFeatures };
  }

  exportAgent(definition: AgentDefinition): ExportResult<LangGraphExport> {
    const warnings: string[] = [];
    const droppedFeatures: string[] = [];

    if (definition.memory) droppedFeatures.push('memory configuration');
    if (definition.payments) droppedFeatures.push('payment configuration');
    if (definition.policies?.length) droppedFeatures.push('policy configurations');

    const tools: LangGraphToolDef[] = (definition.tools ?? []).map((t) => ({
      name: t.name,
      description: t.description,
      schema: t.input ? { type: 'object' } : undefined,
    }));

    const config: LangGraphExport = {
      name: definition.name,
      systemMessage: definition.instructions,
      llm: {
        provider: definition.model.provider,
        model: definition.model.model,
        temperature: definition.model.temperature,
      },
      tools,
      nodes: [
        { id: 'agent', type: 'agent' },
        { id: 'tools', type: 'tools' },
      ],
      edges: [
        { source: 'agent', target: 'tools' },
        { source: 'tools', target: 'agent' },
      ],
    };

    return { config, warnings, droppedFeatures };
  }

  importTool(toolDef: unknown): ToolContract {
    const def = toolDef as LangGraphToolDef;
    const name = this.config.toolNameTransform?.(def.name) ?? def.name;
    const safetyClass =
      this.config.defaultSafetyClass ?? inferSafetyClass(name, def.description);

    return {
      name,
      description: def.description ?? '',
      input: def.schema ? jsonSchemaToZod(def.schema) : undefined,
      safetyClass,
      execute: createStubExecutor(this.framework, def.name),
      metadata: {
        importedFrom: this.name,
        originalName: def.name,
        returnDirect: def.returnDirect,
      },
    } as ToolContract;
  }
}
