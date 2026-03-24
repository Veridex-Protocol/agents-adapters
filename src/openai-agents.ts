/**
 * @module OpenAIAgentsAdapter
 * @description
 * Import/export adapter for the OpenAI Agents SDK format.
 *
 * OpenAI Agents SDK defines agents with:
 * - name, instructions, model
 * - tools (function definitions with JSON Schema parameters)
 * - handoffs (agent-to-agent delegation)
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

// ── OpenAI Agents SDK Types ──

export interface OpenAIAgentConfig {
  name: string;
  instructions?: string;
  model?: string;
  tools?: OpenAIToolDef[];
  handoffs?: OpenAIHandoff[];
  metadata?: Record<string, unknown>;
}

export interface OpenAIToolDef {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenAIHandoff {
  agent: string;
  description?: string;
}

export interface OpenAIAgentExport {
  name: string;
  instructions: string;
  model: string;
  tools: OpenAIToolDef[];
  handoffs: OpenAIHandoff[];
}

// ── Adapter ──

export class OpenAIAgentsAdapter
  implements FrameworkAdapter<OpenAIAgentConfig, OpenAIAgentExport>
{
  readonly name = 'openai-agents';
  readonly framework = 'OpenAI Agents SDK';
  readonly frameworkVersion = '>=0.1.0';
  private config: AdapterConfig;

  constructor(config?: AdapterConfig) {
    this.config = config ?? {};
  }

  importAgent(source: OpenAIAgentConfig): ImportResult {
    const warnings: string[] = [];
    const unsupportedFeatures: string[] = [];

    const tools: ToolContract[] = (source.tools ?? []).map((t) =>
      this.importTool(t),
    );

    if (source.handoffs?.length) {
      unsupportedFeatures.push(
        `handoffs (${source.handoffs.length}) — Veridex uses explicit handoff proposals via the runtime`,
      );
    }

    const definition: AgentDefinition = {
      id: `openai-import-${source.name.toLowerCase().replace(/\s+/g, '-')}`,
      name: source.name,
      model: {
        provider: 'openai',
        model: source.model ?? 'gpt-4o',
      },
      instructions: source.instructions ?? '',
      tools,
      metadata: {
        ...(source.metadata ?? {}),
        importedFrom: 'openai-agents-sdk',
        importedAt: new Date().toISOString(),
      },
    };

    return { definition, warnings, unsupportedFeatures };
  }

  exportAgent(definition: AgentDefinition): ExportResult<OpenAIAgentExport> {
    const warnings: string[] = [];
    const droppedFeatures: string[] = [];

    if (definition.memory) droppedFeatures.push('memory configuration');
    if (definition.payments) droppedFeatures.push('payment configuration');
    if (definition.policies?.length) droppedFeatures.push('policy configurations');
    if (definition.hooks && Object.keys(definition.hooks).length) {
      droppedFeatures.push('lifecycle hooks');
    }

    const tools: OpenAIToolDef[] = (definition.tools ?? []).map((t) => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: t.input ? { type: 'object' } : undefined,
      },
    }));

    const config: OpenAIAgentExport = {
      name: definition.name,
      instructions: definition.instructions,
      model: definition.model.model,
      tools,
      handoffs: [],
    };

    return { config, warnings, droppedFeatures };
  }

  importTool(toolDef: unknown): ToolContract {
    const def = toolDef as OpenAIToolDef;
    const fn = def.function;
    const name = this.config.toolNameTransform?.(fn.name) ?? fn.name;
    const safetyClass =
      this.config.defaultSafetyClass ?? inferSafetyClass(name, fn.description);

    return {
      name,
      description: fn.description ?? '',
      input: fn.parameters ? jsonSchemaToZod(fn.parameters) : undefined,
      safetyClass,
      execute: createStubExecutor(this.framework, fn.name),
    } as ToolContract;
  }
}
