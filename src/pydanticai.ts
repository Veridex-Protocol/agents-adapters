/**
 * @module PydanticAIAdapter
 * @description
 * Import/export adapter for PydanticAI agent definitions.
 *
 * PydanticAI agents use typed Python models (Pydantic) for tool schemas.
 * This adapter maps the JSON-serializable representations.
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

// ── PydanticAI Types ──

export interface PydanticAIConfig {
  name?: string;
  systemPrompt?: string;
  model?: string | { provider: string; model: string };
  tools?: PydanticAIToolDef[];
  resultType?: Record<string, unknown>;
  retries?: number;
  metadata?: Record<string, unknown>;
}

export interface PydanticAIToolDef {
  name: string;
  description?: string;
  parametersSchema?: Record<string, unknown>;
  returnSchema?: Record<string, unknown>;
  retries?: number;
}

export interface PydanticAIExport {
  name: string;
  systemPrompt: string;
  model: string;
  tools: PydanticAIToolDef[];
  resultType?: Record<string, unknown>;
}

// ── Adapter ──

export class PydanticAIAdapter
  implements FrameworkAdapter<PydanticAIConfig, PydanticAIExport>
{
  readonly name = 'pydanticai';
  readonly framework = 'PydanticAI';
  readonly frameworkVersion = '>=0.1.0';
  private config: AdapterConfig;

  constructor(config?: AdapterConfig) {
    this.config = config ?? {};
  }

  importAgent(source: PydanticAIConfig): ImportResult {
    const warnings: string[] = [];
    const unsupportedFeatures: string[] = [];

    const tools: ToolContract[] = (source.tools ?? []).map((t) =>
      this.importTool(t),
    );

    if (source.resultType) {
      warnings.push(
        'PydanticAI resultType detected — Veridex agents return free-form text; structured output can be implemented via hooks',
      );
    }

    if (source.retries) {
      warnings.push(
        `PydanticAI retries=${source.retries} — Veridex handles retries through hook/policy configuration`,
      );
    }

    let provider = 'openai';
    let model = 'gpt-4o';
    if (typeof source.model === 'string') {
      model = source.model;
    } else if (source.model) {
      provider = source.model.provider;
      model = source.model.model;
    }

    const definition: AgentDefinition = {
      id: `pydanticai-import-${(source.name ?? 'agent').toLowerCase().replace(/\s+/g, '-')}`,
      name: source.name ?? 'PydanticAI Agent',
      model: { provider, model },
      instructions: source.systemPrompt ?? '',
      tools,
      metadata: {
        ...(source.metadata ?? {}),
        importedFrom: 'pydanticai',
        importedAt: new Date().toISOString(),
      },
    };

    return { definition, warnings, unsupportedFeatures };
  }

  exportAgent(definition: AgentDefinition): ExportResult<PydanticAIExport> {
    const warnings: string[] = [];
    const droppedFeatures: string[] = [];

    if (definition.memory) droppedFeatures.push('memory configuration');
    if (definition.payments) droppedFeatures.push('payment configuration');
    if (definition.policies?.length) droppedFeatures.push('policy configurations');
    if (definition.hooks && Object.keys(definition.hooks).length) {
      droppedFeatures.push('lifecycle hooks');
    }

    const tools: PydanticAIToolDef[] = (definition.tools ?? []).map((t) => ({
      name: t.name,
      description: t.description,
      parametersSchema: t.input ? { type: 'object' } : undefined,
    }));

    const config: PydanticAIExport = {
      name: definition.name,
      systemPrompt: definition.instructions,
      model: definition.model.model,
      tools,
    };

    return { config, warnings, droppedFeatures };
  }

  importTool(toolDef: unknown): ToolContract {
    const def = toolDef as PydanticAIToolDef;
    const name = this.config.toolNameTransform?.(def.name) ?? def.name;
    const safetyClass =
      this.config.defaultSafetyClass ?? inferSafetyClass(name, def.description);

    return {
      name,
      description: def.description ?? '',
      input: def.parametersSchema ? jsonSchemaToZod(def.parametersSchema) : undefined,
      safetyClass,
      execute: createStubExecutor(this.framework, def.name),
    } as ToolContract;
  }
}
