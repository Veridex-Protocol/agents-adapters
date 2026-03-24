/**
 * @packageDocumentation
 * @module @veridex/agents-adapters
 * @description
 * Framework adapters for the Veridex Agent Runtime.
 * Import/export agents from OpenAI Agents SDK, LangGraph, PydanticAI, and OpenAPI specs.
 */

// ── Base ──
export type {
  FrameworkAdapter,
  AdapterConfig,
  ImportResult,
  ExportResult,
} from './base';
export { inferSafetyClass, jsonSchemaToZod, createStubExecutor } from './base';

// ── OpenAI Agents SDK ──
export { OpenAIAgentsAdapter } from './openai-agents';
export type {
  OpenAIAgentConfig,
  OpenAIToolDef,
  OpenAIHandoff,
  OpenAIAgentExport,
} from './openai-agents';

// ── LangGraph ──
export { LangGraphAdapter } from './langgraph';
export type {
  LangGraphConfig,
  LangGraphToolDef,
  LangGraphNode,
  LangGraphEdge,
  LangGraphExport,
} from './langgraph';

// ── PydanticAI ──
export { PydanticAIAdapter } from './pydanticai';
export type {
  PydanticAIConfig,
  PydanticAIToolDef,
  PydanticAIExport,
} from './pydanticai';

// ── OpenAPI ──
export { OpenAPIImporter } from './openapi';
export type {
  OpenAPISpec,
  OpenAPIPathItem,
  OpenAPIOperation,
  OpenAPIParameter,
  OpenAPIImportOptions,
} from './openapi';
