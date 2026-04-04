/**
 * @packageDocumentation
 * @module FrameworkAdapter
 * @description
 * Base adapter interface for importing/exporting agent definitions
 * between Veridex and other agent frameworks.
 */

import type { AgentDefinition, ToolContract, ToolSafetyClass } from '@veridex/agents';
import { z } from 'zod';
import type { AdapterMode } from './binding';

// ── Base Adapter Interface ──

/** Configuration for framework adapters. */
export interface AdapterConfig {
  /** Default safety class for imported tools without explicit classification */
  defaultSafetyClass?: ToolSafetyClass;
  /** Whether to preserve original metadata from the source framework */
  preserveMetadata?: boolean;
  /** Custom tool name transform */
  toolNameTransform?: (name: string) => string;
  /** Adapter mode: 'inspect' for stub-only, 'execute' for bound execution */
  mode?: AdapterMode;
}

/** Result of an import operation. */
export interface ImportResult {
  /** The imported agent definition */
  definition: AgentDefinition;
  /** Warnings generated during import */
  warnings: string[];
  /** Features that could not be mapped */
  unsupportedFeatures: string[];
}

/** Result of an export operation. */
export interface ExportResult<T = unknown> {
  /** The exported configuration in target framework format */
  config: T;
  /** Warnings generated during export */
  warnings: string[];
  /** Veridex features that have no equivalent in the target */
  droppedFeatures: string[];
}

/**
 * Base interface for framework adapters.
 * Each adapter implements import (from external → Veridex) and
 * export (from Veridex → external) operations.
 */
export interface FrameworkAdapter<TImport = unknown, TExport = unknown> {
  /** Adapter identifier */
  readonly name: string;
  /** Target framework name */
  readonly framework: string;
  /** Target framework version compatibility */
  readonly frameworkVersion: string;

  /** Import an agent definition from the target framework format. */
  importAgent(config: TImport): ImportResult;

  /** Export a Veridex agent definition to the target framework format. */
  exportAgent(definition: AgentDefinition): ExportResult<TExport>;

  /** Import a single tool definition from the target framework. */
  importTool(toolDef: unknown): ToolContract;
}

// ── Shared Utilities ──

/**
 * Infer a safety class from a tool name/description heuristic.
 */
export function inferSafetyClass(
  name: string,
  description?: string,
): ToolSafetyClass {
  const raw = `${name} ${description ?? ''}`;
  // Split camelCase before lowercasing for better word boundary matching
  const normalized = raw.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase();

  const writePatterns = /\b(write|create|update|delete|remove|send|transfer|pay|deploy|execute|modify|set|put|post|patch)\b/;
  const dangerousPatterns = /\b(admin|sudo|root|drop|truncate|destroy|nuke|reset|purge)\b/;

  if (dangerousPatterns.test(normalized)) return 'destructive';
  if (writePatterns.test(normalized)) return 'write';
  return 'read';
}

/**
 * Create a no-op tool execute function for imported tools.
 * The actual execution is delegated to the source framework at runtime.
 */
export function createStubExecutor(frameworkName: string, originalName: string) {
  const executor = async () => ({
    success: false as const,
    llmOutput: `Tool '${originalName}' was imported from ${frameworkName} and requires the original framework for execution.`,
  });
  // Mark as stub for mode detection
  (executor as Record<string, unknown>).__veridex_stub = true;
  (executor as Record<string, unknown>).__veridex_source = frameworkName;
  return executor;
}

/**
 * Convert a JSON Schema to a Zod schema (basic subset).
 * Handles string, number, boolean, object with properties, and arrays.
 */
export function jsonSchemaToZod(schema: Record<string, unknown>): z.ZodType {
  if (!schema || typeof schema !== 'object') return z.unknown();

  const type = schema.type as string | undefined;

  switch (type) {
    case 'string':
      return z.string();
    case 'number':
    case 'integer':
      return z.number();
    case 'boolean':
      return z.boolean();
    case 'array': {
      const items = schema.items as Record<string, unknown> | undefined;
      return z.array(items ? jsonSchemaToZod(items) : z.unknown());
    }
    case 'object': {
      const properties = schema.properties as Record<string, Record<string, unknown>> | undefined;
      if (!properties) return z.record(z.unknown());
      const required = (schema.required as string[]) ?? [];
      const shape: Record<string, z.ZodType> = {};
      for (const [key, prop] of Object.entries(properties)) {
        const zodProp = jsonSchemaToZod(prop);
        shape[key] = required.includes(key) ? zodProp : zodProp.optional();
      }
      return z.object(shape);
    }
    default:
      return z.unknown();
  }
}
