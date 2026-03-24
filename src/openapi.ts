/**
 * @module OpenAPIImporter
 * @description
 * Import tools from OpenAPI 3.x specifications.
 * Each operation becomes a Veridex tool contract with the appropriate
 * safety class inferred from the HTTP method.
 */

import type { ToolContract, ToolSafetyClass } from '@veridex/agents';
import type { AdapterConfig } from './base';
import { jsonSchemaToZod, createStubExecutor } from './base';

// ── OpenAPI Types (minimal subset) ──

export interface OpenAPISpec {
  openapi: string;
  info: { title: string; version: string; description?: string };
  servers?: Array<{ url: string; description?: string }>;
  paths: Record<string, OpenAPIPathItem>;
}

export interface OpenAPIPathItem {
  get?: OpenAPIOperation;
  post?: OpenAPIOperation;
  put?: OpenAPIOperation;
  patch?: OpenAPIOperation;
  delete?: OpenAPIOperation;
}

export interface OpenAPIOperation {
  operationId?: string;
  summary?: string;
  description?: string;
  parameters?: OpenAPIParameter[];
  requestBody?: {
    content?: Record<string, { schema?: Record<string, unknown> }>;
  };
  responses?: Record<string, unknown>;
  tags?: string[];
}

export interface OpenAPIParameter {
  name: string;
  in: 'query' | 'path' | 'header' | 'cookie';
  required?: boolean;
  description?: string;
  schema?: Record<string, unknown>;
}

export interface OpenAPIImportOptions extends AdapterConfig {
  /** Base URL override for the API server */
  baseUrl?: string;
  /** Only import operations with these tags */
  filterTags?: string[];
  /** Only import specific operation IDs */
  filterOperationIds?: string[];
}

// ── Method to Safety Class ──

const METHOD_SAFETY: Record<string, ToolSafetyClass> = {
  get: 'read',
  post: 'write',
  put: 'write',
  patch: 'write',
  delete: 'destructive',
};

// ── Importer ──

export class OpenAPIImporter {
  private config: OpenAPIImportOptions;

  constructor(config?: OpenAPIImportOptions) {
    this.config = config ?? {};
  }

  /**
   * Import all operations from an OpenAPI spec as tool contracts.
   */
  importTools(spec: OpenAPISpec): {
    tools: ToolContract[];
    warnings: string[];
    apiInfo: { title: string; version: string; baseUrl?: string };
  } {
    const tools: ToolContract[] = [];
    const warnings: string[] = [];
    const baseUrl = this.config.baseUrl ?? spec.servers?.[0]?.url;

    for (const [path, pathItem] of Object.entries(spec.paths)) {
      for (const method of ['get', 'post', 'put', 'patch', 'delete'] as const) {
        const operation = pathItem[method];
        if (!operation) continue;

        // Apply tag filter
        if (this.config.filterTags?.length) {
          const hasTag = operation.tags?.some((t) =>
            this.config.filterTags!.includes(t),
          );
          if (!hasTag) continue;
        }

        // Apply operationId filter
        if (this.config.filterOperationIds?.length) {
          if (
            !operation.operationId ||
            !this.config.filterOperationIds.includes(operation.operationId)
          ) {
            continue;
          }
        }

        const tool = this.operationToTool(path, method, operation, baseUrl);
        if (tool) {
          tools.push(tool);
        } else {
          warnings.push(
            `Skipped ${method.toUpperCase()} ${path} — no operationId and could not generate name`,
          );
        }
      }
    }

    return {
      tools,
      warnings,
      apiInfo: {
        title: spec.info.title,
        version: spec.info.version,
        baseUrl,
      },
    };
  }

  private operationToTool(
    path: string,
    method: string,
    operation: OpenAPIOperation,
    baseUrl?: string,
  ): ToolContract | null {
    const rawName =
      operation.operationId ??
      `${method}_${path.replace(/[^a-zA-Z0-9]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '')}`;

    const name = this.config.toolNameTransform?.(rawName) ?? rawName;
    const description =
      operation.summary ?? operation.description ?? `${method.toUpperCase()} ${path}`;
    const safetyClass = this.config.defaultSafetyClass ?? METHOD_SAFETY[method] ?? 'read';

    // Build input schema from parameters + requestBody
    const properties: Record<string, Record<string, unknown>> = {};
    const required: string[] = [];

    for (const param of operation.parameters ?? []) {
      properties[param.name] = param.schema ?? { type: 'string' };
      if (param.required) required.push(param.name);
    }

    if (operation.requestBody?.content) {
      const jsonContent = operation.requestBody.content['application/json'];
      if (jsonContent?.schema) {
        properties['body'] = jsonContent.schema;
        required.push('body');
      }
    }

    const hasInput = Object.keys(properties).length > 0;
    const inputSchema = hasInput
      ? { type: 'object' as const, properties, required }
      : undefined;

    return {
      name,
      description,
      input: inputSchema ? jsonSchemaToZod(inputSchema) : undefined,
      safetyClass,
      execute: createStubExecutor('OpenAPI', rawName),
      metadata: {
        openapi: {
          path,
          method,
          baseUrl,
          tags: operation.tags,
        },
      },
    } as ToolContract;
  }
}
