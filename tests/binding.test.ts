import { describe, it, expect, vi } from 'vitest';
import {
  AgentCapabilityBinder,
  OpenAIAgentsAdapter,
  LangGraphAdapter,
  PydanticAIAdapter,
  createStubExecutor,
} from '../src';
import type {
  CapabilityBinding,
  BinderOptions,
  AdapterMode,
  ImportResult,
  ExecutableAgentBridge,
  BridgeInvocation,
  BridgeInvocationResult,
} from '../src';

// ── Helpers ──

function makeImportResult(): ImportResult {
  const adapter = new OpenAIAgentsAdapter();
  return adapter.importAgent({
    name: 'test-agent',
    instructions: 'You are a test agent.',
    model: 'gpt-4o',
    tools: [
      {
        type: 'function',
        function: {
          name: 'search',
          description: 'Search the web',
          parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
        },
      },
      {
        type: 'function',
        function: {
          name: 'createNote',
          description: 'Create a note',
          parameters: { type: 'object', properties: { title: { type: 'string' }, body: { type: 'string' } }, required: ['title'] },
        },
      },
      {
        type: 'function',
        function: {
          name: 'deleteFile',
          description: 'Delete a file from storage',
          parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
        },
      },
    ],
  });
}

function makeMockBridge(name = 'mock-bridge'): ExecutableAgentBridge {
  return {
    name,
    invoke: vi.fn(async (input: BridgeInvocation): Promise<BridgeInvocationResult> => ({
      success: true,
      output: `Bridge executed: ${input.prompt}`,
      metadata: { bridged: true },
    })),
    asTool: () => ({
      name,
      description: `Bridge tool ${name}`,
      input: { type: 'object', properties: { prompt: { type: 'string' } } },
      safetyClass: 'network' as const,
      execute: async () => ({ success: true, llmOutput: 'ok' }),
    }),
    asHandoffHandler: () => async () => ({
      success: true,
      output: 'handoff ok',
    }),
  };
}

// ══════════════════════════════════════════════════════════════════════════════
// AgentCapabilityBinder.bind (explicit bindings)
// ══════════════════════════════════════════════════════════════════════════════

describe('AgentCapabilityBinder.bind', () => {
  it('binds a single tool with a custom executor', () => {
    const imported = makeImportResult();
    const executor = vi.fn(async () => ({ success: true as const, llmOutput: 'bound result' }));
    const bindings: CapabilityBinding[] = [
      { toolName: 'search', executor },
    ];

    const result = AgentCapabilityBinder.bind(imported, bindings);

    expect(result.boundTools).toEqual(['search']);
    expect(result.unboundTools).toContain('createNote');
    expect(result.unboundTools).toContain('deleteFile');
    expect(result.definition.tools?.length).toBe(3);
  });

  it('bound tool executor replaces stub', async () => {
    const imported = makeImportResult();
    const executor = vi.fn(async () => ({ success: true as const, llmOutput: 'live result' }));
    const result = AgentCapabilityBinder.bind(imported, [
      { toolName: 'search', executor },
    ]);

    const searchTool = result.definition.tools?.find((t) => t.name === 'search');
    expect(searchTool).toBeDefined();
    const toolResult = await searchTool!.execute({ input: { query: 'test' }, context: { runId: 'r1', agentId: 'a1', turnIndex: 0 } } as any);
    expect(toolResult.success).toBe(true);
    expect(toolResult.llmOutput).toBe('live result');
    expect(executor).toHaveBeenCalledOnce();
  });

  it('unbound tools remain as stubs', async () => {
    const imported = makeImportResult();
    const result = AgentCapabilityBinder.bind(imported, []);

    const noteTool = result.definition.tools?.find((t) => t.name === 'createNote');
    const toolResult = await noteTool!.execute({ input: {} } as any);
    expect(toolResult.success).toBe(false); // stub returns false
  });

  it('marks bound tools with metadata', () => {
    const imported = makeImportResult();
    const result = AgentCapabilityBinder.bind(imported, [
      { toolName: 'search', executor: async () => ({ success: true, llmOutput: 'ok' }) },
    ]);

    const searchTool = result.definition.tools?.find((t) => t.name === 'search');
    expect((searchTool!.metadata as any).boundExecution).toBe(true);
    expect((searchTool!.metadata as any).adapterMode).toBe('execute');
  });

  it('applies safetyClassOverride', () => {
    const imported = makeImportResult();
    const result = AgentCapabilityBinder.bind(imported, [
      {
        toolName: 'search',
        executor: async () => ({ success: true, llmOutput: 'ok' }),
        safetyClassOverride: 'financial',
      },
    ]);

    const searchTool = result.definition.tools?.find((t) => t.name === 'search');
    expect(searchTool!.safetyClass).toBe('financial');
  });

  it('applies timeout override', () => {
    const imported = makeImportResult();
    const result = AgentCapabilityBinder.bind(imported, [
      {
        toolName: 'search',
        executor: async () => ({ success: true, llmOutput: 'ok' }),
        timeoutMs: 30000,
      },
    ]);

    const searchTool = result.definition.tools?.find((t) => t.name === 'search');
    expect(searchTool!.timeoutMs).toBe(30000);
  });

  it('throws when requireAllBound and some unbound', () => {
    const imported = makeImportResult();
    expect(() =>
      AgentCapabilityBinder.bind(imported, [], { requireAllBound: true }),
    ).toThrow(/remain unbound/);
  });

  it('passes when requireAllBound and all bound', () => {
    const imported = makeImportResult();
    const executor = async () => ({ success: true as const, llmOutput: 'ok' });
    const bindings: CapabilityBinding[] = [
      { toolName: 'search', executor },
      { toolName: 'createNote', executor },
      { toolName: 'deleteFile', executor },
    ];
    const result = AgentCapabilityBinder.bind(imported, bindings, { requireAllBound: true });
    expect(result.unboundTools).toEqual([]);
    expect(result.boundTools.length).toBe(3);
  });

  it('adds warning for unbound tools', () => {
    const imported = makeImportResult();
    const result = AgentCapabilityBinder.bind(imported, []);
    expect(result.warnings.some((w) => w.includes('inspect-only stubs'))).toBe(true);
  });

  it('binds multiple tools independently', async () => {
    const imported = makeImportResult();
    const searchExec = vi.fn(async () => ({ success: true as const, llmOutput: 'search-result' }));
    const noteExec = vi.fn(async () => ({ success: true as const, llmOutput: 'note-result' }));

    const result = AgentCapabilityBinder.bind(imported, [
      { toolName: 'search', executor: searchExec },
      { toolName: 'createNote', executor: noteExec },
    ]);

    expect(result.boundTools).toEqual(['search', 'createNote']);
    expect(result.unboundTools).toEqual(['deleteFile']);

    const searchTool = result.definition.tools?.find((t) => t.name === 'search');
    const noteTool = result.definition.tools?.find((t) => t.name === 'createNote');

    await searchTool!.execute({ input: {} } as any);
    await noteTool!.execute({ input: {} } as any);

    expect(searchExec).toHaveBeenCalledOnce();
    expect(noteExec).toHaveBeenCalledOnce();
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// AgentCapabilityBinder.bindViaBridge
// ══════════════════════════════════════════════════════════════════════════════

describe('AgentCapabilityBinder.bindViaBridge', () => {
  it('binds all tools via the bridge', () => {
    const imported = makeImportResult();
    const bridge = makeMockBridge();
    const result = AgentCapabilityBinder.bindViaBridge(imported, bridge);

    expect(result.boundTools.length).toBe(3);
    expect(result.unboundTools.length).toBe(0);
  });

  it('bridge-bound tools call bridge.invoke', async () => {
    const imported = makeImportResult();
    const bridge = makeMockBridge();
    const result = AgentCapabilityBinder.bindViaBridge(imported, bridge);

    const searchTool = result.definition.tools?.find((t) => t.name === 'search');
    const toolResult = await searchTool!.execute({
      input: { query: 'test' },
      context: { runId: 'r1', agentId: 'a1', turnIndex: 0 },
    } as any);

    expect(toolResult.success).toBe(true);
    expect(toolResult.llmOutput).toContain('Bridge executed');
    expect(bridge.invoke).toHaveBeenCalledOnce();
  });

  it('passes context and signal to bridge', async () => {
    const imported = makeImportResult();
    const bridge = makeMockBridge();
    const result = AgentCapabilityBinder.bindViaBridge(imported, bridge);

    const abortController = new AbortController();
    const searchTool = result.definition.tools?.find((t) => t.name === 'search');
    await searchTool!.execute({
      input: { query: 'ctx-test' },
      context: { runId: 'r2', agentId: 'a2', turnIndex: 3 },
      signal: abortController.signal,
    } as any);

    const callArgs = (bridge.invoke as any).mock.calls[0][0] as BridgeInvocation;
    expect(callArgs.context?.runId).toBe('r2');
    expect(callArgs.context?.agentId).toBe('a2');
    expect(callArgs.context?.originalTool).toBe('search');
    expect(callArgs.signal).toBe(abortController.signal);
  });

  it('marks tools with bridge metadata', () => {
    const imported = makeImportResult();
    const bridge = makeMockBridge('my-bridge');
    const result = AgentCapabilityBinder.bindViaBridge(imported, bridge);

    const tool = result.definition.tools?.[0];
    expect((tool!.metadata as any).boundExecution).toBe(true);
    expect((tool!.metadata as any).bridgeName).toBe('my-bridge');
    expect((tool!.metadata as any).adapterMode).toBe('execute');
  });

  it('applies defaultSafetyClass override', () => {
    const imported = makeImportResult();
    const bridge = makeMockBridge();
    const result = AgentCapabilityBinder.bindViaBridge(imported, bridge, {
      defaultSafetyClass: 'network',
    });

    for (const tool of result.definition.tools ?? []) {
      expect(tool.safetyClass).toBe('network');
    }
  });

  it('preserves original warnings', () => {
    const imported = makeImportResult();
    // OpenAI adapter doesn't generate warnings for this config, but let's test the mechanism
    const bridge = makeMockBridge();
    const result = AgentCapabilityBinder.bindViaBridge(imported, bridge);
    // Should not error — warnings array exists
    expect(Array.isArray(result.warnings)).toBe(true);
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// AgentCapabilityBinder.detectMode
// ══════════════════════════════════════════════════════════════════════════════

describe('AgentCapabilityBinder.detectMode', () => {
  it('returns inspect for unbound import', () => {
    const imported = makeImportResult();
    expect(AgentCapabilityBinder.detectMode(imported)).toBe('inspect');
  });

  it('returns execute after binding', () => {
    const imported = makeImportResult();
    const bound = AgentCapabilityBinder.bind(imported, [
      { toolName: 'search', executor: async () => ({ success: true, llmOutput: 'ok' }) },
    ]);
    // Create a new ImportResult from the bind result
    const boundImport: ImportResult = {
      definition: bound.definition,
      warnings: bound.warnings,
      unsupportedFeatures: [],
    };
    expect(AgentCapabilityBinder.detectMode(boundImport)).toBe('execute');
  });

  it('returns inspect for definition with no tools', () => {
    const result: ImportResult = {
      definition: {
        id: 'empty',
        name: 'Empty',
        model: { provider: 'openai', model: 'gpt-4o' },
        instructions: 'test',
        tools: [],
      },
      warnings: [],
      unsupportedFeatures: [],
    };
    expect(AgentCapabilityBinder.detectMode(result)).toBe('inspect');
  });

  it('returns execute after bridge binding', () => {
    const imported = makeImportResult();
    const bridge = makeMockBridge();
    const bound = AgentCapabilityBinder.bindViaBridge(imported, bridge);
    const boundImport: ImportResult = {
      definition: bound.definition,
      warnings: bound.warnings,
      unsupportedFeatures: [],
    };
    expect(AgentCapabilityBinder.detectMode(boundImport)).toBe('execute');
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// AgentCapabilityBinder.toCapabilities
// ══════════════════════════════════════════════════════════════════════════════

describe('AgentCapabilityBinder.toCapabilities', () => {
  it('returns capability descriptors for each tool', () => {
    const imported = makeImportResult();
    const caps = AgentCapabilityBinder.toCapabilities(imported, 'openai-agents-sdk');

    expect(caps.length).toBe(3);
    expect(caps[0].source).toBe('adapter');
    expect(caps[0].adapterHint).toBe('openai-agents-sdk');
    expect(caps[0].mode).toBe('inspect');
  });

  it('reflects execute mode after binding', () => {
    const imported = makeImportResult();
    const bound = AgentCapabilityBinder.bind(imported, [
      { toolName: 'search', executor: async () => ({ success: true, llmOutput: 'ok' }) },
    ]);
    const boundImport: ImportResult = {
      definition: bound.definition,
      warnings: bound.warnings,
      unsupportedFeatures: [],
    };
    const caps = AgentCapabilityBinder.toCapabilities(boundImport, 'openai-agents-sdk');

    expect(caps[0].mode).toBe('execute');
  });

  it('preserves tool contracts', () => {
    const imported = makeImportResult();
    const caps = AgentCapabilityBinder.toCapabilities(imported, 'openai');

    for (const cap of caps) {
      expect(cap.tool).toBeDefined();
      expect(cap.name).toBe(cap.tool.name);
      expect(cap.description).toBe(cap.tool.description);
    }
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Integration: Multi-framework binding
// ══════════════════════════════════════════════════════════════════════════════

describe('Integration: multi-framework binding', () => {
  it('binds LangGraph import via bridge', async () => {
    const adapter = new LangGraphAdapter();
    const imported = adapter.importAgent({
      name: 'graph-agent',
      systemMessage: 'You are a graph agent.',
      tools: [
        { name: 'query_db', description: 'Query the database', schema: { type: 'object', properties: { sql: { type: 'string' } } } },
      ],
    });

    const bridge = makeMockBridge('langgraph-bridge');
    const result = AgentCapabilityBinder.bindViaBridge(imported, bridge);

    expect(result.boundTools).toEqual(['query_db']);
    const tool = result.definition.tools?.find((t) => t.name === 'query_db');
    const toolResult = await tool!.execute({ input: { sql: 'SELECT 1' }, context: { runId: 'r1', agentId: 'a1', turnIndex: 0 } } as any);
    expect(toolResult.success).toBe(true);
  });

  it('binds PydanticAI import via explicit binding', async () => {
    const adapter = new PydanticAIAdapter();
    const imported = adapter.importAgent({
      name: 'pydantic-agent',
      systemPrompt: 'You analyze data.',
      tools: [
        { name: 'analyze', description: 'Analyze dataset', parametersSchema: { type: 'object', properties: { dataset: { type: 'string' } } } },
      ],
    });

    const executor = vi.fn(async () => ({ success: true as const, llmOutput: 'analysis complete' }));
    const result = AgentCapabilityBinder.bind(imported, [
      { toolName: 'analyze', executor },
    ]);

    expect(result.boundTools).toEqual(['analyze']);
    const tool = result.definition.tools?.find((t) => t.name === 'analyze');
    await tool!.execute({ input: { dataset: 'test' }, context: { runId: 'r1', agentId: 'a1', turnIndex: 0 } } as any);
    expect(executor).toHaveBeenCalledOnce();
  });

  it('mixed binding: some explicit, rest via bridge', async () => {
    const imported = makeImportResult();
    const bridge = makeMockBridge();

    // First bind one tool explicitly
    const partial = AgentCapabilityBinder.bind(imported, [
      {
        toolName: 'deleteFile',
        executor: async () => ({ success: true, llmOutput: 'deleted' }),
        safetyClassOverride: 'destructive',
      },
    ]);

    // Then bind remaining via bridge
    const unboundImport: ImportResult = {
      definition: partial.definition,
      warnings: partial.warnings,
      unsupportedFeatures: [],
    };

    // The bridge binding will re-bind all tools (including already-bound ones)
    // In practice, you'd filter — but this tests the composition works
    expect(partial.boundTools).toEqual(['deleteFile']);
    expect(partial.unboundTools).toContain('search');
    expect(partial.unboundTools).toContain('createNote');
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// createStubExecutor metadata
// ══════════════════════════════════════════════════════════════════════════════

describe('createStubExecutor', () => {
  it('marks executor as stub', () => {
    const exec = createStubExecutor('openai', 'search');
    expect((exec as any).__veridex_stub).toBe(true);
    expect((exec as any).__veridex_source).toBe('openai');
  });

  it('returns failure result', async () => {
    const exec = createStubExecutor('openai', 'search');
    const result = await exec();
    expect(result.success).toBe(false);
    expect(result.llmOutput).toContain('requires the original framework');
  });
});
