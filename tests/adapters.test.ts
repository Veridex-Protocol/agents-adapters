import { describe, it, expect } from 'vitest';
import {
  OpenAIAgentsAdapter,
  LangGraphAdapter,
  PydanticAIAdapter,
  OpenAPIImporter,
  OpenAIRuntimeBridge,
  LangGraphRuntimeBridge,
  PydanticAIRuntimeBridge,
  inferSafetyClass,
  jsonSchemaToZod,
  PolicyStrippingError,
} from '../src';

// ── inferSafetyClass ──

describe('inferSafetyClass', () => {
  it('returns read for harmless names', () => {
    expect(inferSafetyClass('getBalance', 'Fetch account balance')).toBe('read');
  });

  it('returns write for write-like names', () => {
    expect(inferSafetyClass('createUser', 'Create a user')).toBe('write');
    expect(inferSafetyClass('sendEmail')).toBe('write');
    expect(inferSafetyClass('updateRecord')).toBe('write');
  });

  it('returns destructive for dangerous names', () => {
    expect(inferSafetyClass('dropTable')).toBe('destructive');
    expect(inferSafetyClass('adminReset')).toBe('destructive');
  });
});

// ── jsonSchemaToZod ──

describe('jsonSchemaToZod', () => {
  it('converts string type', () => {
    const schema = jsonSchemaToZod({ type: 'string' });
    expect(schema.parse('hello')).toBe('hello');
  });

  it('converts number type', () => {
    const schema = jsonSchemaToZod({ type: 'number' });
    expect(schema.parse(42)).toBe(42);
  });

  it('converts boolean type', () => {
    const schema = jsonSchemaToZod({ type: 'boolean' });
    expect(schema.parse(true)).toBe(true);
  });

  it('converts object with properties', () => {
    const schema = jsonSchemaToZod({
      type: 'object',
      properties: { name: { type: 'string' }, age: { type: 'number' } },
      required: ['name'],
    });
    expect(schema.parse({ name: 'Alice', age: 30 })).toEqual({ name: 'Alice', age: 30 });
    expect(schema.parse({ name: 'Bob' })).toEqual({ name: 'Bob' });
  });

  it('converts array type', () => {
    const schema = jsonSchemaToZod({ type: 'array', items: { type: 'string' } });
    expect(schema.parse(['a', 'b'])).toEqual(['a', 'b']);
  });

  it('returns unknown for unrecognized schema', () => {
    const schema = jsonSchemaToZod({});
    expect(schema.parse('anything')).toBe('anything');
  });
});

// ── OpenAI Agents SDK Adapter ──

describe('OpenAIAgentsAdapter', () => {
  const adapter = new OpenAIAgentsAdapter();

  describe('importAgent', () => {
    it('imports a basic agent', () => {
      const result = adapter.importAgent({
        name: 'Helper',
        instructions: 'Do stuff',
        model: 'gpt-4o-mini',
        tools: [
          {
            type: 'function',
            function: {
              name: 'greet',
              description: 'Greet user',
              parameters: { type: 'object', properties: { name: { type: 'string' } } },
            },
          },
        ],
      });

      expect(result.definition.name).toBe('Helper');
      expect(result.definition.model.model).toBe('gpt-4o-mini');
      expect(result.definition.tools).toHaveLength(1);
      expect(result.definition.tools![0].name).toBe('greet');
      expect(result.warnings).toEqual([]);
    });

    it('reports handoffs as unsupported', () => {
      const result = adapter.importAgent({
        name: 'Agent',
        handoffs: [{ agent: 'other', description: 'delegate' }],
      });

      expect(result.unsupportedFeatures).toHaveLength(1);
      expect(result.unsupportedFeatures[0]).toContain('handoffs');
    });

    it('defaults to gpt-4o when no model specified', () => {
      const result = adapter.importAgent({ name: 'Agent' });
      expect(result.definition.model.model).toBe('gpt-4o');
    });
  });

  describe('exportAgent', () => {
    it('exports a basic agent', () => {
      const result = adapter.exportAgent({
        id: 'test',
        name: 'Test',
        model: { provider: 'openai', model: 'gpt-4o' },
        instructions: 'Be helpful',
      });

      expect(result.config.name).toBe('Test');
      expect(result.config.model).toBe('gpt-4o');
      expect(result.config.instructions).toBe('Be helpful');
    });

    it('reports dropped features', () => {
      const result = adapter.exportAgent({
        id: 'test',
        name: 'Test',
        model: { provider: 'openai', model: 'gpt-4o' },
        instructions: '',
        payments: { dailyLimitUSD: 100 },
        memory: { defaultTier: 'working' },
      });

      expect(result.droppedFeatures).toContain('memory configuration');
      expect(result.droppedFeatures).toContain('payment configuration');
    });
  });

  describe('importTool', () => {
    it('imports a function tool', () => {
      const tool = adapter.importTool({
        type: 'function',
        function: {
          name: 'deleteFile',
          description: 'Delete a file',
          parameters: { type: 'object', properties: { path: { type: 'string' } } },
        },
      });

      expect(tool.name).toBe('deleteFile');
      expect(tool.safetyClass).toBe('write');
    });
  });
});

// ── LangGraph Adapter ──

describe('LangGraphAdapter', () => {
  const adapter = new LangGraphAdapter();

  describe('importAgent', () => {
    it('imports a basic config', () => {
      const result = adapter.importAgent({
        name: 'Graph Agent',
        systemMessage: 'You solve problems',
        llm: { provider: 'openai', model: 'gpt-4o' },
        tools: [{ name: 'search', description: 'Search the web' }],
      });

      expect(result.definition.name).toBe('Graph Agent');
      expect(result.definition.tools).toHaveLength(1);
    });

    it('reports nodes and edges as unsupported', () => {
      const result = adapter.importAgent({
        name: 'Agent',
        nodes: [
          { id: 'a', type: 'agent' },
          { id: 'b', type: 'tools' },
        ],
        edges: [{ source: 'a', target: 'b' }],
      });

      expect(result.unsupportedFeatures).toHaveLength(2);
      expect(result.unsupportedFeatures[0]).toContain('nodes');
      expect(result.unsupportedFeatures[1]).toContain('edges');
    });

    it('warns about checkpointer', () => {
      const result = adapter.importAgent({
        name: 'Agent',
        checkpointer: { type: 'memory' },
      });

      expect(result.warnings).toHaveLength(1);
      expect(result.warnings[0]).toContain('checkpointer');
    });
  });

  describe('exportAgent', () => {
    it('exports with default graph structure', () => {
      const result = adapter.exportAgent({
        id: 'test',
        name: 'Test',
        model: { provider: 'openai', model: 'gpt-4o' },
        instructions: 'Help',
      });

      expect(result.config.nodes).toHaveLength(2);
      expect(result.config.edges).toHaveLength(2);
    });
  });
});

// ── PydanticAI Adapter ──

describe('PydanticAIAdapter', () => {
  const adapter = new PydanticAIAdapter();

  describe('importAgent', () => {
    it('imports with string model', () => {
      const result = adapter.importAgent({
        name: 'Pydantic Agent',
        systemPrompt: 'Be precise',
        model: 'claude-3-opus',
        tools: [{ name: 'calculate', description: 'Do math' }],
      });

      expect(result.definition.model.model).toBe('claude-3-opus');
      expect(result.definition.tools).toHaveLength(1);
    });

    it('imports with structured model', () => {
      const result = adapter.importAgent({
        name: 'Agent',
        model: { provider: 'anthropic', model: 'claude-3-sonnet' },
      });

      expect(result.definition.model.provider).toBe('anthropic');
    });

    it('warns about resultType', () => {
      const result = adapter.importAgent({
        name: 'Agent',
        resultType: { type: 'object', properties: { answer: { type: 'string' } } },
      });

      expect(result.warnings).toHaveLength(1);
      expect(result.warnings[0]).toContain('resultType');
    });
  });
});

// ── OpenAPI Importer ──

describe('OpenAPIImporter', () => {
  const importer = new OpenAPIImporter();

  const sampleSpec = {
    openapi: '3.0.0',
    info: { title: 'Pet Store', version: '1.0.0' },
    components: {
      schemas: {
        Pet: {
          type: 'object',
          properties: { id: { type: 'string' }, name: { type: 'string' } },
        },
      },
    },
    paths: {
      '/pets': {
        get: {
          operationId: 'listPets',
          summary: 'List all pets',
          parameters: [
            { name: 'limit', in: 'query' as const, schema: { type: 'number' } },
          ],
        },
        post: {
          operationId: 'createPet',
          summary: 'Create a pet',
          requestBody: {
            content: {
              'application/json': {
                schema: { type: 'object', properties: { name: { type: 'string' } } },
              },
            },
          },
        },
      },
      '/pets/{petId}': {
        get: {
          operationId: 'getPet',
          summary: 'Get a pet',
          parameters: [
            { name: 'petId', in: 'path' as const, required: true, schema: { type: 'string' } },
          ],
          responses: {
            '200': {
              content: {
                'application/json': {
                  schema: { $ref: '#/components/schemas/Pet' },
                },
              },
            },
          },
        },
        delete: {
          operationId: 'deletePet',
          summary: 'Delete a pet',
          parameters: [
            { name: 'petId', in: 'path' as const, required: true, schema: { type: 'string' } },
          ],
        },
      },
    },
  };

  it('imports all operations as tools', () => {
    const { tools, apiInfo } = importer.importTools(sampleSpec);
    expect(tools).toHaveLength(4);
    expect(apiInfo.title).toBe('Pet Store');
  });

  it('assigns correct safety classes by HTTP method', () => {
    const { tools } = importer.importTools(sampleSpec);
    const getTools = tools.filter((t) => t.name === 'listPets' || t.name === 'getPet');
    const postTool = tools.find((t) => t.name === 'createPet');
    const deleteTool = tools.find((t) => t.name === 'deletePet');

    expect(getTools.every((t) => t.safetyClass === 'read')).toBe(true);
    expect(postTool?.safetyClass).toBe('write');
    expect(deleteTool?.safetyClass).toBe('destructive');
  });

  it('includes parameters in tool input schema', () => {
    const { tools } = importer.importTools(sampleSpec);
    const getPet = tools.find((t) => t.name === 'getPet');
    expect(getPet?.input).toBeDefined();
  });

  it('resolves component refs into output schemas', () => {
    const { tools } = importer.importTools(sampleSpec);
    const getPet = tools.find((t) => t.name === 'getPet');
    expect(getPet?.output).toEqual({
      type: 'object',
      properties: { id: { type: 'string' }, name: { type: 'string' } },
    });
    expect(getPet?.metadata).toBeDefined();
  });

  it('filters by tag', () => {
    const spec = {
      ...sampleSpec,
      paths: {
        '/pets': {
          get: { operationId: 'listPets', tags: ['pets'] },
        },
        '/admin': {
          get: { operationId: 'adminDashboard', tags: ['admin'] },
        },
      },
    };

    const tagImporter = new OpenAPIImporter({ filterTags: ['pets'] });
    const { tools } = tagImporter.importTools(spec);
    expect(tools).toHaveLength(1);
    expect(tools[0].name).toBe('listPets');
  });

  it('filters by operationId', () => {
    const opImporter = new OpenAPIImporter({
      filterOperationIds: ['getPet'],
    });
    const { tools } = opImporter.importTools(sampleSpec);
    expect(tools).toHaveLength(1);
    expect(tools[0].name).toBe('getPet');
  });
});

// ── Custom config ──

describe('AdapterConfig', () => {
  it('applies toolNameTransform', () => {
    const adapter = new OpenAIAgentsAdapter({
      toolNameTransform: (name) => `veridex_${name}`,
    });
    const tool = adapter.importTool({
      type: 'function',
      function: { name: 'myTool', description: 'test' },
    });
    expect(tool.name).toBe('veridex_myTool');
  });

  it('applies defaultSafetyClass', () => {
    const adapter = new LangGraphAdapter({
      defaultSafetyClass: 'financial',
    });
    const tool = adapter.importTool({ name: 'anything', description: 'test' });
    expect(tool.safetyClass).toBe('financial');
  });
});

// ── Policy stripping guard ──

describe('exportAgent policy guard', () => {
  const defWithPolicies = {
    id: 'test',
    name: 'Test',
    model: { provider: 'openai' as const, model: 'gpt-4o' },
    instructions: 'Help',
    policies: [
      {
        type: 'deny-tool',
        params: { toolName: 'anything' },
      },
    ],
  };

  it('throws PolicyStrippingError from LangGraphAdapter by default', () => {
    const adapter = new LangGraphAdapter();
    expect(() => adapter.exportAgent(defWithPolicies)).toThrow(PolicyStrippingError);
  });

  it('throws PolicyStrippingError from OpenAIAgentsAdapter by default', () => {
    const adapter = new OpenAIAgentsAdapter();
    expect(() => adapter.exportAgent(defWithPolicies)).toThrow(PolicyStrippingError);
  });

  it('throws PolicyStrippingError from PydanticAIAdapter by default', () => {
    const adapter = new PydanticAIAdapter();
    expect(() => adapter.exportAgent(defWithPolicies)).toThrow(PolicyStrippingError);
  });

  it('allows export and reports droppedFeatures when allowPolicyStripping is true', () => {
    const adapter = new LangGraphAdapter({ allowPolicyStripping: true });
    const result = adapter.exportAgent(defWithPolicies);
    expect(result.droppedFeatures).toContain('policy configurations');
  });

  it('does not throw for definitions without policies', () => {
    const adapter = new LangGraphAdapter();
    expect(() =>
      adapter.exportAgent({
        id: 'test',
        name: 'Test',
        model: { provider: 'openai', model: 'gpt-4o' },
        instructions: 'Help',
      }),
    ).not.toThrow();
  });
});

// ── Live Runtime Bridges ──

describe('live runtime bridges', () => {
  it('invokes an OpenAI Agents runtime through a live bridge', async () => {
    const bridge = new OpenAIRuntimeBridge({
      name: 'openai-support',
      agent: { id: 'support-agent' },
      runner: async (_agent, input) => ({
        finalOutput: typeof input === 'string' ? `Handled: ${input}` : 'Handled thread',
        history: [{ role: 'assistant', content: 'Handled' }],
      }),
    });

    const result = await bridge.invoke({ prompt: 'help me' });
    expect(result.success).toBe(true);
    expect(result.output).toBe('Handled: help me');

    const tool = bridge.asTool();
    const toolResult = await tool.execute({
      input: { prompt: 'hello' },
      context: { runId: 'run', agentId: 'agent', turnIndex: 0 },
    });
    expect(toolResult.llmOutput).toBe('Handled: hello');
  });

  it('invokes a LangGraph deployment through the server API', async () => {
    const fetchStub: typeof fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/threads')) {
        return new Response(JSON.stringify({ thread_id: 'thread_123' }), { status: 200 });
      }
      if (url.endsWith('/threads/thread_123/runs/wait')) {
        const body = JSON.parse(String(init?.body));
        expect(body.assistant_id).toBe('assistant_123');
        return new Response(JSON.stringify({
          values: {
            messages: [
              { role: 'assistant', content: [{ text: 'LangGraph completed the task' }] },
            ],
          },
        }), { status: 200 });
      }
      return new Response(JSON.stringify({ error: 'not found' }), { status: 404 });
    };

    const bridge = new LangGraphRuntimeBridge({
      name: 'langgraph-agent',
      endpoint: 'http://langgraph.local',
      assistantId: 'assistant_123',
      fetch: fetchStub,
    });

    const result = await bridge.invoke({ prompt: 'summarize this' });
    expect(result.success).toBe(true);
    expect(result.output).toContain('LangGraph completed the task');
  });

  it('invokes a PydanticAI A2A endpoint through the live bridge', async () => {
    const fetchStub: typeof fetch = async (_input, init) => {
      const body = JSON.parse(String(init?.body));
      expect(body.method).toBe('message/send');
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id: body.id,
        result: {
          status: { state: 'completed' },
          output: { text: 'PydanticAI delivered a result' },
        },
      }), { status: 200 });
    };

    const bridge = new PydanticAIRuntimeBridge({
      name: 'pydantic-agent',
      endpoint: 'http://pydantic.local/a2a',
      fetch: fetchStub,
    });

    const result = await bridge.invoke({ prompt: 'compute value' });
    expect(result.success).toBe(true);
    expect(result.output).toBe('PydanticAI delivered a result');

    const handoff = bridge.asHandoffHandler();
    const handoffResult = await handoff({
      runId: 'run_1',
      agentId: 'source_agent',
      turnIndex: 0,
      targetAgentId: 'pydantic-agent',
      reason: 'Need external computation',
    });
    expect(handoffResult.success).toBe(true);
    expect(handoffResult.output).toBe('PydanticAI delivered a result');
  });
});
