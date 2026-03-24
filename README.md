# @veridex/agents-adapters

Import, export, and live-bridge package for bringing existing agent systems into the Veridex runtime.

## Status

`@veridex/agents-adapters` is usable today and already covers import/export plus live runtime bridges, but this area will keep expanding as more real-world migration paths get exercised.

## What This Package Does

Use `@veridex/agents-adapters` when you want to:

- import OpenAI Agents SDK definitions into Veridex
- import LangGraph or PydanticAI-style agent definitions
- turn OpenAPI operations into tool contracts
- export Veridex definitions back into other framework shapes
- invoke external runtimes live instead of rewriting them first

This package exists to reduce rewrites and make migration incremental.

## Installation

```bash
npm install @veridex/agents-adapters @veridex/agents
```

## Package Surface

| Export | Role |
|---|---|
| `OpenAIAgentsAdapter` | Import/export OpenAI Agents SDK-style definitions |
| `LangGraphAdapter` | Import/export LangGraph-style definitions |
| `PydanticAIAdapter` | Import/export PydanticAI-style definitions |
| `OpenAPIImporter` | Convert OpenAPI operations into Veridex tools |
| `OpenAIRuntimeBridge` | Invoke a live OpenAI Agents SDK runtime |
| `LangGraphRuntimeBridge` | Invoke a deployed LangGraph assistant/thread/run surface |
| `PydanticAIRuntimeBridge` | Invoke a live PydanticAI endpoint or A2A-facing runtime |

## Import and Export Adapters

### OpenAI Agents SDK

```ts
import { OpenAIAgentsAdapter } from '@veridex/agents-adapters';

const adapter = new OpenAIAgentsAdapter();

const imported = adapter.importAgent({
  name: 'Support Agent',
  instructions: 'Handle support requests safely.',
  model: 'gpt-4o-mini',
  tools: [
    {
      type: 'function',
      function: {
        name: 'lookup_order',
        description: 'Look up an order by id',
        parameters: {
          type: 'object',
          properties: {
            orderId: { type: 'string' },
          },
          required: ['orderId'],
        },
      },
    },
  ],
});

console.log(imported.definition);
console.log(imported.warnings);
console.log(imported.unsupportedFeatures);
```

Imported tools use safe stub executors until you bind them to a real executor or a live bridge.

### OpenAPI to tools

```ts
import { OpenAPIImporter } from '@veridex/agents-adapters';

const importer = new OpenAPIImporter({
  filterTags: ['orders'],
});

const { tools, warnings, apiInfo } = importer.importTools(openApiSpec);
console.log(apiInfo.title, tools.length, warnings);
```

Each operation becomes a Veridex tool contract with:

- safety class inferred from the HTTP method
- schema preserved from parameters and request body
- response schema preserved when possible
- OpenAPI metadata attached to the tool

## Live Runtime Bridges

Adapters are for definitions. Bridges are for real execution.

### OpenAI Agents SDK bridge

```ts
import { OpenAIRuntimeBridge } from '@veridex/agents-adapters';

const bridge = new OpenAIRuntimeBridge({
  name: 'existing-openai-agent',
  agent: existingOpenAIAgent,
  runner: {
    run: async (agent, input, options) => existingRunner.run(agent, input, options),
  },
});

const result = await bridge.invoke({
  prompt: 'Summarize the escalation policy',
});

console.log(result.output);
```

### Use a bridge as a Veridex tool

```ts
const remoteTool = bridge.asTool({
  name: 'delegate_to_existing_agent',
  safetyClass: 'network',
});
```

### Use a bridge as a handoff target

```ts
const handoffHandlers = {
  external_support_agent: bridge.asHandoffHandler(),
};
```

This is the recommended migration path when you want Veridex governance, approvals, traces, or budgets around an external runtime without a full rewrite on day one.

## Migration Strategy

The best way to adopt this package is usually:

1. import or bridge the existing agent
2. run it through Veridex for tracing and policy
3. replace stub executors with real executors or tools
4. move sensitive or payment-related behavior into Veridex-native contracts over time

That keeps migration practical instead of forcing a big-bang rewrite.

## What Gets Lost Across Frameworks

No framework adapter is perfect. Common gaps include:

- memory configuration
- hook semantics
- framework-specific workflow graphs
- native approval or governance behavior
- custom runtime middleware

The adapters preserve metadata and emit warnings so these gaps stay visible.

## Related Packages

| Package | Use it with |
|---|---|
| `@veridex/agents` | Required target runtime for imported tools and definitions |
| `@veridex/agents-openclaw` | Use when the external system is OpenClaw or Pi rather than OpenAI, LangGraph, or PydanticAI |
| `@veridex/agents-control-plane` | Add governance, approvals, and audit around bridged runtimes |

## Known Rough Edges

- Import/export support is necessarily lossy across frameworks with different abstractions.
- Live bridge coverage will keep improving as more real deployments are integrated.
- Some adapters intentionally produce stub executors so imports fail closed rather than pretending to be production-ready.

Contributions are especially welcome around migration examples, richer adapter fidelity, and more real deployment fixtures in tests.
