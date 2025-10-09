# Deep Code Understanding System Design (LLM-Native Search)

I'll design a comprehensive LLM-based system for deep code understanding that uses LLM-based search and reasoning instead of embeddings, addressing the limitations of traditional RAG approaches.

## Why Move Away from Embeddings for Code?

**Fundamental Issues with Embedding-Based RAG for Code:**

1. **Chunking Destroys Semantic Boundaries**: Code has natural semantic units (functions, classes, modules) but arbitrary chunking splits these, losing context
2. **Structural Relationships Lost**: Dependencies, call graphs, and type hierarchies aren't captured in vector space
3. **Syntax vs Semantics Gap**: Similar embeddings might represent syntactically similar but semantically different code
4. **Query-Document Mismatch**: User queries are often high-level ("how does auth work?") while code is low-level implementation
5. **Multi-Hop Reasoning Required**: Understanding code flow requires following relationships, not just similarity matching

**When RAG Might Still Be Useful:**

RAG can work as a **supplementary technique** in specific scenarios:
- **Initial filtering** of massive codebases (millions of files) before LLM analysis
- **Finding similar code patterns** when you have a concrete example
- **Documentation search** where text is naturally chunked into paragraphs
- **Hybrid approach**: Use embeddings for broad recall, LLM for precision ranking

However, for core code understanding and reasoning, **LLM-based semantic search** is superior.

## System Architecture Overview

### Phase 1: Knowledge Base Construction (Indexing Time)

#### 1. Multi-Agent Code Analysis System

**Primary Analysis Agents:**

- **Structural Analysis Agent**: Maps the codebase architecture
- **Semantic Analysis Agent**: Understands code intent and business logic
- **Dependency Analysis Agent**: Traces relationships and data flows
- **Pattern Recognition Agent**: Identifies design patterns and architectural styles
- **Documentation Synthesis Agent**: Generates and consolidates understanding

**Agent Orchestration Algorithm:**

```
For each code unit (file/module/package):
  1. Static Analysis Phase:
     - Parse AST and generate structural metadata
     - Extract symbols, types, and call graphs
     - Identify code boundaries and interfaces
  
  2. Agent Analysis Phase (parallel execution):
     - Structural Agent: Analyze architecture patterns
     - Semantic Agent: Understand business logic
     - Dependency Agent: Map relationships
     - Pattern Agent: Recognize design patterns
  
  3. Synthesis Phase:
     - Aggregate agent findings
     - Resolve conflicts through multi-agent debate
     - Generate hierarchical knowledge entries
     - Create cross-references and search indices
  
  4. Validation Phase:
     - Check consistency across agents
     - Verify dependency accuracy
     - Update global knowledge graph
```

#### 2. Knowledge Base Schema

**Hierarchical Structure:**

```
Codebase Knowledge Base
├── Architectural Layer
│   ├── System-level patterns (microservices, monolith, etc.)
│   ├── Module boundaries and responsibilities
│   ├── Cross-cutting concerns
│   └── Technology stack and frameworks
│
├── Component Layer
│   ├── Classes/Modules with semantic descriptions
│   ├── Public interfaces and contracts
│   ├── Design patterns used
│   └── Key algorithms and their purposes
│
├── Relationship Layer
│   ├── Dependency graph (who uses whom)
│   ├── Data flow graph (how data moves)
│   ├── Call graph (execution paths)
│   └── Inheritance/composition hierarchies
│
├── Implementation Layer
│   ├── Function-level semantics
│   ├── Complex logic explanations
│   ├── Edge cases and error handling
│   └── Performance characteristics
│
└── Meta Layer
    ├── Code quality metrics
    ├── Technical debt indicators
    ├── Testing coverage insights
    └── Change frequency and hotspots
```

**Knowledge Entry Structure:**

```json
{
  "id": "unique_identifier",
  "type": "class|function|module|pattern",
  "name": "ComponentName",
  "location": "file/path:line_range",
  
  "semantic_summary": "LLM-generated description of purpose and behavior",
  "technical_summary": "API signature, parameters, return types",
  
  "relationships": {
    "depends_on": ["component_ids"],
    "depended_by": ["component_ids"],
    "calls": ["function_ids"],
    "called_by": ["function_ids"],
    "implements": ["interface_ids"],
    "overrides": ["method_ids"]
  },
  
  "patterns": ["singleton", "factory", "observer"],
  "complexity_metrics": {"cyclomatic": 8, "cognitive": 12},
  
  "business_context": "Why this exists and what problem it solves",
  "key_algorithms": ["description of important algorithms used"],
  
  "ast_metadata": {
    "node_type": "ClassDeclaration",
    "public_methods": [...],
    "private_methods": [...],
    "fields": [...]
  },
  
  "search_metadata": {
    "keywords": ["authentication", "jwt", "security"],
    "capabilities": ["validates user credentials", "generates tokens"],
    "used_for": ["login", "api authentication", "session management"]
  },
  
  "examples": [
    "Typical usage patterns with context"
  ],
  
  "gotchas": [
    "Common mistakes or surprising behaviors"
  ]
}
```

#### 3. Static Analysis Augmentation

**Abstract Syntax Tree (AST) Analysis:**

**Benefits:**
- Precise structural understanding without execution
- Language-specific semantic extraction
- Reliable symbol resolution and type information
- Basis for control flow and data flow analysis

**Techniques to Employ:**

1. **AST-Based Symbol Extraction**
   ```
   - Extract all definitions (classes, functions, variables)
   - Identify scopes and namespaces
   - Resolve references and bindings
   - Track type annotations and inferred types
   ```

2. **Control Flow Graph (CFG)**
   ```
   - Map execution paths through code
   - Identify branches, loops, and exception handlers
   - Enable reachability analysis
   - Support complexity calculations
   ```

3. **Data Flow Analysis**
   ```
   - Track variable definitions and uses (def-use chains)
   - Identify data dependencies
   - Detect potential bugs (uninitialized variables, type mismatches)
   - Map information flow across components
   ```

4. **Call Graph Construction**
   ```
   - Build static call relationships
   - Handle polymorphism and dynamic dispatch
   - Identify entry points and leaf functions
   - Support impact analysis
   ```

5. **Type System Analysis**
   ```
   - Extract type hierarchies
   - Resolve generic/template types
   - Track type constraints and invariants
   - Enable type-aware search
   ```

6. **Program Dependence Graph (PDG)**
   ```
   - Combine control and data dependencies
   - Support slicing (forward/backward)
   - Enable better change impact analysis
   ```

**Integration with LLM Analysis:**

```
Static Analysis Output → LLM Agent Input
- AST provides structure → LLM adds semantic meaning
- CFG shows paths → LLM explains business logic
- Call graph shows connections → LLM describes purposes
- Type info constrains → LLM interprets intent

LLM Output → Enriches Static Analysis
- Semantic summaries label AST nodes
- Business context explains why structures exist
- Pattern recognition names architectural choices
```

### Phase 2: Query Time System

#### LLM-Based Search Architecture

**Core Principle**: Use LLM as the search and reasoning engine, not just for generation.

**Multi-Stage LLM Search Process:**

```
Stage 1: Query Understanding & Planning
├── LLM analyzes user query
├── Extracts key concepts, entities, and intent
├── Generates search strategy
└── Identifies relevant knowledge layers to query

Stage 2: Hierarchical Knowledge Retrieval
├── Start at appropriate abstraction level
├── LLM evaluates KB summaries at that level
├── Selects relevant components/modules
└── Drills down to more detailed levels as needed

Stage 3: Relationship Traversal
├── LLM decides which relationships to follow
├── Explores dependencies, calls, data flows
├── Builds comprehensive context
└── Stops when sufficient understanding achieved

Stage 4: Synthesis & Response
├── LLM integrates all gathered information
├── Reasons about architectural implications
├── Generates coherent explanation
└── Provides actionable insights
```

#### Query-Time Tools for LLM Agent

**1. LLM-Based Semantic Search Tools**

```python
search_components_by_description(
    query: str,
    layer: str,  # "architecture|component|implementation"
    max_candidates: int
) -> List[ComponentSummary]
  """
  Returns a batch of component summaries from the KB.
  LLM then evaluates each to determine relevance.
  
  Process:
  1. Retrieve candidate components from specified layer
  2. LLM reads summaries and decides which are relevant
  3. Returns ranked list based on LLM reasoning
  """

progressive_search(
    query: str,
    start_layer: str,
    max_depth: int
) -> SearchPath
  """
  LLM-guided progressive refinement:
  1. Start at high level (architecture)
  2. LLM identifies relevant areas
  3. Drill down to components
  4. Continue until sufficient detail found
  
  This avoids the chunking problem by respecting
  natural code boundaries at each level.
  """

find_by_capability(description: str) -> List[Component]
  """
  LLM evaluates KB entries to find components
  that provide described capability.
  
  Better than embeddings because LLM can reason about:
  - Semantic equivalence ("authenticate" vs "verify user")
  - Indirect capabilities (uses AuthService → has auth capability)
  - Context-dependent matching
  """
```

**2. Structural Navigation Tools**

```python
get_component_details(component_id: str, depth: int)
  # Retrieves full knowledge entry with relationships
  
traverse_dependency_graph(start_id: str, direction: str, max_depth: int)
  # Walks dependency relationships
  # direction: "upstream" | "downstream" | "both"
  
get_call_chain(from_id: str, to_id: str)
  # Finds execution path between components
  
get_data_flow(variable: str, scope_id: str)
  # Traces how data moves through code
```

**3. Architectural Query Tools**

```python
get_architecture_overview(scope: str)
  # Returns high-level system structure
  
find_layer_components(layer: str)
  # Lists all components in architectural layer
  # layers: "presentation", "business", "data", etc.
  
identify_cross_cutting_concerns()
  # Finds patterns that span multiple components
  
get_module_boundaries()
  # Shows encapsulation and interfaces
```

**4. LLM-Based Pattern Detection**

```python
find_design_patterns_llm(pattern_description: str) -> List[Component]
  """
  LLM reads component descriptions and code snippets
  to identify patterns matching the description.
  
  More flexible than keyword matching:
  - Can recognize pattern variations
  - Understands intent, not just structure
  - Can explain why it's a match
  """

detect_code_smells_llm(scope: str) -> List[Issue]
  """
  LLM analyzes components for:
  - Anti-patterns
  - Code smells
  - Architectural issues
  
  Provides reasoning for each finding.
  """

find_similar_implementations_llm(example_id: str) -> List[Component]
  """
  LLM reads example component and finds similar ones by:
  - Functional similarity (not just syntactic)
  - Problem-solving approach
  - Architectural role
  """
```

**5. Impact Analysis Tools**

```python
analyze_change_impact(component_id: str)
  # Predicts what would be affected by changes
  
find_dependents(component_id: str, depth: int)
  # Lists all components that depend on this
  
get_test_coverage_info(component_id: str)
  # Shows testing related to component
```

**6. Code Retrieval Tools**

```python
get_source_code(component_id: str, include_context: bool)
  # Retrieves actual implementation
  # include_context adds surrounding code
  
get_code_with_annotations(component_id: str)
  # Returns code with AST-based semantic annotations
  
get_usage_examples(component_id: str)
  # Shows real usage from codebase
```

**7. Historical and Quality Tools**

```python
get_change_history(component_id: str)
  # Shows evolution and modification patterns
  
get_complexity_metrics(scope: str)
  # Returns various complexity measures
  
identify_hotspots()
  # Finds frequently changed or complex areas
```

**8. LLM-Based Batch Evaluation Tools**

```python
evaluate_components_batch(
    components: List[ComponentSummary],
    criteria: str,
    task: str
) -> List[EvaluationResult]
  """
  Core tool for LLM-based search.
  
  Given a batch of component summaries, LLM evaluates each
  against criteria for the task.
  
  Example:
    components = get_all_components_in_layer("business")
    results = evaluate_components_batch(
      components,
      criteria="handles user authentication",
      task="find authentication components"
    )
  
  LLM returns structured evaluation:
  {
    "component_id": "AuthService",
    "relevance_score": "high|medium|low",
    "reasoning": "Manages JWT tokens and validates credentials",
    "related_capabilities": ["authorization", "session management"]
  }
  """

compare_and_rank_components(
    components: List[Component],
    comparison_criteria: str
) -> RankedList
  """
  LLM compares multiple components and ranks them
  based on criteria like:
  - Best implementation of pattern
  - Most appropriate for use case
  - Code quality and maintainability
  """
```

#### Query-Time Agent Algorithm with LLM Search

```
User Query → Query Understanding Phase
├── LLM analyzes query intent and extracts key information
├── Identifies query type: understand | debug | implement | refactor | find
├── Determines appropriate starting point in KB hierarchy
└── Generates initial search strategy

↓

Planning Phase (LLM Chain-of-Thought)
├── "To answer this, I need to find components that..."
├── "I should start at [layer] because..."
├── "I'll need to follow [relationships] to understand..."
└── "Potential challenges: [identifies ambiguities]"

↓

LLM-Guided Search Phase
├── Progressive Refinement:
│   ├── Get candidates from appropriate KB layer
│   ├── LLM evaluates summaries in batch
│   ├── Selects most relevant (with reasoning)
│   └── Drills down or pivots based on findings
│
├── Relationship Exploration:
│   ├── For each relevant component found
│   ├── LLM decides which relationships to follow
│   ├── Retrieves related components
│   └── Evaluates their relevance to query
│
└── Iterative Deepening:
    ├── Start with high-level understanding
    ├── Identify gaps in knowledge
    ├── Make targeted retrieval of details
    └── Continue until sufficient for answer

↓

Reasoning Phase
├── LLM synthesizes all gathered information
├── Identifies patterns and relationships
├── Resolves conflicts or ambiguities
├── Considers architectural implications
└── Evaluates trade-offs

↓

Response Generation Phase
├── Structure explanation hierarchically
├── Include relevant code examples
├── Provide architectural context
├── Suggest best practices
└── Cite specific KB components used

↓

Validation Phase (if applicable)
├── Verify consistency with codebase patterns
├── Check for unintended consequences
└── Propose testing strategies
```

### Detailed Search Examples

**Example 1: Finding Authentication Components**

```
User: "How does authentication work in this codebase?"

LLM Search Process:

Step 1: Query Understanding
LLM thinks: "User wants to understand authentication mechanism.
            Need to find: auth-related components, flow, and patterns."

Step 2: Start at Architecture Layer
Tool call: get_architecture_overview("security")
Returns: System uses JWT-based authentication with middleware pattern

Step 3: Find Relevant Components
Tool call: get_all_components_in_layer("security")
Returns: [AuthMiddleware, TokenService, UserValidator, SessionManager, ...]

LLM evaluates each summary:
evaluate_components_batch(
  components,
  criteria="involved in user authentication process",
  task="map authentication flow"
)

LLM reasoning:
- AuthMiddleware: HIGH relevance - entry point for auth requests
- TokenService: HIGH - generates and validates JWT tokens
- UserValidator: HIGH - verifies user credentials
- SessionManager: MEDIUM - manages sessions but not core auth
- [filters out 15 other low-relevance components]

Step 4: Follow Relationships
For each high-relevance component:
  - traverse_dependency_graph(component.id, "both", 2)
  - LLM evaluates dependencies for relevance

Step 5: Build Complete Picture
LLM synthesizes:
"Authentication flow:
 1. Request hits AuthMiddleware
 2. Extracts token from header
 3. TokenService validates token
 4. If no token, redirects to login
 5. Login uses UserValidator to check credentials
 6. On success, TokenService generates new JWT
 7. Token stored in response header"

Response includes: component references, code snippets, flow diagram
```

**Example 2: Implementing New Feature**

```
User: "I need to add rate limiting to the API. Show me where and how."

LLM Search Process:

Step 1: Query Understanding
LLM thinks: "User wants to implement rate limiting. Need to:
            1. Check if similar functionality exists
            2. Find appropriate integration point
            3. Identify pattern to follow"

Step 2: Search for Existing Patterns
Tool call: find_by_capability("throttling or rate limiting or request limiting")

LLM evaluates all components for rate limiting capability
Returns: None found (or finds similar feature if exists)

Step 3: Understand API Architecture
Tool call: get_architecture_overview("api_layer")
Returns: Uses Express.js with middleware pattern, request pipeline

Step 4: Find Middleware Examples
Tool call: progressive_search(
  query="middleware implementations",
  start_layer="component",
  max_depth=2
)

LLM finds: [AuthMiddleware, LoggingMiddleware, CorsMiddleware, ...]

Step 5: Select Best Example
Tool call: compare_and_rank_components(
  middlewares,
  criteria="best pattern for adding request-level functionality"
)

LLM reasoning:
"AuthMiddleware is best example because:
 - Intercepts requests before route handlers
 - Can reject requests (similar to rate limit rejection)
 - Clean error handling pattern
 - Well-tested structure"

Step 6: Analyze Integration Points
Tool call: get_component_details("APIRouter", depth=2)
LLM identifies where middleware is registered

Step 7: Generate Implementation Plan
LLM synthesizes complete plan with:
- Where to create RateLimitMiddleware
- Structure based on AuthMiddleware pattern
- Integration point in middleware chain
- Configuration approach
- Testing strategy
- Code skeleton
```

**Example 3: Debugging with LLM Search**

```
User: "Orders are timing out after payment succeeds. Help me debug."

LLM Search Process:

Step 1: Query Understanding
LLM thinks: "Timing out suggests async operation issue.
            Need to trace payment → order completion flow.
            Look for: async operations, error handling, timing issues"

Step 2: Find Relevant Components
Tool call: search_components_by_description(
  query="payment processing and order completion",
  layer="component",
  max_candidates=20
)

LLM evaluates summaries:
- PaymentProcessor: HIGH - handles payment operations
- OrderService: HIGH - manages order lifecycle
- PaymentWebhook: MEDIUM - async payment notifications
- OrderQueue: HIGH - async order processing

Step 3: Trace Execution Path
Tool call: get_call_chain("PaymentProcessor.processPayment", "OrderService.complete")

Returns: Multiple possible paths found

LLM analyzes paths:
"Path 1: PaymentProcessor → OrderService.complete (synchronous)
 Path 2: PaymentProcessor → PaymentWebhook → OrderQueue → OrderService.complete (async)
 
 The async path explains the timeout!"

Step 4: Investigate Async Path
Tool call: get_component_details("OrderQueue", depth=2)

LLM reads implementation details:
"OrderQueue uses message queue with 30s timeout.
 OrderService.complete has no timeout handling.
 If order processing > 30s, webhook times out but order still processes."

Step 5: Check Error Handling
Tool call: get_code_with_annotations("OrderService.complete")

LLM analyzes:
"No try-catch around slow database operations.
 No timeout configuration.
 No retry logic if initial webhook times out."

Step 6: Identify Root Cause
LLM synthesizes:
"Root cause: Webhook timeout (30s) < Order processing time
 
 Issues:
 1. Synchronous DB operations in webhook handler
 2. No timeout handling
 3. No idempotency for retries
 4. Missing monitoring/logging for slow operations"

Step 7: Propose Solution
LLM provides:
- Immediate fix: increase webhook timeout
- Better fix: use async job processor
- Code changes needed with examples
- Testing approach
- Monitoring to add
```

## Advanced Features

### 1. LLM-Based Incremental Search Refinement

```python
class IncrementalSearchSession:
    """
    Maintains search state across multiple LLM calls.
    Allows progressive refinement without losing context.
    """
    
    def __init__(self, user_query: str):
        self.query = user_query
        self.explored_components = set()
        self.relevant_findings = []
        self.search_path = []
        
    def next_search_iteration(self):
        """
        LLM decides next search action based on:
        - What's been found so far
        - What gaps remain
        - Which paths look promising
        """
        prompt = f"""
        Original query: {self.query}
        
        Already explored: {self.explored_components}
        Relevant findings so far: {self.relevant_findings}
        
        What should I search for next to better answer the query?
        Consider:
        - Are there gaps in understanding?
        - Should I explore related components?
        - Do I have enough detail, or need to drill down?
        - Should I pivot to a different area?
        """
        # LLM returns next search action
```

### 2. Multi-Hop Reasoning with LLM

```python
def multi_hop_search(query: str, max_hops: int = 5):
    """
    LLM performs multi-hop reasoning across KB.
    
    Example for "How does data flow from API to database?":
    
    Hop 1: Find API entry points
    Hop 2: Follow calls to business layer
    Hop 3: Trace to data access layer
    Hop 4: Identify database operations
    Hop 5: Validate complete path
    
    At each hop, LLM decides:
    - Which relationships to follow
    - Whether to continue or stop
    - If answer is complete
    """
    current_context = []
    
    for hop in range(max_hops):
        # LLM decides next hop
        next_action = llm_plan_next_hop(query, current_context, hop)
        
        if next_action.stop:
            break
            
        # Execute search
        results = execute_search(next_action.search_params)
        
        # LLM evaluates results
        relevant = llm_filter_results(results, query, current_context)
        current_context.extend(relevant)
    
    return llm_synthesize_answer(query, current_context)
```

### 3. Confidence Scoring and Gap Detection

```python
def search_with_confidence(query: str) -> SearchResult:
    """
    LLM evaluates its own search results for completeness.
    """
    results = perform_llm_search(query)
    
    confidence_analysis = llm_evaluate_confidence(query, results)
    """
    LLM assesses:
    - Did I find all relevant components?
    - Are there gaps in understanding?
    - What uncertainties remain?
    - Should I search more or differently?
    
    Returns structured confidence score with reasoning
    """
    
    if confidence_analysis.confidence < 0.7:
        # LLM suggests additional searches
        refinement_plan = llm_plan_refinement(
            query, 
            results, 
            confidence_analysis.gaps
        )
        # Execute additional searches
```

### 4. Adaptive Search Strategy

```python
class AdaptiveSearchPlanner:
    """
    LLM learns from search patterns to improve strategy.
    """
    
    def plan_search(self, query: str, codebase_context: dict):
        """
        LLM considers:
        - Codebase size and structure
        - Query complexity
        - Available tools
        - Time/cost constraints
        
        Generates optimal search strategy:
        - Breadth-first vs depth-first
        - Top-down vs bottom-up
        - Relationship-focused vs component-focused
        """
        
        strategy_prompt = f"""
        Query: {query}
        Codebase has: {codebase_context['num_components']} components
        Architecture: {codebase_context['architecture_type']}
        
        Design an efficient search strategy. Consider:
        1. Where to start (which layer/component)
        2. How to traverse (BFS/DFS, relationships to follow)
        3. When to stop (sufficient information criteria)
        4. Estimated number of LLM calls needed
        
        Provide step-by-step search plan.
        """
        
        return llm_generate_search_plan(strategy_prompt)
```

## Comparison: LLM Search vs RAG/Embeddings

| Aspect | Embedding-Based RAG | LLM-Based Search |
|--------|---------------------|------------------|
| **Semantic Understanding** | Limited to vector similarity | Full natural language understanding |
| **Structural Awareness** | None (flat chunks) | Respects code hierarchy and boundaries |
| **Relationship Traversal** | Cannot follow relationships | Explicitly traces dependencies and calls |
| **Multi-Hop Reasoning** | Not supported | Natural through iterative search |
| **Context Preservation** | Lost at chunk boundaries | Maintains semantic units intact |
| **Query Flexibility** | Requires query reformulation | Understands intent directly |
| **Precision** | False positives from similarity | Reasoning-based relevance |
| **Explainability** | Opaque similarity scores | Clear reasoning for each decision |
| **Cost** | Lower (embedding lookups) | Higher (multiple LLM calls) |
| **Latency** | Lower (vector search is fast) | Higher (LLM inference time) |

## When to Use Each Approach

**Use LLM-Based Search (Recommended for Code):**
- Deep understanding required
- Complex architectural queries
- Debugging and root cause analysis
- Implementation planning
- When accuracy matters more than speed
- Medium-sized codebases (< 1M LOC)

**Use Hybrid (RAG + LLM):**
- Initial filtering on massive codebases (> 1M LOC)
- Finding similar code examples
- When latency is critical
- Cost-constrained environments

**Use Pure RAG:**
- Documentation search (not code)
- Simple keyword-based queries
- When "good enough" is acceptable

## Technical Stack Recommendations

**Static Analysis:**
- Tree-sitter (universal parsing)
- Language-specific tools (Roslyn for C#, JavaParser, rust-analyzer)
- LLVM for lower-level analysis

**Knowledge Storage:**
- Graph DB (Neo4j) for relationships and structure
- Document DB (MongoDB/PostgreSQL) for component entries
- Redis for caching LLM search results
- (Optional) Vector DB only for initial filtering on huge codebases

**LLM Infrastructure:**
- Function calling for tool use
- Structured output for consistent results
- Agent framework (LangGraph, Semantic Kernel)
- Prompt caching for efficiency
- Batch API for cost optimization

**Search Optimization:**
- Cache common search patterns
- Pre-compute frequently accessed paths
- Parallel LLM calls for batch evaluation
- Progressive result streaming

## Conclusion

LLM-based search provides superior code understanding by:
1. **Respecting semantic boundaries** instead of arbitrary chunking
2. **Following relationships** explicitly through the codebase
3. **Multi-hop reasoning** to trace complex flows
4. **Explainable relevance** through natural language reasoning
5. **Adaptive strategies** that adjust to different query types

While more expensive than embeddings, the accuracy and depth of understanding make it worthwhile for professional development tools where correctness is critical. For massive codebases, a hybrid approach using embeddings for initial filtering and LLM for precision ranking offers a good balance.