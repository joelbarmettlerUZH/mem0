# Mem0 Search & Query Investigation Report

## Executive Summary
The mem0 search system provides semantic vector search with advanced metadata filtering across Milvus (vector store) and Neo4j (graph store). Session identifiers (user_id, agent_id, run_id) are fundamentally different from custom metadata filters—they are mandatory scoping parameters, while custom filters provide optional refinement.

---

## 1. SEARCH() METHOD COMPLETE ANALYSIS

### Location
**File**: `/home/user/mem0/mem0/memory/main.py`
- **Sync version**: Lines 758-856
- **Async version**: Lines 1807-1868

### Full Method Signature

```python
def search(
    self,
    query: str,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
    rerank: bool = True,
):
```

### Parameter Breakdown

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `query` | str | Required | The search query text |
| `user_id` | str | None | User-level session scoping |
| `agent_id` | str | None | Agent-level session scoping |
| `run_id` | str | None | Run-level session scoping |
| `limit` | int | 100 | Max number of results to return |
| `filters` | Dict | None | Custom metadata filters with advanced operators |
| `threshold` | float | None | Minimum score (0-1) to include results |
| `rerank` | bool | True | Whether to apply reranker if configured |

### Key Requirements & Validation

**Session ID Requirement** (Line 806-807):
```python
if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
    raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")
```

**Critical**: At least one of user_id, agent_id, or run_id MUST be provided. This is a hard requirement.

---

## 2. EXECUTION FLOW - Complete Code Path

### Step 1: Filter Building (Lines 802-815)

```python
# Build effective filters from session IDs
_, effective_filters = _build_filters_and_metadata(
    user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
)

# Check for advanced operators and process if needed
if filters and self._has_advanced_operators(filters):
    processed_filters = self._process_metadata_filters(filters)
    effective_filters.update(processed_filters)
elif filters:
    # Simple filters, merge directly
    effective_filters.update(filters)
```

### Step 2: Parallel Search Execution (Lines 832-843)

```python
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Vector store search (Milvus)
    future_memories = executor.submit(
        self._search_vector_store, query, effective_filters, limit, threshold
    )
    
    # Graph store search (Neo4j) - if enabled
    future_graph_entities = (
        executor.submit(self.graph.search, query, effective_filters, limit) 
        if self.enable_graph else None
    )
    
    # Wait for both to complete
    concurrent.futures.wait(...)
    
    original_memories = future_memories.result()
    graph_entities = future_graph_entities.result() if future_graph_entities else None
```

### Step 3: Vector Store Search Details (Lines 954-990)

```python
def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
    # 1. Embed the query
    embeddings = self.embedding_model.embed(query, "search")
    
    # 2. Search Milvus with filters
    memories = self.vector_store.search(
        query=query, 
        vectors=embeddings, 
        limit=limit, 
        filters=filters  # Already processed by _process_metadata_filters
    )
    
    # 3. Format results
    original_memories = []
    for mem in memories:
        memory_item_dict = MemoryItem(
            id=mem.id,
            memory=mem.payload.get("data", ""),
            hash=mem.payload.get("hash"),
            created_at=mem.payload.get("created_at"),
            updated_at=mem.payload.get("updated_at"),
            score=mem.score,  # Similarity score from Milvus
        ).model_dump()
        
        # Promote session IDs to top level
        for key in ["user_id", "agent_id", "run_id", "actor_id", "role"]:
            if key in mem.payload:
                memory_item_dict[key] = mem.payload[key]
        
        # Include additional metadata
        additional_metadata = {k: v for k, v in mem.payload.items() 
                              if k not in core_and_promoted_keys}
        if additional_metadata:
            memory_item_dict["metadata"] = additional_metadata
        
        # Apply threshold filter
        if threshold is None or mem.score >= threshold:
            original_memories.append(memory_item_dict)
    
    return original_memories
```

### Step 4: Optional Reranking (Lines 845-851)

```python
if rerank and self.reranker and original_memories:
    try:
        reranked_memories = self.reranker.rerank(query, original_memories, limit)
        original_memories = reranked_memories
    except Exception as e:
        logger.warning(f"Reranking failed, using original results: {e}")
```

### Step 5: Return Results (Lines 853-856)

```python
if self.enable_graph:
    return {"results": original_memories, "relations": graph_entities}

return {"results": original_memories}
```

---

## 3. FILTER BUILDING & PROCESSING

### 3a. _build_filters_and_metadata() Function

**Location**: Lines 87-165

**Purpose**: Separates session IDs (user_id, agent_id, run_id) from metadata. Session IDs are MANDATORY and used for query scoping.

```python
def _build_filters_and_metadata(
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
        - base_metadata_template: For storing memories (includes session IDs + input_metadata)
        - effective_query_filters: For querying (includes session IDs + input_filters + actor_id)
    """
    
    # Session IDs are ALWAYS added to filters
    if user_id:
        effective_query_filters["user_id"] = user_id
    if agent_id:
        effective_query_filters["agent_id"] = agent_id
    if run_id:
        effective_query_filters["run_id"] = run_id
    
    # Actor ID is optional and only for querying (not storage)
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id
    
    return base_metadata_template, effective_query_filters
```

### 3b. Session IDs vs Custom Filters - Key Difference

| Aspect | Session IDs (user_id, agent_id, run_id) | Custom Filters |
|--------|------------------------------------------|-----------------|
| **Required** | YES - At least one must be provided | NO - Optional |
| **Purpose** | Scope/partition data for multi-tenant queries | Fine-grained filtering on custom metadata |
| **Storage** | Stored in metadata for every memory | Only if explicitly added via metadata param |
| **Example** | `user_id="alice"` | `filters={"source": "meeting"}` |
| **Querying** | Always included in effective_filters | Must be explicitly provided in search() |

### 3c. Advanced Operators Detection (Lines 927-952)

```python
def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
    """
    Detects if filters use advanced operators that need special processing.
    """
    for key, value in filters.items():
        # Logical operators
        if key in ["AND", "OR", "NOT"]:
            return True
        
        # Comparison operators
        if isinstance(value, dict):
            for op in value.keys():
                if op in ["eq", "ne", "gt", "gte", "lt", "lte", 
                         "in", "nin", "contains", "icontains"]:
                    return True
        
        # Wildcard
        if value == "*":
            return True
    
    return False
```

### 3d. Filter Processing Pipeline (Lines 858-925)

**What Happens**:
1. Detects advanced operators
2. Converts platform operators to universal format
3. Handles logical operators (AND, OR, NOT)
4. Passes to vector store for implementation-specific handling

```python
def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
    processed_filters = {}
    
    def process_condition(key: str, condition: Any) -> Dict[str, Any]:
        if not isinstance(condition, dict):
            # Simple equality: {"key": "value"}
            if condition == "*":
                return {key: "*"}
            return {key: condition}
        
        result = {}
        for operator, value in condition.items():
            # Map operators
            operator_map = {
                "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                "contains": "contains", "icontains": "icontains"
            }
            
            if operator in operator_map:
                result[key] = {operator_map[operator]: value}
            else:
                raise ValueError(f"Unsupported metadata filter operator: {operator}")
        return result
    
    for key, value in metadata_filters.items():
        if key == "AND":
            # Flatten AND conditions into main filter dict
            for condition in value:
                for sub_key, sub_value in condition.items():
                    processed_filters.update(process_condition(sub_key, sub_value))
        
        elif key == "OR":
            # Store as $or for vector store handling
            processed_filters["$or"] = []
            for condition in value:
                or_condition = {}
                for sub_key, sub_value in condition.items():
                    or_condition.update(process_condition(sub_key, sub_value))
                processed_filters["$or"].append(or_condition)
        
        elif key == "NOT":
            # Store as $not for vector store handling
            processed_filters["$not"] = []
            for condition in value:
                not_condition = {}
                for sub_key, sub_value in condition.items():
                    not_condition.update(process_condition(sub_key, sub_value))
                processed_filters["$not"].append(not_condition)
        
        else:
            processed_filters.update(process_condition(key, value))
    
    return processed_filters
```

---

## 4. MILVUS VECTOR STORE SEARCH

### Location
**File**: `/home/user/mem0/mem0/vector_stores/milvus.py`

### Milvus Search Implementation (Lines 142-164)

```python
class MilvusDB(VectorStoreBase):
    def search(self, query: str, vectors: list, limit: int = 5, 
              filters: dict = None) -> list:
        """
        Search for similar vectors in Milvus collection.
        """
        # Create filter string for Milvus
        query_filter = self._create_filter(filters) if filters else None
        
        # Execute search
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[vectors],              # Query vector embeddings
            limit=limit,                 # Max results to return
            filter=query_filter,         # Metadata filter (as Milvus filter string)
            output_fields=["*"],         # Return all fields
        )
        
        # Parse and return results
        result = self._parse_output(data=hits[0])
        return result
```

### Milvus Filter Creation (Lines 100-116)

**CRITICAL LIMITATION**: Milvus filter creation only handles simple equality and basic operators.

```python
def _create_filter(self, filters: dict):
    """
    Create Milvus filter string from filter dictionary.
    
    Example input: {"user_id": "alice", "agent_id": "agent1"}
    Example output: '(metadata["user_id"] == "alice") and (metadata["agent_id"] == "agent1")'
    """
    operands = []
    for key, value in filters.items():
        if isinstance(value, str):
            operands.append(f'(metadata["{key}"] == "{value}")')
        else:
            operands.append(f'(metadata["{key}"] == {value})')
    
    return " and ".join(operands)
```

### Milvus Filter Limitations & Gotchas

1. **Only handles simple equality**: The _create_filter method only creates "==" comparisons
2. **AND only**: Multiple filters are combined with "and" operator
3. **No operator support**: Doesn't handle OR, gt, lt, contains, etc. directly
4. **String quoting**: Strings must be quoted; numbers must not be
5. **Nested metadata access**: Uses `metadata["key"]` syntax
6. **Advanced operators not translated**: If advanced operators like "$or" or "$not" reach Milvus, they may fail silently

### Milvus Parse Output (Lines 118-140)

```python
def _parse_output(self, data: list):
    """Parse Milvus search results"""
    memory = []
    
    for value in data:
        uid, score, metadata = (
            value.get("id"),
            value.get("distance"),  # Similarity distance from Milvus
            value.get("entity", {}).get("metadata"),
        )
        
        memory_obj = OutputData(id=uid, score=score, payload=metadata)
        memory.append(memory_obj)
    
    return memory
```

### OutputData Structure

```python
class OutputData(BaseModel):
    id: Optional[str]      # Memory ID
    score: Optional[float] # Similarity score/distance
    payload: Optional[Dict] # Metadata dictionary
```

### Milvus Collection Schema (Lines 55-83)

```python
def create_col(self, collection_name: str, vector_size: int, metric_type: MetricType):
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
        FieldSchema(name="vectors", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    
    index = self.client.prepare_index_params(
        field_name="vectors",
        metric_type=metric_type,
        index_type="AUTOINDEX",
        index_name="vector_index"
    )
    
    self.client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index
    )
```

**Schema Details**:
- **id**: Primary key, string, max 512 chars
- **vectors**: Float vectors with configurable dimensions
- **metadata**: JSON field containing all metadata (session IDs, custom metadata)
- **Dynamic fields**: Enabled for flexibility

---

## 5. NEO4J GRAPH STORE SEARCH

### Location
**File**: `/home/user/mem0/mem0/memory/graph_memory.py`

### Graph Search Method (Lines 96-130)

```python
class MemoryGraph:
    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.
        
        Returns:
            list: Graph relationships matching the query, reranked by BM25
        """
        # 1. Extract entities from query text using LLM
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        
        # 2. Search graph for similar nodes and relationships
        search_output = self._search_graph_db(
            node_list=list(entity_type_map.keys()), 
            filters=filters
        )
        
        if not search_output:
            return []
        
        # 3. Rerank results using BM25 (not vector-based)
        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] 
            for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)
        
        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(
            tokenized_query, 
            search_outputs_sequence, 
            n=5
        )
        
        search_results = []
        for item in reranked_results:
            search_results.append({
                "source": item[0],
                "relationship": item[1],
                "destination": item[2]
            })
        
        return search_results
```

### Graph Database Query (Lines 271-320)

**File**: `/home/user/mem0/mem0/memory/graph_memory.py`

```python
def _search_graph_db(self, node_list, filters, limit=100):
    """
    Search Neo4j for nodes matching query and their relationships.
    Uses vector similarity on node embeddings.
    """
    result_relations = []
    
    # Build filter clauses for user_id, agent_id, run_id
    node_props = ["user_id: $user_id"]
    if filters.get("agent_id"):
        node_props.append("agent_id: $agent_id")
    if filters.get("run_id"):
        node_props.append("run_id: $run_id")
    node_props_str = ", ".join(node_props)
    
    # For each entity extracted from query
    for node in node_list:
        # Embed the node name
        n_embedding = self.embedding_model.embed(node)
        
        # Cypher query with vector similarity
        cypher_query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        WHERE n.embedding IS NOT NULL
        WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity
        WHERE similarity >= $threshold
        CALL {{
            WITH n
            MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
            RETURN n.name AS source, elementId(n) AS source_id, 
                   type(r) AS relationship, elementId(r) AS relation_id, 
                   m.name AS destination, elementId(m) AS destination_id
            UNION
            WITH n  
            MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
            RETURN m.name AS source, elementId(m) AS source_id, 
                   type(r) AS relationship, elementId(r) AS relation_id, 
                   n.name AS destination, elementId(n) AS destination_id
        }}
        WITH distinct source, source_id, relationship, relation_id, 
             destination, destination_id, similarity
        RETURN source, source_id, relationship, relation_id, 
               destination, destination_id, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        params = {
            "n_embedding": n_embedding,
            "threshold": self.threshold,  # Default 0.7
            "user_id": filters["user_id"],
            "limit": limit,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        
        ans = self.graph.query(cypher_query, params=params)
        result_relations.extend(ans)
    
    return result_relations
```

### Graph Search Key Characteristics

| Feature | Details |
|---------|---------|
| **Entity Extraction** | LLM-based extraction from query text |
| **Vector Search** | Cosine similarity on node embeddings |
| **Filter Application** | Via Cypher MATCH clauses (user_id, agent_id, run_id) |
| **Relationship Query** | Returns both outgoing and incoming relationships |
| **Reranking** | BM25 text-based reranking |
| **Threshold** | Default 0.7, configurable via config |
| **Supported Filters** | Only session IDs (user_id, agent_id, run_id) |

### Graph Search Limitations

1. **No advanced filters**: Does NOT support custom metadata filters
2. **LLM dependency**: Requires extracting entities via LLM first
3. **Relationship-focused**: Returns relationships, not the original memory text
4. **Limited to entity-based queries**: Can't search for arbitrary text patterns

---

## 6. RETURN VALUE STRUCTURE

### Complete Return Format (Lines 853-856)

```python
if self.enable_graph:
    return {"results": original_memories, "relations": graph_entities}

return {"results": original_memories}
```

### Memory Item Structure (From MemoryItem model)

**Location**: `/home/user/mem0/mem0/configs/base.py`

```python
class MemoryItem(BaseModel):
    id: str                           # Unique memory ID
    memory: str                       # The memory text
    hash: Optional[str]               # Hash of memory content
    score: Optional[float]            # Similarity score (0-1)
    created_at: Optional[str]         # ISO timestamp
    updated_at: Optional[str]         # ISO timestamp
    metadata: Optional[Dict[str, Any]] # Additional metadata
    # Promoted fields:
    # user_id, agent_id, run_id, actor_id, role (if in payload)
```

### Example Search Result

```python
{
    "results": [
        {
            "id": "mem_abc123",
            "memory": "User likes to play tennis on weekends",
            "hash": "abc123hash",
            "score": 0.92,
            "created_at": "2024-01-15T10:30:00.000Z",
            "updated_at": None,
            "user_id": "alice",
            "agent_id": "sports-agent",
            "metadata": {
                "category": "hobbies",
                "source": "conversation",
                "timestamp": "2024-01-15"
            }
        },
        {
            "id": "mem_def456",
            "memory": "Prefers outdoor activities",
            "hash": "def456hash",
            "score": 0.87,
            "created_at": "2024-01-16T14:22:00.000Z",
            "updated_at": None,
            "user_id": "alice",
            "metadata": {
                "category": "preferences"
            }
        }
    ],
    "relations": [  # If graph store enabled
        {
            "source": "tennis",
            "relationship": "IS_ACTIVITY_FOR",
            "destination": "alice"
        }
    ]
}
```

### Score Range
- **Minimum**: 0.0 (no similarity)
- **Maximum**: 1.0 (perfect match)
- **Threshold filtering**: Applied after vector search, removes scores < threshold

---

## 7. SEARCH vs GET_ALL COMPARISON

### GET_ALL Method (Lines 653-710)

**Location**: `/home/user/mem0/mem0/memory/main.py`

```python
def get_all(
    self,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
):
```

### Key Differences

| Aspect | search() | get_all() |
|--------|----------|-----------|
| **Query Parameter** | Required (string) | Not used |
| **Embedding** | Query is embedded for semantic search | No embedding (retrieval only) |
| **Scoring** | Semantic similarity score returned | No score (order may vary) |
| **Vector Store Method** | vector_store.search() | vector_store.list() |
| **Speed** | Slower (requires embedding + semantic search) | Faster (direct metadata lookup) |
| **Graph Search** | Semantic search on graph embeddings | get_all on graph (all relationships) |
| **Use Case** | Find semantically similar memories | Retrieve all memories in a scope |
| **Result Type** | Ranked by relevance | Potentially unordered |
| **Reranking** | Supported | Not applicable |
| **Threshold** | Supported | Not applicable |

### Implementation Details

**get_all Vector Store Call** (Lines 711-712):
```python
def _get_all_from_vector_store(self, filters, limit):
    memories_result = self.vector_store.list(filters=filters, limit=limit)
```

**Milvus list() method** (Lines 227-244):
```python
def list(self, filters: dict = None, limit: int = 100) -> list:
    """List all vectors in a collection"""
    query_filter = self._create_filter(filters) if filters else None
    result = self.client.query(
        collection_name=self.collection_name,
        filter=query_filter,
        limit=limit
    )
    # Parse and return results (without scores)
```

---

## 8. FILTER OPERATORS REFERENCE

### Supported Operators

| Operator | Syntax | Example | Purpose |
|----------|--------|---------|---------|
| `eq` | `{"field": {"eq": value}}` | `{"status": {"eq": "active"}}` | Exact match |
| `ne` | `{"field": {"ne": value}}` | `{"status": {"ne": "archived"}}` | Not equal |
| `gt` | `{"field": {"gt": number}}` | `{"score": {"gt": 0.8}}` | Greater than |
| `gte` | `{"field": {"gte": number}}` | `{"priority": {"gte": 5}}` | Greater or equal |
| `lt` | `{"field": {"lt": number}}` | `{"age": {"lt": 30}}` | Less than |
| `lte` | `{"field": {"lte": number}}` | `{"rating": {"lte": 3}}` | Less or equal |
| `in` | `{"field": {"in": [val1, val2]}}` | `{"status": {"in": ["active", "pending"]}}` | In list |
| `nin` | `{"field": {"nin": [val1, val2]}}` | `{"status": {"nin": ["deleted", "archived"]}}` | Not in list |
| `contains` | `{"field": {"contains": "text"}}` | `{"description": {"contains": "meeting"}}` | Contains substring |
| `icontains` | `{"field": {"icontains": "text"}}` | `{"tags": {"icontains": "URGENT"}}` | Contains (case-insensitive) |
| `*` | `{"field": "*"}` | `{"category": "*"}` | Wildcard (field exists) |

### Logical Operators

| Operator | Syntax | Example | Purpose |
|----------|--------|---------|---------|
| `AND` | `{"AND": [filter1, filter2]}` | `{"AND": [{"user_id": "alice"}, {"category": "work"}]}` | All conditions true |
| `OR` | `{"OR": [filter1, filter2]}` | `{"OR": [{"status": "urgent"}, {"priority": {"gte": 9}}]}` | Any condition true |
| `NOT` | `{"NOT": [filter1]}` | `{"NOT": [{"archived": True}]}` | Exclude condition |

### Examples of Filter Combinations

```python
# Simple equality
filters = {"category": "work"}

# Comparison operator
filters = {"priority": {"gte": 7}}

# Multiple conditions (AND)
filters = {
    "AND": [
        {"user_id": "alice"},
        {"category": "work"},
        {"priority": {"gte": 5}}
    ]
}

# Multiple categories (OR)
filters = {
    "AND": [
        {"user_id": "alice"},
        {"OR": [
            {"category": "work"},
            {"category": "personal"}
        ]}
    ]
}

# Complex nested logic
filters = {
    "AND": [
        {
            "OR": [
                {"status": "active"},
                {"status": "pending"}
            ]
        },
        {"priority": {"gte": 7}},
        {"NOT": [{"archived": True}]},
        {"created_date": {"gte": "2024-01-01"}}
    ]
}

# Wildcard - match any value
filters = {
    "AND": [
        {"user_id": "alice"},
        {"run_id": "*"}  # Any run_id
    ]
}
```

---

## 9. SEARCH PARAMETERS DETAILED

### limit Parameter (Lines 765)

**Default**: 100
**Range**: Recommended 1-1000
**Effect**: Maximum number of results to return from vector store
**Before threshold**: Applied at vector store level
**After reranking**: Reranker may reduce result count

### threshold Parameter (Lines 767, 987)

**Default**: None (no filtering)
**Range**: 0.0 to 1.0
**Purpose**: Post-search filtering by similarity score
**Application**: Happens AFTER vector store search, BEFORE reranking

```python
# Applied in _search_vector_store (line 987)
if threshold is None or mem.score >= threshold:
    original_memories.append(memory_item_dict)
```

**Example**:
```python
# Only return results with similarity >= 0.8
results = memory.search(
    "tennis activities",
    user_id="alice",
    threshold=0.8
)
```

### rerank Parameter (Lines 768, 845-851)

**Default**: True
**Type**: bool
**Purpose**: Whether to apply reranker if configured
**Requirement**: Reranker must be configured in config for effect

```python
# Reranking applies after vector search
if rerank and self.reranker and original_memories:
    try:
        reranked_memories = self.reranker.rerank(query, original_memories, limit)
        original_memories = reranked_memories
    except Exception as e:
        logger.warning(f"Reranking failed, using original results: {e}")
```

### Version Parameter

**Note**: The examples show `version="v2"` parameter, but the search() method signature in main.py doesn't include it. This may be a Platform API parameter not in the core Memory class.

---

## 10. LIMITATIONS & GOTCHAS

### Milvus Limitations

1. **Advanced Filter Support**: Milvus' _create_filter() only handles simple equality. Complex operators like `$or`, `$not` may not work as expected.

   ```python
   # This will be converted but may not work correctly in Milvus:
   filters = {"OR": [{"status": "active"}, {"priority": {"gte": 8}}]}
   # Becomes {"$or": [...]} which Milvus doesn't understand
   ```

2. **String vs Number Quoting**: The filter string must properly quote strings and not quote numbers:
   ```python
   # Correct
   '(metadata["user_id"] == "alice")' and '(metadata["priority"] == 8)'
   # Filter creation handles this
   ```

3. **JSON Metadata Access**: All metadata stored in JSON field requires bracket notation:
   ```python
   # Correct filter: metadata["key"] == "value"
   # Not: metadata.key == "value"
   ```

4. **No Nested Filter Translation**: If _process_metadata_filters produces "$or" or "$not", it's passed to Milvus which may not handle it. Only simple "and" combined filters reliably work.

### Neo4j / Graph Store Limitations

1. **No Custom Metadata Filters**: Graph search only supports session ID filters (user_id, agent_id, run_id). Custom metadata filters are IGNORED.

   ```python
   # These custom filters are ignored by graph.search()
   filters = {"category": "work", "priority": {"gte": 5}}
   ```

2. **Entity Extraction First**: Graph search requires entities to be extracted from query via LLM before searching. May fail if LLM extraction is inaccurate.

3. **Fixed BM25 Reranking**: Results are always reranked with BM25 (top 5 only hardcoded). Not configurable.

4. **Relationship-Focused Results**: Returns edges (relationships), not original memories. Must cross-reference with vector search results.

### General Limitations

1. **Session ID Requirement**: Must provide at least one of user_id, agent_id, run_id. Cannot search across all users.

2. **Threshold Applied After Search**: Threshold filtering happens after vector store returns results. If all results are below threshold, returns empty list.

3. **Reranker Errors Silently Degrade**: If reranker fails, original vector search results are returned without warning to caller (only logged).

4. **Filter Processing Overhead**: Advanced filters with nested AND/OR require processing before vector store. Complex filters may impact performance.

5. **Score Range Varies by Metric**: Milvus score/distance interpretation depends on metric_type (L2, COSINE, etc.). Threshold behavior varies.

6. **No Query Timeout**: Large limit values with complex filters may timeout without explicit timeout parameter.

---

## 11. BEST PRACTICES FOR QUERYING

### 1. Session ID Scoping
Always provide explicit user_id, agent_id, or run_id for predictable results:

```python
# Good: Explicit session scoping
results = memory.search("preferences", user_id="alice")

# Also valid: Multiple session IDs
results = memory.search(
    "recent activities",
    user_id="alice",
    agent_id="sports-agent"
)

# Bad: No session ID - will raise error
results = memory.search("tennis")  # ValueError!
```

### 2. Filter Ordering
For Milvus, put indexed fields first (simpler filters before complex):

```python
# Better performance: Simple filters first
filters = {
    "AND": [
        {"user_id": "alice"},           # Simple equality
        {"category": "work"},            # Simple equality
        {"priority": {"gte": 5}},        # Comparison (if supported)
        {"description": {"contains": "meeting"}}  # Text search
    ]
}
```

### 3. Threshold Usage
Use threshold to filter low-confidence results:

```python
# Require high confidence matches
results = memory.search(
    "specific fact",
    user_id="alice",
    threshold=0.85  # Only very similar results
)

# For broader searches, lower threshold
results = memory.search(
    "general preferences",
    user_id="alice",
    threshold=0.6   # More lenient
)
```

### 4. Graph Search Integration
Graph results are separate from memory results. Use both for comprehensive context:

```python
# With graph store enabled
result = memory.search("alice tennis", user_id="alice")

# result contains:
# - results: Direct memory matches
# - relations: Graph relationships (entities and connections)

for memory in result["results"]:
    print(f"Memory: {memory['memory']}")

for relation in result["relations"]:
    print(f"{relation['source']} -{relation['relationship']}-> {relation['destination']}")
```

### 5. Limit Tuning
Start with reasonable limits; adjust based on performance:

```python
# Default - reasonable for most cases
results = memory.search("query", user_id="alice", limit=100)

# For quick checks
results = memory.search("query", user_id="alice", limit=10)

# For comprehensive retrieval
results = memory.search("query", user_id="alice", limit=500)
```

### 6. Custom Metadata Filtering
Add metadata during storage, filter during search:

```python
# Store with metadata
memory.add(
    "User likes skiing",
    user_id="alice",
    metadata={
        "category": "hobbies",
        "activity_type": "winter_sport",
        "confidence": 0.95
    }
)

# Search with custom filters
results = memory.search(
    "outdoor activities",
    user_id="alice",
    filters={
        "AND": [
            {"category": "hobbies"},
            {"activity_type": "winter_sport"},
            {"confidence": {"gte": 0.8}}
        ]
    }
)
```

### 7. Error Handling
Always validate session IDs and handle missing results:

```python
try:
    results = memory.search(
        "query",
        user_id=user_id,
        limit=100,
        threshold=0.7
    )
    
    if not results["results"]:
        print("No memories found matching the query")
    else:
        for mem in results["results"]:
            print(f"Score: {mem['score']:.2f} - {mem['memory']}")

except ValueError as e:
    if "session" in str(e).lower():
        print("Error: Must provide user_id, agent_id, or run_id")
    else:
        raise
except Exception as e:
    logger.error(f"Search failed: {e}")
```

---

## 12. EXECUTION PATH SUMMARY

```
memory.search(query, user_id, filters, limit, threshold, rerank)
    │
    ├─ _build_filters_and_metadata()
    │  └─ Returns: effective_filters (session IDs + custom filters)
    │
    ├─ _has_advanced_operators() 
    │  └─ Check if filters use advanced operators
    │
    ├─ _process_metadata_filters() (if advanced operators)
    │  └─ Convert to universal format, handle AND/OR/NOT
    │
    ├─ Parallel execution:
    │  │
    │  ├─ Vector Store Search (_search_vector_store):
    │  │  ├─ embedding_model.embed(query)
    │  │  ├─ vector_store.search(query, embeddings, limit, filters)
    │  │  │  └─ Milvus._create_filter() → equality filter string
    │  │  │  └─ Milvus.search() → returns scored results
    │  │  ├─ Apply threshold filter (mem.score >= threshold)
    │  │  └─ Format results with promoted fields
    │  │
    │  └─ Graph Store Search (if enabled):
    │     ├─ graph._retrieve_nodes_from_data() → LLM entity extraction
    │     ├─ graph._search_graph_db() → Neo4j vector similarity search
    │     ├─ BM25 reranking
    │     └─ Return top 5 relationships
    │
    ├─ Optional reranking (if enabled)
    │  └─ reranker.rerank(query, memories, limit)
    │
    └─ Return: {"results": memories, "relations": graph_entities?}
```

---

## 13. COMPLETE FILTER EXAMPLES

### Example 1: Simple User Scope
```python
results = memory.search(
    "What do I like?",
    user_id="alice"
)
# Returns all semantically similar memories for user "alice"
```

### Example 2: Multi-Session Scoping
```python
results = memory.search(
    "recent activities",
    user_id="alice",
    agent_id="sports-agent"
)
# Returns memories for alice's sports-agent context
```

### Example 3: Custom Metadata with Operators
```python
results = memory.search(
    "important meetings",
    user_id="alice",
    filters={
        "AND": [
            {"category": "work"},
            {"priority": {"gte": 7}},
            {"sentiment": {"ne": "negative"}}
        ]
    }
)
# Returns high-priority work memories (not negative sentiment)
```

### Example 4: Complex Nested Logic
```python
results = memory.search(
    "activities",
    user_id="alice",
    filters={
        "AND": [
            {
                "OR": [
                    {"type": "indoor"},
                    {"type": "outdoor"}
                ]
            },
            {
                "OR": [
                    {"season": {"in": ["summer", "winter"]}},
                    {"all_year": True}
                ]
            },
            {"difficulty": {"lte": 5}},
            {
                "NOT": [
                    {"restricted": True},
                    {"age_limit": {"lt": 18}}
                ]
            }
        ]
    },
    threshold=0.7,
    limit=20
)
# Returns activities that are seasonal/all-year, not difficult, unrestricted
```

### Example 5: Wildcard with Run Scoping
```python
results = memory.search(
    "query",
    user_id="alice",
    filters={
        "AND": [
            {"run_id": "*"},  # Match any run_id (field must exist)
            {"status": "active"}
        ]
    }
)
# Returns memories across all runs for alice, only active ones
```

### Example 6: Text Search
```python
results = memory.search(
    "preferences",
    user_id="alice",
    filters={
        "AND": [
            {"category": "preferences"},
            {"tags": {"contains": "travel"}}  # Case-sensitive
        ]
    }
)
```

### Example 7: Using get_all for Comparison
```python
# Semantic search (search)
semantic_results = memory.search(
    "What are my hobbies?",
    user_id="alice",
    limit=5
)

# Get all memories (get_all)
all_results = memory.get_all(
    user_id="alice",
    limit=5,
    filters={"category": "hobbies"}
)

# Semantic results are ranked by relevance
# All results are just listed (no ranking)
```

---

## REFERENCES

**Source Files**:
- `/home/user/mem0/mem0/memory/main.py` - Memory class and search logic
- `/home/user/mem0/mem0/vector_stores/milvus.py` - Milvus implementation
- `/home/user/mem0/mem0/memory/graph_memory.py` - Graph search (Neo4j)
- `/home/user/mem0/mem0/configs/base.py` - MemoryItem model
- `/home/user/mem0/docs/open-source/features/metadata-filtering.mdx` - Filter documentation
- `/home/user/mem0/docs/api-reference/memory/search-memories.mdx` - Search API docs

