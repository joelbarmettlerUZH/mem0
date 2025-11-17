# Mem0 Customization Guide: Neo4j + Milvus + OpenAI

This guide provides comprehensive customization points for mem0 using Neo4j (graph storage), Milvus (vector storage), and OpenAI (LLM + embeddings).

---

## 1. MemoryConfig Class - Core Configuration

**File**: `/home/user/mem0/mem0/configs/base.py` (Lines 30-67)

The `MemoryConfig` class is the main configuration object that controls all aspects of mem0. It has the following customizable parameters:

### Parameters:

```python
from mem0.configs.base import MemoryConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.graphs.configs import GraphStoreConfig, Neo4jConfig
from mem0.configs.vector_stores.milvus import MilvusDBConfig
from mem0.configs.rerankers.config import RerankerConfig

# Minimal configuration with Neo4j, Milvus, and OpenAI
config = MemoryConfig(
    # Vector Store Configuration
    vector_store=VectorStoreConfig(
        provider="milvus",
        config={
            "url": "http://localhost:19530",
            "token": None,  # Optional, use for Zilliz
            "collection_name": "mem0",
            "embedding_model_dims": 1536,  # Match OpenAI embedding dimensions
            "metric_type": "L2",  # Options: L2, IP (Inner Product), COSINE, HAMMING, JACCARD
            "db_name": "",  # Optional database name
        }
    ),
    
    # LLM Configuration (OpenAI)
    llm=LlmConfig(
        provider="openai",
        config={
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "api_key": "your-openai-api-key",  # or set OPENAI_API_KEY env var
            "max_tokens": 2000,
            "top_p": 0.1,
            "top_k": 1,
            "enable_vision": False,  # Set True for vision capabilities
            "vision_details": "auto",  # Options: "auto", "low", "high"
            "openai_base_url": None,  # Override OpenAI base URL if needed
        }
    ),
    
    # Embeddings Configuration (OpenAI)
    embedder=EmbedderConfig(
        provider="openai",
        config={
            "model": "text-embedding-3-small",
            "api_key": "your-openai-api-key",
            "embedding_dims": 1536,  # Dimensions for text-embedding-3-small
            "openai_base_url": None,  # Override OpenAI base URL if needed
        }
    ),
    
    # Graph Store Configuration (Neo4j)
    graph_store=GraphStoreConfig(
        provider="neo4j",
        config=Neo4jConfig(
            url="bolt://localhost:7687",  # Neo4j connection URL
            username="neo4j",
            password="your-password",
            database="neo4j",  # Default database name
            base_label=True,  # Use __Entity__ base label for all entities
        ),
        # Optional: Use different LLM for graph operations
        llm=None,  # Defaults to main LLM if None
        # Custom prompt for entity extraction
        custom_prompt=None,
        # Threshold for embedding similarity when matching nodes
        # Range: 0.0 to 1.0 (higher = stricter matching)
        threshold=0.7,
    ),
    
    # Optional: Reranker configuration
    reranker=RerankerConfig(
        provider="cohere",
        config={
            "model": "rerank-english-v2.0",
            "api_key": "your-cohere-api-key",
        }
    ),
    
    # API version
    version="v1.1",
    
    # Custom prompts for fact extraction
    custom_fact_extraction_prompt=None,  # Override default fact extraction prompt
    custom_update_memory_prompt=None,  # Override default memory update prompt
    
    # History database path
    history_db_path="~/.mem0/history.db",
)

from mem0.memory import Memory
memory = Memory(config=config)
```

### Configuration Options Breakdown:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `vector_store` | VectorStoreConfig | Vector store configuration | Qdrant |
| `llm` | LlmConfig | LLM configuration | OpenAI |
| `embedder` | EmbedderConfig | Embeddings configuration | OpenAI |
| `graph_store` | GraphStoreConfig | Graph store configuration | Neo4j |
| `history_db_path` | str | Path to SQLite history database | ~/.mem0/history.db |
| `reranker` | RerankerConfig | Optional reranker configuration | None |
| `version` | str | API version | v1.1 |
| `custom_fact_extraction_prompt` | str | Custom prompt for fact extraction | None |
| `custom_update_memory_prompt` | str | Custom prompt for memory updates | None |

---

## 2. Custom Prompts - Fact Extraction & Memory Updates

**Files**: 
- `/home/user/mem0/mem0/configs/prompts.py`
- `/home/user/mem0/mem0/memory/main.py` (Lines 176-177, 425-426, 498)

### A. Custom Fact Extraction Prompt

Control how facts are extracted from conversations:

```python
from mem0.memory import Memory
from mem0.configs.base import MemoryConfig

custom_extraction_prompt = """You are a specialized fact extractor for a healthcare system.
Extract only medical and health-related facts from the conversation.

Categories to extract:
1. Medical conditions
2. Medications
3. Allergies
4. Symptoms
5. Health preferences

Return as JSON: {"facts": [...]}"""

config = MemoryConfig(
    custom_fact_extraction_prompt=custom_extraction_prompt
)
memory = Memory(config=config)
```

**Location in code**: Line 425-426 in `/home/user/mem0/mem0/memory/main.py`
- If `custom_fact_extraction_prompt` is provided, it overrides the default prompt
- Otherwise, it uses `USER_MEMORY_EXTRACTION_PROMPT` or `AGENT_MEMORY_EXTRACTION_PROMPT`

### B. Custom Memory Update Prompt

Control how existing memories are updated:

```python
custom_update_prompt = """You are a memory manager for a personalized healthcare assistant.
Decide whether to ADD, UPDATE, DELETE, or keep NONE for each fact.

Special rules:
1. Medical facts override previous versions
2. Medication changes require UPDATE
3. New allergies trigger immediate storage

Return JSON with memory updates as specified in the format."""

config = MemoryConfig(
    custom_update_memory_prompt=custom_update_prompt
)
```

**Location in code**: Line 498 in `/home/user/mem0/mem0/memory/main.py`
- Used in the `get_update_memory_messages()` function
- Controls ADD, UPDATE, DELETE, NONE operations on memories

### C. Built-in Prompts Available

```python
from mem0.configs.prompts import (
    FACT_RETRIEVAL_PROMPT,  # Legacy prompt for all types
    USER_MEMORY_EXTRACTION_PROMPT,  # For user-generated facts only
    AGENT_MEMORY_EXTRACTION_PROMPT,  # For agent-generated facts only
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,  # For procedural memory creation
    DEFAULT_UPDATE_MEMORY_PROMPT,  # For memory updates
    MEMORY_ANSWER_PROMPT,  # For answering questions using memories
)
```

---

## 3. Neo4j Graph Store Configuration

**Files**:
- `/home/user/mem0/mem0/graphs/configs.py` (Lines 8-24)
- `/home/user/mem0/mem0/memory/graph_memory.py`

### Neo4j Connection Parameters

```python
from mem0.graphs.configs import Neo4jConfig, GraphStoreConfig
from mem0.configs.base import MemoryConfig

neo4j_config = GraphStoreConfig(
    provider="neo4j",
    config=Neo4jConfig(
        url="bolt://localhost:7687",  # Connection URL (bolt://, neo4j://, etc.)
        username="neo4j",
        password="your-password",
        database="neo4j",  # Database name
        base_label=True,  # Whether to use __Entity__ as base node label
    ),
    # Custom prompt for entity extraction from text
    custom_prompt="Extract entities related to organizational structure",
    # Threshold for embedding similarity matching (0.0 to 1.0)
    threshold=0.7,  # Default: 0.7
    # Optional: Use different LLM for graph operations
    llm=None,  # Uses main config LLM if None
)

config = MemoryConfig(graph_store=neo4j_config)
memory = Memory(config=config)
```

### Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `url` | str | Neo4j connection URL | Required |
| `username` | str | Neo4j username | Required |
| `password` | str | Neo4j password | Required |
| `database` | str | Database name | "neo4j" |
| `base_label` | bool | Use __Entity__ base label | None |
| `threshold` | float | Embedding similarity threshold (0.0-1.0) | 0.7 |
| `custom_prompt` | str | Custom entity/relation extraction prompt | None |

### Threshold Configuration

**Location**: `/home/user/mem0/mem0/memory/graph_memory.py` (Lines 73-74, 308, 438-439)

The `threshold` parameter controls how similar node embeddings must be to match:

```python
# Low threshold (0.5-0.7): More lenient, matches similar but different entities
threshold=0.5  # Matches entities with cosine similarity >= 0.5

# Medium threshold (0.7-0.85): Balanced matching
threshold=0.7  # Default - good for general use

# High threshold (0.85-1.0): Strict matching, only exact matches
threshold=0.9  # Very strict - only nearly identical entities
```

**Usage in code** (Line 290 in graph_memory.py):
```
WHERE similarity >= $threshold
```

### Custom Graph Extraction Prompt

```python
custom_graph_prompt = """Extract relationships between organizations, teams, and projects.
Relationships: MANAGES, BELONGS_TO, COLLABORATES_WITH, LEADS"""

graph_config = GraphStoreConfig(
    provider="neo4j",
    config=Neo4jConfig(...),
    custom_prompt=custom_graph_prompt,
)
```

**Location**: Lines 239-242 in `/home/user/mem0/mem0/memory/graph_memory.py`
- Replaces "CUSTOM_PROMPT" placeholder in EXTRACT_RELATIONS_PROMPT

---

## 4. Milvus Vector Store Configuration

**File**: `/home/user/mem0/mem0/configs/vector_stores/milvus.py` (Lines 22-42)

### Milvus Connection Parameters

```python
from mem0.configs.vector_stores.milvus import MilvusDBConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.configs.base import MemoryConfig

milvus_config = VectorStoreConfig(
    provider="milvus",
    config={
        "url": "http://localhost:19530",  # Milvus server URL
        "token": None,  # Token for Zilliz cloud (None for local setup)
        "collection_name": "mem0",  # Collection to store embeddings
        "embedding_model_dims": 1536,  # Must match embedder output (1536 for text-embedding-3-small)
        "metric_type": "L2",  # Similarity metric
        "db_name": "",  # Database name (empty string for default)
    }
)

config = MemoryConfig(vector_store=milvus_config)
memory = Memory(config=config)
```

### Metric Types

```python
# Different distance/similarity metrics for Milvus
metric_types = {
    "L2": "Euclidean distance - use when embeddings are not normalized",
    "IP": "Inner Product - good for normalized vectors",
    "COSINE": "Cosine similarity - most common for embeddings",
    "HAMMING": "Hamming distance - for binary vectors",
    "JACCARD": "Jaccard distance - for binary vectors",
}

# For OpenAI embeddings, L2 and COSINE are recommended
milvus_config_l2 = VectorStoreConfig(
    provider="milvus",
    config={
        "url": "http://localhost:19530",
        "collection_name": "mem0",
        "embedding_model_dims": 1536,
        "metric_type": "L2",  # Euclidean distance
    }
)

milvus_config_cosine = VectorStoreConfig(
    provider="milvus",
    config={
        "url": "http://localhost:19530",
        "collection_name": "mem0",
        "embedding_model_dims": 1536,
        "metric_type": "COSINE",  # Cosine similarity
    }
)
```

### Zilliz Cloud Setup

```python
# For Zilliz Cloud (managed Milvus)
zilliz_config = VectorStoreConfig(
    provider="milvus",
    config={
        "url": "https://in03-xxxxx.aws-us-west-2.zillizcloud.com",
        "token": "your-zilliz-api-token",
        "collection_name": "mem0",
        "embedding_model_dims": 1536,
        "metric_type": "L2",
        "db_name": "default",
    }
)
```

### Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `url` | str | Milvus server URL | http://localhost:19530 |
| `token` | str | Zilliz auth token | None |
| `collection_name` | str | Collection name for embeddings | "mem0" |
| `embedding_model_dims` | int | Embedding dimension (must match model) | 1536 |
| `metric_type` | str | Distance metric (L2, IP, COSINE, HAMMING, JACCARD) | "L2" |
| `db_name` | str | Database name | "" |

---

## 5. OpenAI LLM Configuration

**File**: `/home/user/mem0/mem0/configs/llms/openai.py` (Lines 6-79)

### OpenAI LLM Parameters

```python
from mem0.llms.configs import LlmConfig
from mem0.configs.base import MemoryConfig

openai_llm_config = LlmConfig(
    provider="openai",
    config={
        # Model selection
        "model": "gpt-4o-mini",  # Options: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, etc.
        
        # Core parameters
        "temperature": 0.1,  # 0.0 = deterministic, 1.0 = creative (default: 0.1)
        "max_tokens": 2000,  # Maximum response length (default: 2000)
        "top_p": 0.1,  # Nucleus sampling (default: 0.1)
        "top_k": 1,  # Top-k sampling (default: 1)
        
        # API configuration
        "api_key": "sk-xxx",  # Or use OPENAI_API_KEY env var
        "openai_base_url": None,  # Custom API endpoint
        
        # Vision capabilities
        "enable_vision": False,  # Enable image understanding
        "vision_details": "auto",  # Options: "auto", "low", "high"
        
        # OpenRouter compatibility (if using OpenRouter)
        "openrouter_base_url": None,
        "models": None,  # List of models for OpenRouter fallback
        "route": "fallback",  # OpenRouter routing strategy
        "site_url": None,  # Your site URL for OpenRouter
        "app_name": None,  # Your app name for OpenRouter
        
        # Data storage
        "store": False,  # Whether OpenAI stores conversation for improvement
        
        # Advanced: Response monitoring callback
        "response_callback": None,  # Optional callback function for responses
        
        # Proxy configuration
        "http_client_proxies": None,  # HTTP proxy settings
    }
)

config = MemoryConfig(llm=openai_llm_config)
```

### Popular OpenAI Models

```python
# For memory operations (recommended for accuracy)
models = {
    "gpt-4o": "Latest, fastest, good for fact extraction",
    "gpt-4o-mini": "Cheaper, fast, good for memory operations",
    "gpt-4-turbo": "Older model, still powerful",
    "gpt-3.5-turbo": "Cheapest, decent quality",
}

# Recommended for memory operations
memory_optimized_config = LlmConfig(
    provider="openai",
    config={
        "model": "gpt-4o-mini",  # Fast and cost-effective
        "temperature": 0.1,  # Low randomness for consistent fact extraction
        "max_tokens": 2000,
    }
)
```

### Vision Capabilities

```python
# Enable image understanding for memory with images
vision_config = LlmConfig(
    provider="openai",
    config={
        "model": "gpt-4o",  # Must support vision
        "enable_vision": True,
        "vision_details": "high",  # Options: "low", "high", "auto"
    }
)

# When using memory with images
memory.add(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ],
    user_id="user_123"
)
```

### Response Callback

```python
def my_llm_callback(response, messages, config):
    """Monitor LLM responses"""
    print(f"LLM Response: {response}")
    print(f"Input tokens: {messages}")

callback_config = LlmConfig(
    provider="openai",
    config={
        "model": "gpt-4o-mini",
        "response_callback": my_llm_callback,
    }
)
```

### Environment Variables

```bash
# Use these env vars instead of passing api_key
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional custom endpoint
export OPENAI_API_BASE="https://api.openai.com/v1"  # Deprecated but still supported
```

---

## 6. OpenAI Embeddings Configuration

**File**: `/home/user/mem0/mem0/embeddings/openai.py` (Lines 1-50)

### OpenAI Embeddings Parameters

```python
from mem0.embeddings.configs import BaseEmbedderConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.configs.base import MemoryConfig

openai_embedder_config = EmbedderConfig(
    provider="openai",
    config={
        "model": "text-embedding-3-small",  # Most models and cost-effective
        "api_key": "sk-xxx",  # Or use OPENAI_API_KEY env var
        "embedding_dims": 1536,  # Must match configured dimensions
        "openai_base_url": None,  # Custom API endpoint
        
        # Memory-action specific embeddings (not directly used by OpenAI)
        "memory_add_embedding_type": None,
        "memory_update_embedding_type": None,
        "memory_search_embedding_type": None,
    }
)

config = MemoryConfig(embedder=openai_embedder_config)
memory = Memory(config=config)
```

### OpenAI Embedding Models

```python
embedding_models = {
    "text-embedding-3-small": {
        "dims": 1536,  # Default dimensions
        "cost": "Cheapest",
        "quality": "Good for most use cases"
    },
    "text-embedding-3-large": {
        "dims": 3072,
        "cost": "More expensive",
        "quality": "Better semantic understanding"
    },
}

# For Milvus, ensure embedding_dims matches vector store config:
embedder_config = EmbedderConfig(
    provider="openai",
    config={
        "model": "text-embedding-3-small",
        "embedding_dims": 1536,  # Must match Milvus collection dimensions
    }
)

milvus_config = VectorStoreConfig(
    provider="milvus",
    config={
        "embedding_model_dims": 1536,  # MUST match embedder dimensions
    }
)
```

### Memory Action-Specific Embeddings

The OpenAI embedder supports different embeddings for different actions:

```python
from mem0.embeddings.openai import OpenAIEmbedding

embedder = OpenAIEmbedding(config)

# Embeddings used in different memory operations
add_embedding = embedder.embed("My name is John", memory_action="add")
search_embedding = embedder.embed("Find memories about John", memory_action="search")
update_embedding = embedder.embed("Updated information", memory_action="update")

# Currently OpenAI uses the same model for all actions
# This parameter exists for provider extensibility
```

---

## 7. Memory Addition Parameters & Thresholds

**File**: `/home/user/mem0/mem0/memory/main.py` (Lines 281-384)

### Memory.add() Method Parameters

```python
from mem0.memory import Memory
from mem0.configs.base import MemoryConfig

memory = Memory(config=config)

# Add memories with various options
result = memory.add(
    # Message content (string, dict, or list of dicts)
    messages="I like playing tennis on weekends",  # Simple string
    # Or dict format
    # messages={"role": "user", "content": "I like tennis"}
    # Or conversation
    # messages=[
    #     {"role": "user", "content": "I like tennis"},
    #     {"role": "assistant", "content": "Nice! How often?"}
    # ]
    
    # Session identifiers (at least one required)
    user_id="user_123",  # User identifier
    agent_id="agent_456",  # Optional: Agent identifier
    run_id="run_789",  # Optional: Run identifier
    
    # Custom metadata to attach to memories
    metadata={
        "source": "conversation",  # Custom field
        "context": "fitness",  # Custom field
        "custom_field": "any_value",  # Any custom field
    },
    
    # Control memory inference
    infer=True,  # True: Use LLM to extract facts (default)
            # False: Add messages as raw memories directly
    
    # Memory type
    memory_type=None,  # Default: None (creates semantic/episodic)
                      # Or: "procedural_memory" (requires agent_id)
    
    # Custom prompt for this specific add operation
    prompt=None,  # Only used when memory_type="procedural_memory"
)

print(result)
# Output format:
# {
#     "results": [
#         {"id": "xxx", "memory": "...", "event": "ADD"},
#         {"id": "yyy", "memory": "...", "event": "UPDATE"},
#     ],
#     "relations": [...]  # Only if graph_store enabled
# }
```

### Inference Control

```python
# With inference (default) - Uses LLM to extract and deduplicate facts
memory.add(
    messages="I love Italian food and hate spicy food",
    user_id="user_123",
    infer=True,  # LLM extracts separate facts: "loves Italian", "hates spicy"
)

# Without inference - Stores raw messages
memory.add(
    messages="I love Italian food and hate spicy food",
    user_id="user_123",
    infer=False,  # Stores entire message as-is
)
```

### Metadata & Filters

```python
# Store custom metadata with memories
memory.add(
    messages="Had a meeting with John today",
    user_id="user_123",
    metadata={
        "source": "work_meeting",  # Where memory came from
        "date": "2024-01-15",  # When it happened
        "tags": ["work", "meeting"],  # Custom tags
        "sentiment": "positive",  # Custom sentiment
        "importance": "high",  # Custom importance level
        "actor_id": "john_doe",  # Who's involved (auto-extracted from "name" field)
    }
)

# Metadata is attached to every memory entry for filtering/retrieval
```

### Procedural Memory Creation

```python
# Create procedural memories for agents
result = memory.add(
    messages=[
        {"role": "user", "content": "Generate a report"},
        {"role": "assistant", "content": "Creating report with 50 items..."}
    ],
    user_id="user_123",
    agent_id="agent_456",  # Required for procedural memory
    memory_type="procedural_memory",  # Explicit procedural type
    prompt=None,  # Optional: Custom prompt for generation
)

# Procedural memories store sequences of actions for agent continuity
```

### Search & Retrieval Limits

**File**: `/home/user/mem0/mem0/memory/main.py` (Lines 660-690, 765-850)

```python
# Limit number of results
memories = memory.get_all(
    user_id="user_123",
    limit=100,  # Maximum 100 results (default)
)

# Search with limit and threshold
results = memory.search(
    query="favorite food",
    user_id="user_123",
    limit=100,  # Maximum 100 results
    threshold=None,  # Minimum score threshold (None = no threshold)
)

# With threshold filtering
results = memory.search(
    query="favorite food",
    user_id="user_123",
    limit=50,
    threshold=0.7,  # Only return memories with score >= 0.7
)
```

---

## 8. Callbacks, Hooks & Extension Points

**File**: `/home/user/mem0/mem0/memory/main.py`

### Telemetry & Event Capturing

```python
from mem0.memory.telemetry import capture_event

# Capture custom events
capture_event("custom_event_name", self, {
    "user_id": "user_123",
    "custom_field": "value"
})
```

**Location**: Line 233 in main.py - Used throughout for event tracking

### OpenAI Response Callbacks

```python
def response_monitor(response, messages, config):
    """Monitor and log LLM responses"""
    print(f"LLM returned: {len(response)} chars")
    # Log, monitor, validate responses

llm_config = LlmConfig(
    provider="openai",
    config={
        "model": "gpt-4o-mini",
        "response_callback": response_monitor,
    }
)
```

**Location**: Lines 33, 79 in `/home/user/mem0/mem0/configs/llms/openai.py`

### Reranker Configuration (Optional)

```python
from mem0.configs.rerankers.config import RerankerConfig
from mem0.configs.base import MemoryConfig

# Add reranking to improve search quality
config = MemoryConfig(
    reranker=RerankerConfig(
        provider="cohere",
        config={
            "model": "rerank-english-v2.0",
            "api_key": "your-cohere-api-key",
        }
    )
)

# Reranker is automatically used in search operations
results = memory.search(
    query="my favorite food",
    user_id="user_123",
    limit=50  # Will be reranked and limited
)
```

---

## 9. Session Scoping & Filtering

**File**: `/home/user/mem0/mem0/memory/main.py` (Lines 87-165)

### Session Identifiers

```python
# At least one session identifier is required
memory.add(
    messages="I like tennis",
    # Must provide at least one of: user_id, agent_id, or run_id
    user_id="user_123",  # Single user, multiple sessions
    # OR
    agent_id="agent_456",  # Agent memory
    # OR
    run_id="run_789",  # Specific run/conversation
    # OR combination:
    user_id="user_123",
    agent_id="agent_456",
    run_id="run_789",
)

# Query with same filters
results = memory.search(
    query="tennis",
    user_id="user_123",  # Search only this user's memories
    # Optionally narrow further:
    agent_id="agent_456",  # And this agent
    actor_id="john_doe",  # And memories about this actor
)
```

### Actor-Based Filtering

```python
# Add memory with actor information
memory.add(
    messages=[{"role": "user", "name": "john_doe", "content": "I met with Alice"}],
    user_id="user_123",
    metadata={"actor_id": "alice"},  # Explicit actor_id
)

# Query by actor
results = memory.search(
    query="Alice",
    user_id="user_123",
    actor_id="alice",  # Filter by actor_id
)
```

### Filter Building

```python
# Internal filter building (shown for understanding)
def build_filters(user_id=None, agent_id=None, run_id=None, actor_id=None):
    filters = {}
    if user_id:
        filters["user_id"] = user_id
    if agent_id:
        filters["agent_id"] = agent_id
    if run_id:
        filters["run_id"] = run_id
    if actor_id:
        filters["actor_id"] = actor_id
    return filters

# All these result in same filters:
memory.search(query="...", user_id="u1", agent_id="a1", run_id="r1")
memory.search(query="...", user_id="u1", agent_id="a1")  # Narrower
memory.search(query="...", user_id="u1")  # Broadest
```

---

## 10. Memory Types & Operations

**File**: `/home/user/mem0/mem0/memory/main.py` (Lines 310-362)

### Standard Memory vs Procedural Memory

```python
# Standard memories (ADD, UPDATE, DELETE operations)
memory.add(
    messages="I have a dog named Max",
    user_id="user_123",
    # memory_type is None (default)
)

# Procedural memories (stores sequences/procedures)
memory.add(
    messages=[
        {"role": "user", "content": "Step 1: Open browser"},
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Step 2: Navigate to site"},
        {"role": "assistant", "content": "Navigating..."},
    ],
    user_id="user_123",
    agent_id="agent_456",  # Required
    memory_type="procedural_memory",  # Explicit type
)
```

### Memory Update Operations

```python
# Update specific memory
memory.update(
    memory_id="memory_uuid",
    data="Updated information"
)

# Delete specific memory
memory.delete(memory_id="memory_uuid")

# Delete all memories for a user
memory.delete_all(user_id="user_123")

# Delete for specific session
memory.delete_all(
    user_id="user_123",
    agent_id="agent_456"
)

# Get memory history
history = memory.history(memory_id="memory_uuid")
```

---

## Complete Example Configuration

Here's a complete example with all customizations for Neo4j + Milvus + OpenAI:

```python
from mem0.configs.base import MemoryConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.graphs.configs import GraphStoreConfig, Neo4jConfig
from mem0.configs.rerankers.config import RerankerConfig
from mem0.memory import Memory

# Complete configuration
config = MemoryConfig(
    # Vector store: Milvus
    vector_store=VectorStoreConfig(
        provider="milvus",
        config={
            "url": "http://localhost:19530",
            "token": None,
            "collection_name": "mem0",
            "embedding_model_dims": 1536,
            "metric_type": "L2",
            "db_name": "",
        }
    ),
    
    # LLM: OpenAI
    llm=LlmConfig(
        provider="openai",
        config={
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "api_key": "sk-xxx",
            "max_tokens": 2000,
            "top_p": 0.1,
            "enable_vision": False,
        }
    ),
    
    # Embeddings: OpenAI
    embedder=EmbedderConfig(
        provider="openai",
        config={
            "model": "text-embedding-3-small",
            "api_key": "sk-xxx",
            "embedding_dims": 1536,
        }
    ),
    
    # Graph store: Neo4j
    graph_store=GraphStoreConfig(
        provider="neo4j",
        config=Neo4jConfig(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="neo4j",
            base_label=True,
        ),
        custom_prompt=None,
        threshold=0.7,
    ),
    
    # Optional reranker
    reranker=RerankerConfig(
        provider="cohere",
        config={
            "model": "rerank-english-v2.0",
            "api_key": "xxx",
        }
    ),
    
    # Custom prompts
    custom_fact_extraction_prompt=None,
    custom_update_memory_prompt=None,
    
    # API version
    version="v1.1",
)

# Initialize memory
memory = Memory(config=config)

# Use memory
result = memory.add(
    messages=[
        {"role": "user", "content": "I love tennis and hiking"},
        {"role": "assistant", "content": "Great hobbies!"}
    ],
    user_id="user_123",
    metadata={
        "source": "conversation",
        "tags": ["hobbies"],
    }
)

# Search memories
results = memory.search(
    query="What are my hobbies?",
    user_id="user_123",
    limit=10,
    threshold=0.5,
)

print(results)
```

---

## Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Neo4j
export NEO4J_URL="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

# Milvus
export MILVUS_URL="http://localhost:19530"
export MILVUS_TOKEN=""  # For Zilliz
```

---

## Summary Table of Customization Points

| Component | File | Customizable Parameters | Key Feature |
|-----------|------|------------------------|-------------|
| **MemoryConfig** | base.py | vector_store, llm, embedder, graph_store, reranker, prompts | Core configuration |
| **OpenAI LLM** | configs/llms/openai.py | model, temperature, max_tokens, vision, callback | LLM behavior |
| **OpenAI Embeddings** | embeddings/openai.py | model, embedding_dims | Embedding quality |
| **Neo4j Graph** | graphs/configs.py | url, username, password, threshold, custom_prompt | Knowledge graph |
| **Milvus Vector** | configs/vector_stores/milvus.py | url, collection_name, metric_type, embedding_dims | Vector search |
| **Memory Operations** | memory/main.py | infer, memory_type, metadata, filters | Memory behavior |
| **Prompts** | configs/prompts.py | custom_fact_extraction_prompt, custom_update_memory_prompt | Fact extraction |

