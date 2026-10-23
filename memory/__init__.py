"""
Memory Layer

This module handles system memory:
- Vector store (ChromaDB) for episodic memory
- Working memory for current state
- Pattern storage and retrieval
"""

from memory.vector_store import VectorStore, create_vector_store
from memory.episodic import EpisodicMemory, Episode, create_episodic_memory
from memory.working import (
    WorkingMemory,
    MarketContext,
    OpenPosition,
    RecentAlert,
    get_working_memory,
    reset_working_memory,
)

__all__ = [
    # Vector Store
    "VectorStore",
    "create_vector_store",
    # Episodic Memory
    "EpisodicMemory",
    "Episode",
    "create_episodic_memory",
    # Working Memory
    "WorkingMemory",
    "MarketContext",
    "OpenPosition",
    "RecentAlert",
    "get_working_memory",
    "reset_working_memory",
]
