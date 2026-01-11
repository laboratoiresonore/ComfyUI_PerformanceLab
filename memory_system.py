"""
Performance Lab v2.0 - Memory System
====================================
Persistent memory for workflow optimization learnings.

The memory system stores:
- Workflow optimization patterns
- Successful parameter configurations
- Failed optimization attempts (to avoid repeating)
- Performance metrics over time
- User feedback and preferences

Storage: JSON files in memory/ directory
Structure: Hierarchical with categories and embeddings for retrieval
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# MEMORY TYPES
# ============================================================================

class MemoryType(Enum):
    """Types of memories stored."""
    OPTIMIZATION = "optimization"      # Successful optimization
    FAILURE = "failure"                # Failed optimization attempt
    PATTERN = "pattern"                # Recognized optimization pattern
    USER_PREFERENCE = "user_preference"  # User feedback/preference
    PERFORMANCE = "performance"        # Performance metric
    EXPERIMENT = "experiment"          # A/B test result


class MemoryPriority(Enum):
    """Priority levels for memory importance."""
    CRITICAL = "critical"  # Never forget
    HIGH = "high"          # Keep long-term
    MEDIUM = "medium"      # Standard retention
    LOW = "low"            # Can be pruned


# ============================================================================
# MEMORY DATA CLASS
# ============================================================================

@dataclass
class Memory:
    """Single memory entry."""
    memory_id: str
    memory_type: MemoryType
    priority: MemoryPriority
    content: Dict[str, Any]
    tags: List[str]
    created_at: str
    accessed_count: int = 0
    last_accessed: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
            "accessed_count": self.accessed_count,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            memory_type=MemoryType(data["memory_type"]),
            priority=MemoryPriority(data["priority"]),
            content=data["content"],
            tags=data["tags"],
            created_at=data["created_at"],
            accessed_count=data.get("accessed_count", 0),
            last_accessed=data.get("last_accessed"),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# MEMORY SYSTEM
# ============================================================================

class MemorySystem:
    """
    Persistent memory system for Performance Lab.
    
    Features:
    - Hierarchical storage by type
    - Tag-based retrieval
    - Access tracking
    - Priority-based retention
    - JSON persistence
    """
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.memories: Dict[str, Memory] = {}
        self.tag_index: Dict[str, List[str]] = {}  # tag -> memory_ids
        
        # Load existing memories
        self._load_all_memories()
        
        logger.info(f"Memory system initialized with {len(self.memories)} memories")
    
    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        tags: List[str],
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a new memory.
        
        Args:
            memory_type: Type of memory
            content: Memory content (dict)
            tags: Tags for retrieval
            priority: Memory priority
            memory_id: Optional custom ID
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        # Generate ID if not provided
        if not memory_id:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            memory_id = f"{memory_type.value}_{timestamp}_{len(self.memories)}"
        
        # Create memory
        memory = Memory(
            memory_id=memory_id,
            memory_type=memory_type,
            priority=priority,
            content=content,
            tags=tags,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata
        )
        
        # Store in cache
        self.memories[memory_id] = memory
        
        # Update tag index
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            if memory_id not in self.tag_index[tag]:
                self.tag_index[tag].append(memory_id)
        
        # Persist to disk
        self._save_memory(memory)
        
        logger.info(f"Stored memory: {memory_id} (type={memory_type.value}, tags={tags})")
        
        return memory_id
    
    def retrieve(
        self,
        memory_id: str,
        increment_access: bool = True
    ) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory ID
            increment_access: Whether to increment access count
            
        Returns:
            Memory or None if not found
        """
        memory = self.memories.get(memory_id)
        
        if memory and increment_access:
            memory.accessed_count += 1
            memory.last_accessed = datetime.utcnow().isoformat()
            self._save_memory(memory)
        
        return memory
    
    def search_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """
        Search memories by tags.
        
        Args:
            tags: Tags to search for
            match_all: If True, require all tags; if False, any tag
            limit: Optional result limit
            
        Returns:
            List of matching memories
        """
        if match_all:
            # Find memories that have ALL tags
            memory_id_sets = [set(self.tag_index.get(tag, [])) for tag in tags]
            if memory_id_sets:
                matching_ids = set.intersection(*memory_id_sets)
            else:
                matching_ids = set()
        else:
            # Find memories that have ANY tag
            matching_ids = set()
            for tag in tags:
                matching_ids.update(self.tag_index.get(tag, []))
        
        # Retrieve memories
        memories = [self.memories[mid] for mid in matching_ids if mid in self.memories]
        
        # Sort by priority and access count
        memories.sort(
            key=lambda m: (
                ["critical", "high", "medium", "low"].index(m.priority.value),
                -m.accessed_count
            )
        )
        
        # Apply limit
        if limit:
            memories = memories[:limit]
        
        return memories
    
    def search_by_type(
        self,
        memory_type: MemoryType,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """Search memories by type."""
        memories = [m for m in self.memories.values() if m.memory_type == memory_type]
        
        # Sort by priority and recency
        memories.sort(
            key=lambda m: (
                ["critical", "high", "medium", "low"].index(m.priority.value),
                m.created_at
            ),
            reverse=True
        )
        
        if limit:
            memories = memories[:limit]
        
        return memories
    
    def search_by_content(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """
        Search memories by content (simple text search).
        
        For production, this should use embeddings and vector search.
        """
        query_lower = query.lower()
        matching_memories = []
        
        for memory in self.memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Convert content to searchable string
            content_str = json.dumps(memory.content).lower()
            
            if query_lower in content_str:
                matching_memories.append(memory)
        
        # Sort by priority and relevance (accessed count as proxy)
        matching_memories.sort(
            key=lambda m: (
                ["critical", "high", "medium", "low"].index(m.priority.value),
                -m.accessed_count
            )
        )
        
        if limit:
            matching_memories = matching_memories[:limit]
        
        return matching_memories
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        
        # Remove from tag index
        for tag in memory.tags:
            if tag in self.tag_index:
                self.tag_index[tag] = [mid for mid in self.tag_index[tag] if mid != memory_id]
        
        # Remove from cache
        del self.memories[memory_id]
        
        # Delete file
        file_path = self._get_memory_file_path(memory_id)
        if file_path.exists():
            file_path.unlink()
        
        logger.info(f"Deleted memory: {memory_id}")
        return True
    
    def update(
        self,
        memory_id: str,
        content: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[MemoryPriority] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory."""
        memory = self.memories.get(memory_id)
        if not memory:
            return False
        
        # Update fields
        if content is not None:
            memory.content = content
        
        if tags is not None:
            # Remove old tags from index
            for old_tag in memory.tags:
                if old_tag in self.tag_index:
                    self.tag_index[old_tag] = [mid for mid in self.tag_index[old_tag] if mid != memory_id]
            
            # Add new tags to index
            memory.tags = tags
            for tag in tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                if memory_id not in self.tag_index[tag]:
                    self.tag_index[tag].append(memory_id)
        
        if priority is not None:
            memory.priority = priority
        
        if metadata is not None:
            memory.metadata = metadata
        
        # Save to disk
        self._save_memory(memory)
        
        logger.info(f"Updated memory: {memory_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        type_counts = {}
        for memory in self.memories.values():
            type_name = memory.memory_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        priority_counts = {}
        for memory in self.memories.values():
            priority_name = memory.priority.value
            priority_counts[priority_name] = priority_counts.get(priority_name, 0) + 1
        
        return {
            "total_memories": len(self.memories),
            "total_tags": len(self.tag_index),
            "memories_by_type": type_counts,
            "memories_by_priority": priority_counts,
            "most_accessed": self._get_most_accessed(5)
        }
    
    def _get_most_accessed(self, limit: int) -> List[Dict[str, Any]]:
        """Get most accessed memories."""
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.accessed_count,
            reverse=True
        )[:limit]
        
        return [
            {
                "memory_id": m.memory_id,
                "type": m.memory_type.value,
                "accessed_count": m.accessed_count,
                "tags": m.tags
            }
            for m in sorted_memories
        ]
    
    def _save_memory(self, memory: Memory):
        """Save memory to disk."""
        file_path = self._get_memory_file_path(memory.memory_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _load_all_memories(self):
        """Load all memories from disk."""
        if not self.memory_dir.exists():
            return
        
        # Load memories from all subdirectories
        for memory_file in self.memory_dir.rglob("*.json"):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                memory = Memory.from_dict(data)
                self.memories[memory.memory_id] = memory
                
                # Build tag index
                for tag in memory.tags:
                    if tag not in self.tag_index:
                        self.tag_index[tag] = []
                    if memory.memory_id not in self.tag_index[tag]:
                        self.tag_index[tag].append(memory.memory_id)
                
            except Exception as e:
                logger.warning(f"Failed to load memory from {memory_file}: {e}")
    
    def _get_memory_file_path(self, memory_id: str) -> Path:
        """Get file path for memory."""
        # Organize by type subdirectory
        memory = self.memories.get(memory_id)
        if memory:
            type_dir = self.memory_dir / memory.memory_type.value
        else:
            # Fallback for new memories
            type_dir = self.memory_dir / "unknown"
        
        return type_dir / f"{memory_id}.json"


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Global memory system instance
memory_system = MemorySystem()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def store_optimization(
    workflow_name: str,
    optimization_details: Dict[str, Any],
    performance_gain: float,
    tags: List[str],
    priority: MemoryPriority = MemoryPriority.HIGH
) -> str:
    """Store successful optimization."""
    content = {
        "workflow_name": workflow_name,
        "optimization": optimization_details,
        "performance_gain": performance_gain,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return memory_system.store(
        memory_type=MemoryType.OPTIMIZATION,
        content=content,
        tags=tags + ["optimization", workflow_name],
        priority=priority
    )


def store_failure(
    workflow_name: str,
    attempted_optimization: Dict[str, Any],
    failure_reason: str,
    tags: List[str]
) -> str:
    """Store failed optimization attempt."""
    content = {
        "workflow_name": workflow_name,
        "attempted_optimization": attempted_optimization,
        "failure_reason": failure_reason,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return memory_system.store(
        memory_type=MemoryType.FAILURE,
        content=content,
        tags=tags + ["failure", workflow_name],
        priority=MemoryPriority.MEDIUM
    )


def retrieve_similar_optimizations(
    workflow_name: str,
    tags: List[str],
    limit: int = 5
) -> List[Memory]:
    """Retrieve similar successful optimizations."""
    all_tags = tags + ["optimization", workflow_name]
    return memory_system.search_by_tags(all_tags, match_all=False, limit=limit)
