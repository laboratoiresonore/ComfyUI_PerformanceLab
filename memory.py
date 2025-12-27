"""
Performance Lab Memory System

Persistent preference learning and storage for workflow optimization.
Tracks user choices over time to improve suggestions.

Inspired by WhimWeaver's preference_harvester.py and ai_auto_tuner.py.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelPreference:
    """Learned preferences for a specific model type."""
    model_type: str
    preferred_cfg: float = 7.0
    preferred_steps: int = 25
    preferred_resolution: int = 1024
    preferred_sampler: str = "euler"

    # Quality vs speed preference (0.0 = speed, 1.0 = quality)
    quality_preference: float = 0.5

    # Statistics
    usage_count: int = 0
    successful_generations: int = 0
    last_used: float = 0.0

    # CFG choices history (for computing average)
    cfg_history: List[float] = field(default_factory=list)

    def update_cfg_preference(self, cfg: float):
        """Update CFG preference with new choice."""
        self.cfg_history.append(cfg)
        # Keep last 20 choices
        self.cfg_history = self.cfg_history[-20:]
        # Update preferred CFG as weighted average (recent choices weighted more)
        if self.cfg_history:
            weights = [i + 1 for i in range(len(self.cfg_history))]
            total_weight = sum(weights)
            self.preferred_cfg = sum(c * w for c, w in zip(self.cfg_history, weights)) / total_weight


@dataclass
class WorkflowMemory:
    """Complete memory state for Performance Lab."""

    # User preferences per model type
    model_preferences: Dict[str, ModelPreference] = field(default_factory=dict)

    # Global preferences
    preferred_quality_level: float = 0.5  # 0.0 = speed, 1.0 = quality

    # Feedback history
    total_feedback_count: int = 0
    a_choices: int = 0
    b_choices: int = 0
    neither_choices: int = 0

    # Issues and solutions that worked
    known_issues: List[Dict[str, str]] = field(default_factory=list)
    solutions_that_worked: List[Dict[str, Any]] = field(default_factory=list)

    # Workflow fingerprints we've seen
    known_workflows: Dict[str, str] = field(default_factory=dict)

    # Session history
    session_count: int = 0
    first_session: float = 0.0
    last_session: float = 0.0

    def get_model_preference(self, model_type: str) -> ModelPreference:
        """Get or create preferences for a model type."""
        model_key = model_type.lower().replace(" ", "_")

        if model_key not in self.model_preferences:
            # Initialize with model-specific defaults
            defaults = {
                "flux": {"cfg": 3.5, "steps": 28, "resolution": 1024},
                "sdxl": {"cfg": 7.0, "steps": 25, "resolution": 1024},
                "sd15": {"cfg": 7.5, "steps": 20, "resolution": 512},
                "sd3": {"cfg": 4.5, "steps": 28, "resolution": 1024},
            }

            default = defaults.get(model_key, {"cfg": 7.0, "steps": 25, "resolution": 1024})
            self.model_preferences[model_key] = ModelPreference(
                model_type=model_key,
                preferred_cfg=default["cfg"],
                preferred_steps=default["steps"],
                preferred_resolution=default["resolution"]
            )

        return self.model_preferences[model_key]

    def record_feedback(self, choice: str, settings_a: Dict, settings_b: Dict, reason: str = ""):
        """Record a user's A/B choice."""
        self.total_feedback_count += 1

        if choice == "A":
            self.a_choices += 1
            chosen = settings_a
        elif choice == "B":
            self.b_choices += 1
            chosen = settings_b
        else:
            self.neither_choices += 1
            return

        # Update model-specific preferences if we have model info
        model_type = chosen.get("model_type", "unknown")
        pref = self.get_model_preference(model_type)

        if "cfg" in chosen:
            pref.update_cfg_preference(chosen["cfg"])
        if "steps" in chosen:
            pref.preferred_steps = chosen["steps"]
        if "resolution" in chosen:
            pref.preferred_resolution = chosen["resolution"]

        pref.usage_count += 1
        pref.last_used = time.time()

        # Record the solution that worked
        if reason:
            self.solutions_that_worked.append({
                "timestamp": time.time(),
                "settings": chosen,
                "reason": reason
            })
            # Keep last 50 solutions
            self.solutions_that_worked = self.solutions_that_worked[-50:]

    def record_issue(self, issue: str, solution: Optional[str] = None):
        """Record an issue and its solution."""
        self.known_issues.append({
            "timestamp": time.time(),
            "issue": issue,
            "solution": solution
        })
        # Keep last 100 issues
        self.known_issues = self.known_issues[-100:]

    def get_suggestions_for_model(self, model_type: str) -> Dict[str, Any]:
        """Get learned optimal settings for a model."""
        pref = self.get_model_preference(model_type)

        return {
            "cfg": pref.preferred_cfg,
            "steps": pref.preferred_steps,
            "resolution": pref.preferred_resolution,
            "sampler": pref.preferred_sampler,
            "confidence": min(1.0, pref.usage_count / 10),  # Higher with more usage
        }

    def get_context_for_llm(self) -> str:
        """Generate context string for LLM prompts."""
        lines = ["## User Preference Profile"]

        lines.append(f"\nQuality vs Speed preference: {self.preferred_quality_level:.1%} quality focused")
        lines.append(f"Total feedback given: {self.total_feedback_count}")

        if self.model_preferences:
            lines.append("\n### Model-Specific Learned Settings")
            for model_key, pref in self.model_preferences.items():
                if pref.usage_count > 0:
                    lines.append(f"\n**{model_key.upper()}**:")
                    lines.append(f"  - Preferred CFG: {pref.preferred_cfg:.1f}")
                    lines.append(f"  - Preferred Steps: {pref.preferred_steps}")
                    lines.append(f"  - Resolution: {pref.preferred_resolution}")
                    lines.append(f"  - Usage count: {pref.usage_count}")

        if self.solutions_that_worked:
            lines.append("\n### Recent Solutions That Worked")
            for sol in self.solutions_that_worked[-5:]:
                lines.append(f"  - {sol.get('reason', 'Unknown reason')}")

        return "\n".join(lines)


class MemoryManager:
    """Manages loading/saving of Performance Lab memory."""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize memory manager.

        Args:
            storage_path: Path to memory storage directory.
                         Defaults to ~/.performance_lab/
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".performance_lab"

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.storage_path / "memory.json"
        self.memory: Optional[WorkflowMemory] = None

    def load(self) -> WorkflowMemory:
        """Load memory from disk or create new."""
        if self.memory is not None:
            return self.memory

        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)

                # Reconstruct memory from JSON
                memory = WorkflowMemory()
                memory.preferred_quality_level = data.get("preferred_quality_level", 0.5)
                memory.total_feedback_count = data.get("total_feedback_count", 0)
                memory.a_choices = data.get("a_choices", 0)
                memory.b_choices = data.get("b_choices", 0)
                memory.neither_choices = data.get("neither_choices", 0)
                memory.known_issues = data.get("known_issues", [])
                memory.solutions_that_worked = data.get("solutions_that_worked", [])
                memory.known_workflows = data.get("known_workflows", {})
                memory.session_count = data.get("session_count", 0)
                memory.first_session = data.get("first_session", 0.0)
                memory.last_session = data.get("last_session", 0.0)

                # Reconstruct model preferences
                for model_key, pref_data in data.get("model_preferences", {}).items():
                    memory.model_preferences[model_key] = ModelPreference(
                        model_type=pref_data.get("model_type", model_key),
                        preferred_cfg=pref_data.get("preferred_cfg", 7.0),
                        preferred_steps=pref_data.get("preferred_steps", 25),
                        preferred_resolution=pref_data.get("preferred_resolution", 1024),
                        preferred_sampler=pref_data.get("preferred_sampler", "euler"),
                        quality_preference=pref_data.get("quality_preference", 0.5),
                        usage_count=pref_data.get("usage_count", 0),
                        successful_generations=pref_data.get("successful_generations", 0),
                        last_used=pref_data.get("last_used", 0.0),
                        cfg_history=pref_data.get("cfg_history", [])
                    )

                self.memory = memory

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Performance Lab] Error loading memory: {e}. Starting fresh.")
                self.memory = WorkflowMemory()
        else:
            self.memory = WorkflowMemory()

        # Update session tracking
        if self.memory.first_session == 0.0:
            self.memory.first_session = time.time()
        self.memory.session_count += 1
        self.memory.last_session = time.time()

        return self.memory

    def save(self):
        """Save memory to disk."""
        if self.memory is None:
            return

        # Convert to JSON-serializable dict
        data = {
            "preferred_quality_level": self.memory.preferred_quality_level,
            "total_feedback_count": self.memory.total_feedback_count,
            "a_choices": self.memory.a_choices,
            "b_choices": self.memory.b_choices,
            "neither_choices": self.memory.neither_choices,
            "known_issues": self.memory.known_issues,
            "solutions_that_worked": self.memory.solutions_that_worked,
            "known_workflows": self.memory.known_workflows,
            "session_count": self.memory.session_count,
            "first_session": self.memory.first_session,
            "last_session": self.memory.last_session,
            "model_preferences": {}
        }

        for model_key, pref in self.memory.model_preferences.items():
            data["model_preferences"][model_key] = {
                "model_type": pref.model_type,
                "preferred_cfg": pref.preferred_cfg,
                "preferred_steps": pref.preferred_steps,
                "preferred_resolution": pref.preferred_resolution,
                "preferred_sampler": pref.preferred_sampler,
                "quality_preference": pref.quality_preference,
                "usage_count": pref.usage_count,
                "successful_generations": pref.successful_generations,
                "last_used": pref.last_used,
                "cfg_history": pref.cfg_history
            }

        try:
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Performance Lab] Error saving memory: {e}")

    def reset(self):
        """Reset memory to defaults."""
        self.memory = WorkflowMemory()
        self.save()

    def get_summary(self) -> str:
        """Get a human-readable summary of the memory state."""
        memory = self.load()

        lines = [
            "═══ Performance Lab Memory ═══",
            f"Sessions: {memory.session_count}",
            f"Feedback given: {memory.total_feedback_count}",
        ]

        if memory.total_feedback_count > 0:
            lines.append(f"  A choices: {memory.a_choices} ({memory.a_choices/memory.total_feedback_count*100:.0f}%)")
            lines.append(f"  B choices: {memory.b_choices} ({memory.b_choices/memory.total_feedback_count*100:.0f}%)")
            lines.append(f"  Neither: {memory.neither_choices}")

        if memory.model_preferences:
            lines.append("\nLearned Model Preferences:")
            for model_key, pref in memory.model_preferences.items():
                if pref.usage_count > 0:
                    lines.append(f"  {model_key.upper()}: CFG {pref.preferred_cfg:.1f}, {pref.preferred_steps} steps")

        return "\n".join(lines)


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def get_memory() -> WorkflowMemory:
    """Convenience function to get the loaded memory."""
    return get_memory_manager().load()


def save_memory():
    """Convenience function to save memory."""
    get_memory_manager().save()
