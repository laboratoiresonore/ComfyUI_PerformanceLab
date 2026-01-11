"""
Performance Lab v2.0 - ComfyUI Custom Nodes
===========================================
LLM-guided workflow optimization nodes for ComfyUI.

Nodes:
- PerfLab_LLMPrompt: Send prompts to LLM
- PerfLab_WorkflowAnalyzer: Analyze current workflow
- PerfLab_ParameterOptimizer: Optimize node parameters
- PerfLab_MemoryRetrieval: Retrieve optimization memories
- PerfLab_QualityTierSelector: Select quality tier for generation
"""

import sys
import os
from pathlib import Path

# Add performance_lab to path
performance_lab_path = Path(__file__).parent
if str(performance_lab_path) not in sys.path:
    sys.path.insert(0, str(performance_lab_path))

# Import Performance Lab modules
try:
    from litellm_config import litellm_manager, QualityTier
    from memory_system import memory_system, MemoryType
    PERFLAB_AVAILABLE = True
except ImportError as e:
    print(f"Performance Lab modules not available: {e}")
    PERFLAB_AVAILABLE = False

import asyncio
import json
from typing import Tuple, Dict, Any


# ============================================================================
# NODE: LLM PROMPT
# ============================================================================

class PerfLab_LLMPrompt:
    """
    Send prompt to LLM and get response.
    
    Inputs:
    - prompt: Text prompt
    - quality_tier: Quality tier (CRITICAL/IMPORTANT/STANDARD/QUICK)
    - max_tokens: Maximum tokens to generate
    - temperature: Generation temperature
    - system_prompt: Optional system prompt
    
    Outputs:
    - text: Generated text
    - metadata: Generation metadata (model, cost, tokens)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Analyze this workflow and suggest optimizations..."
                }),
                "quality_tier": (["CRITICAL", "IMPORTANT", "STANDARD", "QUICK"], {
                    "default": "STANDARD"
                }),
            },
            "optional": {
                "max_tokens": ("INT", {
                    "default": 500,
                    "min": 50,
                    "max": 4096,
                    "step": 50
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "metadata")
    FUNCTION = "generate"
    CATEGORY = "Performance Lab"
    
    def generate(
        self,
        prompt: str,
        quality_tier: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: str = ""
    ) -> Tuple[str, str]:
        """Generate text using LLM."""
        
        if not PERFLAB_AVAILABLE:
            return ("Performance Lab not available", "{}")
        
        # Convert quality tier string to enum
        tier = QualityTier[quality_tier]
        
        # Run async generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                litellm_manager.generate(
                    prompt=prompt,
                    quality_tier=tier,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt if system_prompt else None
                )
            )
        finally:
            loop.close()
        
        # Extract result
        if result.get("success"):
            text = result.get("text", "")
            metadata = {
                "model": result.get("model"),
                "tokens_used": result.get("tokens_used"),
                "cost": result.get("cost"),
                "quality_tier": quality_tier
            }
            return (text, json.dumps(metadata, indent=2))
        else:
            error = result.get("error", "Unknown error")
            return (f"Error: {error}", "{}")


# ============================================================================
# NODE: WORKFLOW ANALYZER
# ============================================================================

class PerfLab_WorkflowAnalyzer:
    """
    Analyze current ComfyUI workflow and identify optimization opportunities.
    
    Inputs:
    - workflow_json: Current workflow as JSON string
    - analysis_focus: What to analyze (performance/quality/efficiency)
    
    Outputs:
    - analysis: Analysis text
    - suggestions: Optimization suggestions
    - bottlenecks: Identified bottlenecks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json": ("STRING", {
                    "multiline": True,
                    "default": "{}"
                }),
                "analysis_focus": (["performance", "quality", "efficiency", "all"], {
                    "default": "all"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("analysis", "suggestions", "bottlenecks")
    FUNCTION = "analyze"
    CATEGORY = "Performance Lab"
    
    def analyze(
        self,
        workflow_json: str,
        analysis_focus: str
    ) -> Tuple[str, str, str]:
        """Analyze workflow using LLM."""
        
        if not PERFLAB_AVAILABLE:
            return ("Not available", "Not available", "Not available")
        
        # Parse workflow
        try:
            workflow = json.loads(workflow_json)
        except json.JSONDecodeError:
            return ("Invalid workflow JSON", "", "")
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(workflow, analysis_focus)
        
        # Call LLM
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                litellm_manager.generate(
                    prompt=prompt,
                    quality_tier=QualityTier.IMPORTANT,
                    max_tokens=1000,
                    temperature=0.5,
                    system_prompt="You are an expert at analyzing ComfyUI workflows for optimization opportunities."
                )
            )
        finally:
            loop.close()
        
        if result.get("success"):
            text = result.get("text", "")
            
            # Parse response sections
            analysis, suggestions, bottlenecks = self._parse_analysis_response(text)
            
            return (analysis, suggestions, bottlenecks)
        else:
            error = result.get("error", "Analysis failed")
            return (f"Error: {error}", "", "")
    
    def _build_analysis_prompt(self, workflow: Dict, focus: str) -> str:
        """Build analysis prompt."""
        node_count = len(workflow.get("nodes", []))
        
        prompt = f"""Analyze this ComfyUI workflow with {node_count} nodes.

Focus: {focus}

Workflow structure:
{json.dumps(workflow, indent=2)[:2000]}  # Truncate for token limits

Provide:
1. ANALYSIS: Overall workflow analysis
2. SUGGESTIONS: Specific optimization suggestions
3. BOTTLENECKS: Identified performance bottlenecks

Format your response with clear section headers."""
        
        return prompt
    
    def _parse_analysis_response(self, text: str) -> Tuple[str, str, str]:
        """Parse LLM response into sections."""
        analysis = ""
        suggestions = ""
        bottlenecks = ""
        
        current_section = None
        for line in text.split("\n"):
            line_lower = line.lower()
            
            if "analysis" in line_lower and ":" in line:
                current_section = "analysis"
            elif "suggestion" in line_lower and ":" in line:
                current_section = "suggestions"
            elif "bottleneck" in line_lower and ":" in line:
                current_section = "bottlenecks"
            elif current_section == "analysis":
                analysis += line + "\n"
            elif current_section == "suggestions":
                suggestions += line + "\n"
            elif current_section == "bottlenecks":
                bottlenecks += line + "\n"
        
        return (
            analysis.strip() or text,  # Fallback to full text
            suggestions.strip(),
            bottlenecks.strip()
        )


# ============================================================================
# NODE: PARAMETER OPTIMIZER
# ============================================================================

class PerfLab_ParameterOptimizer:
    """
    Optimize node parameters using LLM and memory system.
    
    Inputs:
    - node_type: Type of node to optimize
    - current_params: Current parameter values
    - optimization_goal: What to optimize for
    
    Outputs:
    - optimized_params: Suggested parameter values
    - reasoning: Why these parameters were suggested
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "node_type": ("STRING", {
                    "default": "KSampler"
                }),
                "current_params": ("STRING", {
                    "multiline": True,
                    "default": "{}"
                }),
                "optimization_goal": (["speed", "quality", "balance"], {
                    "default": "balance"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("optimized_params", "reasoning")
    FUNCTION = "optimize"
    CATEGORY = "Performance Lab"
    
    def optimize(
        self,
        node_type: str,
        current_params: str,
        optimization_goal: str
    ) -> Tuple[str, str]:
        """Optimize parameters using LLM and memories."""
        
        if not PERFLAB_AVAILABLE:
            return ("{}", "Not available")
        
        # Search memory for similar optimizations
        memories = memory_system.search_by_tags(
            tags=[node_type, optimization_goal, "optimization"],
            match_all=False,
            limit=5
        )
        
        # Build optimization prompt
        prompt = self._build_optimization_prompt(
            node_type,
            current_params,
            optimization_goal,
            memories
        )
        
        # Call LLM
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                litellm_manager.generate(
                    prompt=prompt,
                    quality_tier=QualityTier.IMPORTANT,
                    max_tokens=800,
                    temperature=0.3,
                    system_prompt="You are an expert at optimizing ComfyUI node parameters."
                )
            )
        finally:
            loop.close()
        
        if result.get("success"):
            text = result.get("text", "")
            
            # Parse response
            params, reasoning = self._parse_optimization_response(text)
            
            return (params, reasoning)
        else:
            return ("{}", f"Error: {result.get('error')}")
    
    def _build_optimization_prompt(
        self,
        node_type: str,
        current_params: str,
        goal: str,
        memories: list
    ) -> str:
        """Build optimization prompt with memory context."""
        
        memory_context = ""
        if memories:
            memory_context = "\n\nPrevious successful optimizations:\n"
            for mem in memories[:3]:
                content = mem.content
                memory_context += f"- {content.get('optimization', {})}\n"
        
        prompt = f"""Optimize parameters for {node_type} node.

Current parameters:
{current_params}

Optimization goal: {goal}
{memory_context}

Provide:
1. OPTIMIZED_PARAMS: JSON object with optimized parameter values
2. REASONING: Why these values are better

Format as:
OPTIMIZED_PARAMS:
{{...}}

REASONING:
..."""
        
        return prompt
    
    def _parse_optimization_response(self, text: str) -> Tuple[str, str]:
        """Parse LLM response."""
        params = "{}"
        reasoning = ""
        
        # Try to extract JSON
        if "OPTIMIZED_PARAMS" in text:
            parts = text.split("OPTIMIZED_PARAMS:")
            if len(parts) > 1:
                json_part = parts[1].split("REASONING:")[0].strip()
                try:
                    # Validate JSON
                    json.loads(json_part)
                    params = json_part
                except:
                    pass
        
        # Extract reasoning
        if "REASONING:" in text:
            reasoning = text.split("REASONING:")[1].strip()
        else:
            reasoning = text
        
        return (params, reasoning)


# ============================================================================
# NODE: MEMORY RETRIEVAL
# ============================================================================

class PerfLab_MemoryRetrieval:
    """
    Retrieve relevant optimization memories.
    
    Inputs:
    - query_tags: Tags to search for
    - memory_type: Type of memory to retrieve
    - limit: Maximum number of memories
    
    Outputs:
    - memories: Retrieved memories as JSON
    - summary: Summary of memories
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query_tags": ("STRING", {
                    "default": "optimization"
                }),
                "memory_type": (["optimization", "failure", "pattern", "all"], {
                    "default": "all"
                }),
                "limit": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("memories", "summary")
    FUNCTION = "retrieve"
    CATEGORY = "Performance Lab"
    
    def retrieve(
        self,
        query_tags: str,
        memory_type: str,
        limit: int
    ) -> Tuple[str, str]:
        """Retrieve memories."""
        
        if not PERFLAB_AVAILABLE:
            return ("[]", "Not available")
        
        # Parse tags
        tags = [t.strip() for t in query_tags.split(",")]
        
        # Search memories
        if memory_type == "all":
            memories = memory_system.search_by_tags(tags, match_all=False, limit=limit)
        else:
            mem_type = MemoryType(memory_type)
            type_memories = memory_system.search_by_type(mem_type, limit=limit * 2)
            # Filter by tags
            memories = [m for m in type_memories if any(t in m.tags for t in tags)][:limit]
        
        # Format output
        memories_json = json.dumps([m.to_dict() for m in memories], indent=2)
        
        # Generate summary
        summary = f"Found {len(memories)} memories for tags: {query_tags}\n\n"
        for i, mem in enumerate(memories, 1):
            summary += f"{i}. {mem.memory_type.value} (priority: {mem.priority.value})\n"
            summary += f"   Tags: {', '.join(mem.tags[:5])}\n"
            summary += f"   Accessed: {mem.accessed_count} times\n\n"
        
        return (memories_json, summary)


# ============================================================================
# NODE: QUALITY TIER SELECTOR
# ============================================================================

class PerfLab_QualityTierSelector:
    """
    Select appropriate quality tier based on context.
    
    Inputs:
    - task_complexity: Complexity of task
    - time_constraint: Time available
    - cost_sensitivity: How cost-sensitive
    
    Outputs:
    - quality_tier: Recommended quality tier
    - explanation: Why this tier was selected
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_complexity": (["low", "medium", "high", "critical"], {
                    "default": "medium"
                }),
                "time_constraint": (["urgent", "normal", "relaxed"], {
                    "default": "normal"
                }),
                "cost_sensitivity": (["low", "medium", "high"], {
                    "default": "medium"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("quality_tier", "explanation")
    FUNCTION = "select_tier"
    CATEGORY = "Performance Lab"
    
    def select_tier(
        self,
        task_complexity: str,
        time_constraint: str,
        cost_sensitivity: str
    ) -> Tuple[str, str]:
        """Select quality tier based on context."""
        
        # Decision logic
        if task_complexity == "critical":
            tier = "CRITICAL"
            explanation = "Critical task complexity requires best available model"
        elif task_complexity == "high" and cost_sensitivity == "low":
            tier = "CRITICAL"
            explanation = "High complexity with low cost sensitivity - using best model"
        elif task_complexity == "high":
            tier = "IMPORTANT"
            explanation = "High complexity - using high-quality model"
        elif time_constraint == "urgent" and task_complexity == "low":
            tier = "QUICK"
            explanation = "Urgent with low complexity - using fast local model"
        elif time_constraint == "urgent":
            tier = "STANDARD"
            explanation = "Urgent timing - using balanced model"
        elif cost_sensitivity == "high":
            tier = "STANDARD"
            explanation = "High cost sensitivity - using standard model"
        else:
            tier = "STANDARD"
            explanation = "Balanced requirements - using standard model"
        
        return (tier, explanation)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "PerfLab_LLMPrompt": PerfLab_LLMPrompt,
    "PerfLab_WorkflowAnalyzer": PerfLab_WorkflowAnalyzer,
    "PerfLab_ParameterOptimizer": PerfLab_ParameterOptimizer,
    "PerfLab_MemoryRetrieval": PerfLab_MemoryRetrieval,
    "PerfLab_QualityTierSelector": PerfLab_QualityTierSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerfLab_LLMPrompt": "🧠 Performance Lab: LLM Prompt",
    "PerfLab_WorkflowAnalyzer": "🔍 Performance Lab: Workflow Analyzer",
    "PerfLab_ParameterOptimizer": "⚙️ Performance Lab: Parameter Optimizer",
    "PerfLab_MemoryRetrieval": "💾 Performance Lab: Memory Retrieval",
    "PerfLab_QualityTierSelector": "🎯 Performance Lab: Quality Tier Selector"
}
