#!/usr/bin/env python3
"""
LLM Enhancer Module for Performance Lab
Provides comprehensive context and validation for LLM-assisted workflow optimization.

Features:
- Node Catalog Export: Query installed nodes from ComfyUI
- System Specs: GPU, VRAM, CPU information
- Prompt Templates: Goal-oriented optimization templates
- Mod Validation: Validate LLM-generated workflow modifications
- Conversation Memory: Persist context across sessions
- Community Knowledge Base: Common issues and solutions
- Workflow Graph Export: ASCII visualization of workflow structure
- Error History: Track and export errors for debugging
"""

import json
import os
import re
import hashlib
import platform
import subprocess
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

try:
    import urllib.request
    import urllib.error
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

COMFYUI_API_URL = "http://127.0.0.1:8188"
MEMORY_DB_PATH = Path.home() / ".performance_lab" / "memory.db"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.json"


# ═══════════════════════════════════════════════════════════════════════════════
# NODE CATALOG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeInfo:
    """Information about a ComfyUI node."""
    name: str
    category: str
    input_types: Dict[str, Any]
    output_types: List[str]
    output_names: List[str]
    description: str = ""
    is_output_node: bool = False

    def to_compact_string(self) -> str:
        """Compact representation for LLM context."""
        inputs = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in self.input_types.get('required', {}).items())
        outputs = ", ".join(self.output_types) if self.output_types else "none"
        return f"{self.name} [{self.category}] - in({inputs}) -> out({outputs})"


class NodeCatalog:
    """Fetches and manages the ComfyUI node catalog."""

    def __init__(self, api_url: str = COMFYUI_API_URL):
        self.api_url = api_url
        self._cache: Dict[str, NodeInfo] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._last_fetch: Optional[datetime] = None

    def fetch_catalog(self, force_refresh: bool = False) -> bool:
        """Fetch node catalog from ComfyUI API."""
        if not URLLIB_AVAILABLE:
            return False

        if self._cache and not force_refresh:
            return True

        try:
            url = f"{self.api_url}/object_info"
            req = urllib.request.Request(url, headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            self._cache.clear()
            self._categories.clear()

            for node_name, node_data in data.items():
                category = node_data.get('category', 'uncategorized')

                node_info = NodeInfo(
                    name=node_name,
                    category=category,
                    input_types=node_data.get('input', {}),
                    output_types=node_data.get('output', []),
                    output_names=node_data.get('output_name', []),
                    description=node_data.get('description', ''),
                    is_output_node=node_data.get('output_node', False)
                )

                self._cache[node_name] = node_info
                self._categories[category].append(node_name)

            self._last_fetch = datetime.now()
            return True

        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            print(f"Failed to fetch node catalog: {e}")
            return False

    def get_node(self, name: str) -> Optional[NodeInfo]:
        """Get info for a specific node."""
        return self._cache.get(name)

    def get_nodes_by_category(self, category: str) -> List[NodeInfo]:
        """Get all nodes in a category."""
        return [self._cache[name] for name in self._categories.get(category, [])]

    def search_nodes(self, query: str) -> List[NodeInfo]:
        """Search nodes by name or category."""
        query_lower = query.lower()
        results = []
        for node in self._cache.values():
            if query_lower in node.name.lower() or query_lower in node.category.lower():
                results.append(node)
        return results

    def node_exists(self, name: str) -> bool:
        """Check if a node exists in the catalog."""
        return name in self._cache

    def get_all_categories(self) -> List[str]:
        """Get all node categories."""
        return sorted(self._categories.keys())

    def export_for_llm(self,
                       include_categories: Optional[List[str]] = None,
                       compact: bool = True,
                       max_nodes: int = 500) -> str:
        """Export catalog in LLM-friendly format."""
        if not self._cache:
            self.fetch_catalog()

        lines = [
            "# Available ComfyUI Nodes",
            f"Total nodes: {len(self._cache)}",
            f"Categories: {len(self._categories)}",
            ""
        ]

        count = 0
        for category in sorted(self._categories.keys()):
            if include_categories and category not in include_categories:
                continue

            nodes = self._categories[category]
            lines.append(f"\n## {category} ({len(nodes)} nodes)")

            for node_name in sorted(nodes):
                if count >= max_nodes:
                    lines.append(f"\n... and {len(self._cache) - count} more nodes")
                    break

                node = self._cache[node_name]
                if compact:
                    lines.append(f"  - {node.to_compact_string()}")
                else:
                    lines.append(f"  - {node_name}")
                    if node.description:
                        lines.append(f"    Description: {node.description}")
                    lines.append(f"    Inputs: {json.dumps(node.input_types.get('required', {}))}")
                    lines.append(f"    Outputs: {node.output_types}")
                count += 1

            if count >= max_nodes:
                break

        return "\n".join(lines)

    def export_workflow_relevant_nodes(self, workflow: Dict) -> str:
        """Export only nodes that are used or related to a workflow."""
        if not self._cache:
            self.fetch_catalog()

        used_types = set()
        for node in workflow.get('nodes', []):
            if 'type' in node:
                used_types.add(node['type'])

        # Find related nodes (same category or common connections)
        related_categories = set()
        for node_type in used_types:
            if node := self._cache.get(node_type):
                related_categories.add(node.category)

        lines = [
            "# Workflow-Relevant Nodes",
            f"Used in workflow: {len(used_types)}",
            ""
        ]

        # List used nodes with full details
        lines.append("## Nodes in Workflow")
        for node_type in sorted(used_types):
            if node := self._cache.get(node_type):
                lines.append(f"  - {node.to_compact_string()}")
            else:
                lines.append(f"  - {node_type} [NOT INSTALLED - MISSING]")

        # List related nodes from same categories
        lines.append("\n## Related Nodes (same categories)")
        for category in sorted(related_categories):
            related = [n for n in self._categories.get(category, []) if n not in used_types]
            if related:
                lines.append(f"  {category}: {', '.join(sorted(related)[:10])}")
                if len(related) > 10:
                    lines.append(f"    ... and {len(related) - 10} more")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM SPECS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemSpecs:
    """System hardware specifications."""
    os_name: str
    os_version: str
    cpu_name: str
    cpu_cores: int
    ram_gb: float
    gpu_name: str = "Unknown"
    gpu_vram_gb: float = 0.0
    cuda_version: str = "Unknown"
    python_version: str = ""
    pytorch_version: str = "Unknown"
    comfyui_version: str = "Unknown"

    def to_llm_context(self) -> str:
        """Format specs for LLM context."""
        return f"""# System Specifications
- OS: {self.os_name} {self.os_version}
- CPU: {self.cpu_name} ({self.cpu_cores} cores)
- RAM: {self.ram_gb:.1f} GB
- GPU: {self.gpu_name}
- VRAM: {self.gpu_vram_gb:.1f} GB
- CUDA: {self.cuda_version}
- Python: {self.python_version}
- PyTorch: {self.pytorch_version}
- ComfyUI: {self.comfyui_version}

## Optimization Notes
{self._get_optimization_notes()}
"""

    def _get_optimization_notes(self) -> str:
        notes = []

        if self.gpu_vram_gb > 0:
            if self.gpu_vram_gb <= 4:
                notes.append("- LOW VRAM (<4GB): Use fp8, aggressive tiling, small batches")
            elif self.gpu_vram_gb <= 8:
                notes.append("- MEDIUM VRAM (4-8GB): Can use fp16, moderate tiling")
            elif self.gpu_vram_gb <= 12:
                notes.append("- GOOD VRAM (8-12GB): Most workflows work, may need tiling for high-res")
            else:
                notes.append("- HIGH VRAM (>12GB): Can run most workflows without optimization")

        if self.ram_gb < 16:
            notes.append("- LOW RAM: Avoid loading multiple large models simultaneously")

        if "nvidia" in self.gpu_name.lower():
            notes.append("- NVIDIA GPU: Full CUDA support, can use all accelerations")
        elif "amd" in self.gpu_name.lower():
            notes.append("- AMD GPU: Use ROCm, some nodes may not be compatible")
        elif "intel" in self.gpu_name.lower():
            notes.append("- Intel GPU: Limited support, use CPU fallbacks when needed")

        return "\n".join(notes) if notes else "- Standard configuration"


class SystemSpecsCollector:
    """Collects system specifications."""

    @staticmethod
    def collect() -> SystemSpecs:
        """Collect current system specifications."""
        specs = SystemSpecs(
            os_name=platform.system(),
            os_version=platform.release(),
            cpu_name=platform.processor() or "Unknown",
            cpu_cores=os.cpu_count() or 1,
            ram_gb=SystemSpecsCollector._get_ram_gb(),
            python_version=platform.python_version()
        )

        # Try to get GPU info
        gpu_info = SystemSpecsCollector._get_gpu_info()
        specs.gpu_name = gpu_info.get('name', 'Unknown')
        specs.gpu_vram_gb = gpu_info.get('vram_gb', 0.0)
        specs.cuda_version = gpu_info.get('cuda_version', 'Unknown')

        # Try to get PyTorch version
        try:
            import torch
            specs.pytorch_version = torch.__version__
        except ImportError:
            pass

        return specs

    @staticmethod
    def _get_ram_gb() -> float:
        """Get system RAM in GB."""
        try:
            if platform.system() == "Linux":
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            return kb / (1024 * 1024)
            elif platform.system() == "Darwin":
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                       capture_output=True, text=True)
                return int(result.stdout.strip()) / (1024**3)
            elif platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                class MEMORYSTATUS(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', c_ulong),
                        ('dwMemoryLoad', c_ulong),
                        ('dwTotalPhys', c_ulong),
                        ('dwAvailPhys', c_ulong),
                        ('dwTotalPageFile', c_ulong),
                        ('dwAvailPageFile', c_ulong),
                        ('dwTotalVirtual', c_ulong),
                        ('dwAvailVirtual', c_ulong),
                    ]
                memoryStatus = MEMORYSTATUS()
                memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
                kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
                return memoryStatus.dwTotalPhys / (1024**3)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _get_gpu_info() -> Dict[str, Any]:
        """Get GPU information."""
        info = {'name': 'Unknown', 'vram_gb': 0.0, 'cuda_version': 'Unknown'}

        # Try PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                info['name'] = torch.cuda.get_device_name(0)
                info['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info['cuda_version'] = torch.version.cuda or 'Unknown'
                return info
        except Exception:
            pass

        # Try nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 2:
                    info['name'] = parts[0]
                    info['vram_gb'] = float(parts[1]) / 1024
        except Exception:
            pass

        return info


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

class PromptGoal(Enum):
    """Goals for workflow optimization."""
    DEBUG = "debug"
    QUALITY = "quality"
    SPEED = "speed"
    VRAM = "vram"
    EXPLAIN = "explain"
    COMPARE = "compare"
    DISTRIBUTED = "distributed"  # Multi-machine optimization
    CUSTOM = "custom"


@dataclass
class PromptTemplate:
    """A template for generating LLM prompts."""
    name: str
    goal: PromptGoal
    template: str
    description: str
    requires_workflow: bool = True
    requires_error: bool = False
    requires_metrics: bool = False


PROMPT_TEMPLATES: Dict[PromptGoal, PromptTemplate] = {
    PromptGoal.DEBUG: PromptTemplate(
        name="Debug Workflow",
        goal=PromptGoal.DEBUG,
        description="Find and fix errors in a workflow",
        requires_error=True,
        template="""# Debug Request

## Problem Description
{error_description}

## Error Details
```
{error_log}
```

## Workflow JSON
```json
{workflow_json}
```

## System Context
{system_specs}

## Available Nodes
{relevant_nodes}

## Instructions
Analyze this ComfyUI workflow that is producing errors. Please:
1. Identify the root cause of the error
2. Check for missing nodes or incorrect connections
3. Verify widget values are within valid ranges
4. Provide a corrected workflow JSON that fixes the issue

Return your fix in this format:
```json
{{
  "diagnosis": "explanation of what's wrong",
  "fix_type": "node_change|connection_change|value_change|missing_node",
  "changes": [
    {{"node_id": "X", "change": "description", "before": "...", "after": "..."}}
  ],
  "workflow": {{ ... corrected workflow JSON ... }}
}}
```
"""
    ),

    PromptGoal.QUALITY: PromptTemplate(
        name="Improve Quality",
        goal=PromptGoal.QUALITY,
        description="Optimize workflow for better output quality",
        requires_metrics=True,
        template="""# Quality Optimization Request

## Current Workflow
```json
{workflow_json}
```

## Current Performance Metrics
{metrics}

## System Specifications
{system_specs}

## Available Nodes
{relevant_nodes}

## Goal
Optimize this workflow to produce higher quality outputs while maintaining reasonable generation times.

## Suggestions to Consider
- Increase sampling steps if currently low
- Use better samplers (DPM++ 2M Karras, UniPC)
- Add upscaling nodes for higher resolution
- Include refinement passes
- Adjust CFG scale for better prompt adherence
- Add ControlNet for structural consistency

## Instructions
Provide an optimized workflow with improved quality settings. Explain each change and its impact on quality.

Return your optimization in this format:
```json
{{
  "optimization_summary": "brief description of changes",
  "quality_improvements": ["list", "of", "improvements"],
  "trade_offs": ["any", "trade-offs", "made"],
  "estimated_quality_gain": "low|medium|high",
  "estimated_time_impact": "+X% or -X%",
  "workflow": {{ ... optimized workflow JSON ... }}
}}
```
"""
    ),

    PromptGoal.SPEED: PromptTemplate(
        name="Optimize Speed",
        goal=PromptGoal.SPEED,
        description="Make workflow faster while maintaining acceptable quality",
        requires_metrics=True,
        template="""# Speed Optimization Request

## Current Workflow
```json
{workflow_json}
```

## Current Performance Metrics
{metrics}

## System Specifications
{system_specs}

## Available Nodes
{relevant_nodes}

## Goal
Make this workflow run faster while maintaining acceptable output quality.

## Suggestions to Consider
- Reduce sampling steps (find optimal minimum)
- Use faster samplers (Euler, LCM, Lightning)
- Lower resolution during generation, upscale after
- Enable model optimizations (fp16, fp8)
- Use caching where possible
- Remove unnecessary nodes
- Batch operations efficiently

## Instructions
Provide a speed-optimized workflow. Quantify expected speed improvements.

Return your optimization in this format:
```json
{{
  "optimization_summary": "brief description of changes",
  "speed_improvements": ["list", "of", "changes"],
  "quality_impact": "none|minimal|moderate|significant",
  "estimated_speedup": "X% faster",
  "workflow": {{ ... optimized workflow JSON ... }}
}}
```
"""
    ),

    PromptGoal.VRAM: PromptTemplate(
        name="Reduce VRAM",
        goal=PromptGoal.VRAM,
        description="Optimize workflow to use less GPU memory",
        requires_metrics=True,
        template="""# VRAM Optimization Request

## Current Workflow
```json
{workflow_json}
```

## Current Performance Metrics
{metrics}

## System Specifications
{system_specs}

## Available Nodes
{relevant_nodes}

## Target VRAM
{target_vram} GB (current system has {current_vram} GB)

## Goal
Reduce VRAM usage so this workflow can run on systems with limited GPU memory.

## Suggestions to Consider
- Use tiled VAE encoding/decoding
- Switch to fp16 or fp8 models
- Reduce batch size to 1
- Lower resolution
- Use memory-efficient attention
- Unload models between steps
- Split workflow into sequential parts

## Instructions
Provide a VRAM-optimized workflow that can run within the target VRAM limit.

Return your optimization in this format:
```json
{{
  "optimization_summary": "brief description of changes",
  "vram_reductions": ["list", "of", "changes"],
  "estimated_vram_usage": "X GB",
  "quality_impact": "none|minimal|moderate|significant",
  "speed_impact": "faster|same|slower",
  "workflow": {{ ... optimized workflow JSON ... }}
}}
```
"""
    ),

    PromptGoal.EXPLAIN: PromptTemplate(
        name="Explain Workflow",
        goal=PromptGoal.EXPLAIN,
        description="Get a detailed explanation of how a workflow works",
        template="""# Workflow Explanation Request

## Workflow JSON
```json
{workflow_json}
```

## System Context
{system_specs}

## Instructions
Please provide a comprehensive explanation of this ComfyUI workflow:

1. **Overview**: What does this workflow do? What is it designed to generate?

2. **Node Analysis**: For each node, explain:
   - Its purpose in the workflow
   - Key settings and their effects
   - How it connects to other nodes

3. **Data Flow**: Trace the flow of data through the workflow:
   - Where do inputs come from?
   - How is data transformed at each step?
   - What are the outputs?

4. **Critical Settings**: Identify the most important parameters that affect:
   - Output quality
   - Generation speed
   - VRAM usage

5. **Potential Improvements**: Suggest areas where the workflow could be enhanced.

Format your response as a clear, educational guide that someone new to ComfyUI could understand.
"""
    ),

    PromptGoal.COMPARE: PromptTemplate(
        name="Compare Workflows",
        goal=PromptGoal.COMPARE,
        description="Compare two workflows and explain differences",
        template="""# Workflow Comparison Request

## Workflow A (Original)
```json
{workflow_a_json}
```

## Workflow B (Modified)
```json
{workflow_b_json}
```

## Metrics Comparison
### Workflow A
{metrics_a}

### Workflow B
{metrics_b}

## Instructions
Compare these two workflows and provide:

1. **Structural Differences**:
   - Nodes added/removed
   - Connection changes
   - Node repositioning

2. **Parameter Differences**:
   - Changed settings
   - Different model selections
   - Altered sampling parameters

3. **Impact Analysis**:
   - How do the changes affect output quality?
   - How do the changes affect speed?
   - How do the changes affect VRAM usage?

4. **Recommendation**:
   - Which workflow is better for quality?
   - Which workflow is better for speed?
   - When would you use each one?

Provide a clear summary table of differences.
"""
    ),

    PromptGoal.DISTRIBUTED: PromptTemplate(
        name="Distributed Optimization",
        goal=PromptGoal.DISTRIBUTED,
        description="Optimize multi-machine AI pipeline",
        template="""# Distributed AI Pipeline Optimization Request

## Goal
Optimize this distributed workflow that orchestrates multiple AI services across different machines.

## Workflow JSON
```json
{workflow_json}
```

## Network Topology
{network_topology}

## Machine Profiles
{machine_profiles}

## Current Latencies
{latencies}

## Dependency Graph
{dependency_graph}

## Detected Bottlenecks
{bottlenecks}

## System Context
{system_specs}

## Considerations
- Network latency between services
- GPU VRAM and memory constraints per machine
- CPU vs GPU workload distribution
- Opportunities for parallel execution
- Model quantization options (int8, int4, q4_k_m, etc.)
- Context window sizes for LLMs
- Batch sizes for image generation
- Failure handling and retry strategies
- Load balancing if multiple endpoints available

## Instructions
Based on this distributed AI pipeline analysis, please provide:

1. **Bottleneck Analysis**: Which machine/service is slowing down the pipeline and why?

2. **Hardware Recommendations**: Based on the machine specs:
   - Should any service move to a different machine?
   - Are there VRAM/RAM constraints causing issues?
   - Would different quantization help?

3. **Configuration Recommendations**: Specific settings to adjust:
   - Context sizes for LLMs
   - Batch sizes for image generation
   - Quality vs speed tradeoffs
   - Suggested model alternatives

4. **Architecture Recommendations**: Pipeline restructuring:
   - Which operations can run in parallel?
   - Should any operations be combined or split?
   - Would caching help anywhere?

5. **Optimized Workflow**: Provide the modified workflow JSON with your recommendations applied.

Return your response in this format:
```json
{{
  "bottleneck_analysis": {{
    "primary_bottleneck": "description",
    "machines_affected": ["endpoint1", "endpoint2"],
    "root_cause": "explanation"
  }},
  "recommendations": {{
    "hardware": ["recommendation1", "recommendation2"],
    "configuration": ["recommendation1", "recommendation2"],
    "architecture": ["recommendation1", "recommendation2"]
  }},
  "estimated_improvement": "X% faster / Y% less VRAM",
  "workflow": {{ ... optimized workflow JSON ... }}
}}
```

Please be specific and actionable. Reference the actual machine specs and endpoints provided.
"""
    ),

    PromptGoal.CUSTOM: PromptTemplate(
        name="Custom Request",
        goal=PromptGoal.CUSTOM,
        description="Custom optimization request",
        template="""# Custom Workflow Request

## Workflow JSON
```json
{workflow_json}
```

## System Specifications
{system_specs}

## Available Nodes
{relevant_nodes}

## Your Request
{custom_request}

## Instructions
Please address the custom request above. Provide:
1. Analysis of the current workflow
2. Proposed changes to meet the request
3. Complete modified workflow JSON

Return your response in this format:
```json
{{
  "analysis": "your analysis",
  "proposed_changes": ["list", "of", "changes"],
  "workflow": {{ ... modified workflow JSON ... }}
}}
```
"""
    )
}


class PromptGenerator:
    """Generates LLM prompts from templates."""

    def __init__(self, node_catalog: Optional[NodeCatalog] = None):
        self.node_catalog = node_catalog or NodeCatalog()
        self.specs_collector = SystemSpecsCollector()

    def generate(self,
                 goal: PromptGoal,
                 workflow: Optional[Dict] = None,
                 metrics: Optional[Dict] = None,
                 error_log: str = "",
                 error_description: str = "",
                 custom_request: str = "",
                 workflow_b: Optional[Dict] = None,
                 metrics_b: Optional[Dict] = None,
                 target_vram: float = 6.0) -> str:
        """Generate a prompt from a template."""

        template = PROMPT_TEMPLATES.get(goal)
        if not template:
            raise ValueError(f"Unknown goal: {goal}")

        # Collect system specs
        specs = self.specs_collector.collect()

        # Get relevant nodes
        if workflow:
            self.node_catalog.fetch_catalog()
            relevant_nodes = self.node_catalog.export_workflow_relevant_nodes(workflow)
        else:
            relevant_nodes = "No workflow provided"

        # Format workflow JSON
        workflow_json = json.dumps(workflow, indent=2) if workflow else "{}"

        # Format metrics
        metrics_str = self._format_metrics(metrics) if metrics else "No metrics available"

        # Build context dict
        context = {
            'workflow_json': workflow_json,
            'system_specs': specs.to_llm_context(),
            'relevant_nodes': relevant_nodes,
            'metrics': metrics_str,
            'error_log': error_log,
            'error_description': error_description,
            'custom_request': custom_request,
            'target_vram': target_vram,
            'current_vram': specs.gpu_vram_gb,
        }

        # Handle comparison
        if goal == PromptGoal.COMPARE:
            context['workflow_a_json'] = workflow_json
            context['workflow_b_json'] = json.dumps(workflow_b, indent=2) if workflow_b else "{}"
            context['metrics_a'] = metrics_str
            context['metrics_b'] = self._format_metrics(metrics_b) if metrics_b else "No metrics"

        # Fill template
        prompt = template.template.format(**context)

        return prompt

    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics for inclusion in prompt."""
        if not metrics:
            return "No metrics available"

        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MOD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of validating a workflow modification."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    fixed_workflow: Optional[Dict] = None

    def to_string(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("✓ Validation PASSED")
        else:
            lines.append("✗ Validation FAILED")

        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        if self.suggestions:
            lines.append("\nSuggestions:")
            for s in self.suggestions:
                lines.append(f"  → {s}")

        return "\n".join(lines)


class ModValidator:
    """Validates LLM-generated workflow modifications."""

    def __init__(self, node_catalog: Optional[NodeCatalog] = None):
        self.node_catalog = node_catalog or NodeCatalog()

    def validate(self,
                 mod_json: str,
                 original_workflow: Optional[Dict] = None,
                 auto_fix: bool = True) -> ValidationResult:
        """Validate a workflow modification from LLM."""

        result = ValidationResult(is_valid=True)

        # Parse JSON
        try:
            mod_data = json.loads(mod_json)
        except json.JSONDecodeError as e:
            result.is_valid = False
            result.errors.append(f"Invalid JSON: {e}")

            # Try to fix common JSON issues
            if auto_fix:
                fixed = self._try_fix_json(mod_json)
                if fixed:
                    result.suggestions.append("Attempted to fix JSON syntax")
                    mod_data = fixed
                else:
                    return result
            else:
                return result

        # Extract workflow from various formats
        workflow = self._extract_workflow(mod_data)
        if not workflow:
            result.is_valid = False
            result.errors.append("Could not find workflow in response")
            return result

        # Validate structure
        self._validate_structure(workflow, result)

        # Validate nodes
        self._validate_nodes(workflow, result)

        # Validate links
        self._validate_links(workflow, result)

        # Check for breaking changes
        if original_workflow:
            self._check_breaking_changes(workflow, original_workflow, result)

        # Store fixed workflow if valid
        if result.is_valid or (result.warnings and not result.errors):
            result.fixed_workflow = workflow

        return result

    def _try_fix_json(self, json_str: str) -> Optional[Dict]:
        """Attempt to fix common JSON issues."""
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)

        # Try to find JSON object
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try fixing trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None

    def _extract_workflow(self, data: Dict) -> Optional[Dict]:
        """Extract workflow from various response formats."""
        # Direct workflow
        if 'nodes' in data and 'links' in data:
            return data

        # Nested under 'workflow' key
        if 'workflow' in data:
            return data['workflow']

        # Nested under 'modified_workflow'
        if 'modified_workflow' in data:
            return data['modified_workflow']

        # Nested under 'result'
        if 'result' in data and isinstance(data['result'], dict):
            return self._extract_workflow(data['result'])

        return None

    def _validate_structure(self, workflow: Dict, result: ValidationResult):
        """Validate basic workflow structure."""
        required_keys = ['nodes', 'links']

        for key in required_keys:
            if key not in workflow:
                result.is_valid = False
                result.errors.append(f"Missing required key: '{key}'")

        if 'nodes' in workflow and not isinstance(workflow['nodes'], list):
            result.is_valid = False
            result.errors.append("'nodes' must be a list")

        if 'links' in workflow and not isinstance(workflow['links'], list):
            result.is_valid = False
            result.errors.append("'links' must be a list")

    def _validate_nodes(self, workflow: Dict, result: ValidationResult):
        """Validate workflow nodes."""
        self.node_catalog.fetch_catalog()

        node_ids = set()

        for node in workflow.get('nodes', []):
            # Check required node fields
            if 'id' not in node:
                result.warnings.append(f"Node missing 'id' field: {node.get('type', 'unknown')}")
                continue

            node_id = node['id']

            # Check for duplicate IDs
            if node_id in node_ids:
                result.is_valid = False
                result.errors.append(f"Duplicate node ID: {node_id}")
            node_ids.add(node_id)

            # Check node type exists
            node_type = node.get('type')
            if node_type and not self.node_catalog.node_exists(node_type):
                result.warnings.append(f"Node type may not be installed: {node_type}")

            # Validate widget values
            if 'widgets_values' in node and node_type:
                self._validate_widgets(node, result)

    def _validate_widgets(self, node: Dict, result: ValidationResult):
        """Validate widget values for a node."""
        node_info = self.node_catalog.get_node(node.get('type', ''))
        if not node_info:
            return

        widgets = node.get('widgets_values', [])

        # Check for common issues
        for i, value in enumerate(widgets):
            # Null values
            if value is None:
                result.warnings.append(
                    f"Node {node['id']} ({node['type']}): widget {i} is null"
                )

            # Negative dimensions
            if isinstance(value, (int, float)) and value < 0:
                # Check if it's likely a dimension
                if i < len(widgets) - 1:
                    result.warnings.append(
                        f"Node {node['id']}: widget {i} has negative value {value}"
                    )

    def _validate_links(self, workflow: Dict, result: ValidationResult):
        """Validate workflow links."""
        node_ids = {node['id'] for node in workflow.get('nodes', []) if 'id' in node}

        for link in workflow.get('links', []):
            if not isinstance(link, list) or len(link) < 5:
                result.warnings.append(f"Malformed link: {link}")
                continue

            link_id, src_node, src_slot, dst_node, dst_slot = link[:5]

            # Check source node exists
            if src_node not in node_ids:
                result.is_valid = False
                result.errors.append(f"Link {link_id}: source node {src_node} doesn't exist")

            # Check destination node exists
            if dst_node not in node_ids:
                result.is_valid = False
                result.errors.append(f"Link {link_id}: destination node {dst_node} doesn't exist")

    def _check_breaking_changes(self,
                                new_workflow: Dict,
                                original: Dict,
                                result: ValidationResult):
        """Check for potentially breaking changes."""
        orig_nodes = {n['id']: n for n in original.get('nodes', []) if 'id' in n}
        new_nodes = {n['id']: n for n in new_workflow.get('nodes', []) if 'id' in n}

        # Check for removed nodes
        removed = set(orig_nodes.keys()) - set(new_nodes.keys())
        if removed:
            result.warnings.append(f"Nodes removed: {removed}")

        # Check for type changes
        for node_id in set(orig_nodes.keys()) & set(new_nodes.keys()):
            orig_type = orig_nodes[node_id].get('type')
            new_type = new_nodes[node_id].get('type')
            if orig_type != new_type:
                result.warnings.append(
                    f"Node {node_id} type changed: {orig_type} -> {new_type}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationEntry:
    """A single entry in conversation history."""
    timestamp: str
    workflow_hash: str
    goal: str
    user_message: str
    llm_response: str
    applied: bool = False
    success: bool = False
    notes: str = ""


class ConversationMemory:
    """Persists conversation context across sessions."""

    def __init__(self, db_path: Path = MEMORY_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    workflow_hash TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    llm_response TEXT NOT NULL,
                    applied INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 0,
                    notes TEXT DEFAULT ''
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_metadata (
                    workflow_hash TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    model_family TEXT,
                    first_seen TEXT,
                    last_modified TEXT,
                    optimization_count INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_hash
                ON conversations(workflow_hash)
            """)

    @staticmethod
    def hash_workflow(workflow: Dict) -> str:
        """Generate a stable hash for a workflow."""
        # Normalize workflow for hashing
        normalized = json.dumps(workflow, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def add_entry(self,
                  workflow: Dict,
                  goal: str,
                  user_message: str,
                  llm_response: str,
                  applied: bool = False,
                  success: bool = False,
                  notes: str = "") -> int:
        """Add a conversation entry."""
        workflow_hash = self.hash_workflow(workflow)
        timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO conversations
                (timestamp, workflow_hash, goal, user_message, llm_response, applied, success, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, workflow_hash, goal, user_message, llm_response,
                  int(applied), int(success), notes))

            # Update workflow metadata
            conn.execute("""
                INSERT INTO workflow_metadata (workflow_hash, first_seen, last_modified, optimization_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(workflow_hash) DO UPDATE SET
                    last_modified = excluded.first_seen,
                    optimization_count = optimization_count + 1
            """, (workflow_hash, timestamp, timestamp))

            return cursor.lastrowid

    def get_history(self,
                    workflow: Optional[Dict] = None,
                    limit: int = 10) -> List[ConversationEntry]:
        """Get conversation history, optionally filtered by workflow."""
        with sqlite3.connect(self.db_path) as conn:
            if workflow:
                workflow_hash = self.hash_workflow(workflow)
                rows = conn.execute("""
                    SELECT timestamp, workflow_hash, goal, user_message, llm_response,
                           applied, success, notes
                    FROM conversations
                    WHERE workflow_hash = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (workflow_hash, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT timestamp, workflow_hash, goal, user_message, llm_response,
                           applied, success, notes
                    FROM conversations
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,)).fetchall()

        return [ConversationEntry(
            timestamp=row[0],
            workflow_hash=row[1],
            goal=row[2],
            user_message=row[3],
            llm_response=row[4],
            applied=bool(row[5]),
            success=bool(row[6]),
            notes=row[7]
        ) for row in rows]

    def mark_applied(self, entry_id: int, success: bool, notes: str = ""):
        """Mark a conversation entry as applied."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE conversations
                SET applied = 1, success = ?, notes = ?
                WHERE id = ?
            """, (int(success), notes, entry_id))

    def get_context_for_llm(self, workflow: Dict, max_entries: int = 5) -> str:
        """Get conversation history formatted for LLM context."""
        history = self.get_history(workflow, limit=max_entries)

        if not history:
            return "No previous optimization history for this workflow."

        lines = ["# Previous Optimization History", ""]

        for entry in reversed(history):
            status = "✓ Applied successfully" if entry.success else "✗ Not applied" if entry.applied else "Pending"
            lines.append(f"## {entry.timestamp} - {entry.goal}")
            lines.append(f"Status: {status}")
            lines.append(f"Request: {entry.user_message[:200]}...")
            if entry.notes:
                lines.append(f"Notes: {entry.notes}")
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            applied = conn.execute("SELECT COUNT(*) FROM conversations WHERE applied = 1").fetchone()[0]
            successful = conn.execute("SELECT COUNT(*) FROM conversations WHERE success = 1").fetchone()[0]
            workflows = conn.execute("SELECT COUNT(*) FROM workflow_metadata").fetchone()[0]

            goals = conn.execute("""
                SELECT goal, COUNT(*) FROM conversations GROUP BY goal
            """).fetchall()

        return {
            'total_conversations': total,
            'applied': applied,
            'successful': successful,
            'unique_workflows': workflows,
            'success_rate': successful / applied if applied > 0 else 0,
            'goals': dict(goals)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeEntry:
    """An entry in the community knowledge base."""
    id: str
    category: str
    title: str
    problem: str
    solution: str
    tags: List[str]
    node_types: List[str] = field(default_factory=list)
    model_families: List[str] = field(default_factory=list)
    votes: int = 0

    def matches_workflow(self, workflow: Dict) -> bool:
        """Check if this entry is relevant to a workflow."""
        workflow_nodes = {n.get('type', '') for n in workflow.get('nodes', [])}

        # Check node type overlap
        if self.node_types:
            if not any(nt in workflow_nodes for nt in self.node_types):
                return False

        return True

    def to_llm_context(self) -> str:
        """Format entry for LLM context."""
        return f"""### {self.title}
**Category**: {self.category}
**Tags**: {', '.join(self.tags)}

**Problem**: {self.problem}

**Solution**: {self.solution}
"""


# Default knowledge base entries
DEFAULT_KNOWLEDGE_BASE: List[Dict] = [
    {
        "id": "vram_oom",
        "category": "VRAM",
        "title": "Out of Memory (OOM) Error",
        "problem": "CUDA out of memory error when running workflows with large models",
        "solution": """1. Enable tiled VAE (add VAE Decode Tiled node)
2. Reduce batch size to 1
3. Lower resolution during generation
4. Use fp16 or fp8 models
5. Enable --lowvram or --medvram flag
6. Close other GPU applications""",
        "tags": ["oom", "cuda", "memory", "vram"],
        "node_types": ["VAEDecode", "KSampler"],
        "model_families": ["SDXL", "SD3", "Flux"]
    },
    {
        "id": "black_image",
        "category": "Quality",
        "title": "Black or Corrupted Output Image",
        "problem": "Generated images are completely black or have corrupted patterns",
        "solution": """1. Check VAE model matches checkpoint (SD 1.5 vs SDXL)
2. Verify CFG scale isn't too high (try 7-8)
3. Check clip skip settings
4. Ensure proper model loading order
5. Try different sampler (euler_ancestral is reliable)
6. Check for NaN values in conditioning""",
        "tags": ["black", "corrupted", "quality", "vae"],
        "node_types": ["VAEDecode", "KSampler", "CheckpointLoaderSimple"]
    },
    {
        "id": "slow_generation",
        "category": "Speed",
        "title": "Slow Generation Speed",
        "problem": "Image generation takes much longer than expected",
        "solution": """1. Ensure CUDA is being used (check ComfyUI startup logs)
2. Reduce sampling steps (20-25 is often enough)
3. Use faster samplers (euler, dpmpp_2m)
4. Enable fp16/bf16 if supported
5. Disable preview during generation
6. Check for CPU offloading (indicates VRAM issues)""",
        "tags": ["slow", "speed", "performance", "cuda"],
        "node_types": ["KSampler", "KSamplerAdvanced"]
    },
    {
        "id": "flux_cfg",
        "category": "Models",
        "title": "Flux Model CFG Settings",
        "problem": "Flux outputs look oversaturated or distorted",
        "solution": """Flux models require different settings:
1. CFG should be 1.0-4.0 (NOT 7-8 like SD)
2. Use Flux-specific samplers when available
3. Don't use negative prompts (Flux handles them differently)
4. Enable Flux-specific guidance if using dev model""",
        "tags": ["flux", "cfg", "settings", "model"],
        "node_types": ["KSampler"],
        "model_families": ["Flux"]
    },
    {
        "id": "missing_node",
        "category": "Nodes",
        "title": "Missing Custom Node Error",
        "problem": "Workflow fails with 'node type not found' error",
        "solution": """1. Install missing node via ComfyUI Manager
2. Check node name spelling (case sensitive)
3. Restart ComfyUI after installing
4. Check node compatibility with ComfyUI version
5. Look for alternative nodes with similar functionality""",
        "tags": ["missing", "node", "error", "install"],
        "node_types": []
    },
    {
        "id": "controlnet_artifacts",
        "category": "Quality",
        "title": "ControlNet Causing Artifacts",
        "problem": "ControlNet produces unwanted patterns or artifacts",
        "solution": """1. Reduce ControlNet strength (0.5-0.8)
2. Use appropriate preprocessor for input type
3. Match ControlNet model to base model (SD vs SDXL)
4. Apply ControlNet to fewer steps (start/end percentage)
5. Ensure input image resolution matches output""",
        "tags": ["controlnet", "artifacts", "quality"],
        "node_types": ["ControlNetApply", "ControlNetLoader"]
    },
    {
        "id": "lora_strength",
        "category": "Models",
        "title": "LoRA Not Having Effect",
        "problem": "LoRA doesn't seem to change the output",
        "solution": """1. Increase LoRA strength (try 0.7-1.0)
2. Check LoRA is compatible with model (SD vs SDXL)
3. Ensure LoRA is properly loaded before sampling
4. Check trigger words are in prompt
5. Try adjusting model_strength vs clip_strength separately""",
        "tags": ["lora", "strength", "model"],
        "node_types": ["LoraLoader", "LoraLoaderModelOnly"]
    },
    {
        "id": "resolution_mismatch",
        "category": "Quality",
        "title": "Wrong Resolution Output",
        "problem": "Output doesn't match expected resolution or aspect ratio",
        "solution": """1. Check Empty Latent Image dimensions
2. Verify upscale nodes have correct settings
3. For SDXL, use 1024x1024 base (not 512x512)
4. For SD 1.5, use 512x512 base
5. Check for aspect ratio nodes affecting output""",
        "tags": ["resolution", "size", "dimensions"],
        "node_types": ["EmptyLatentImage", "ImageScale", "LatentUpscale"]
    }
]


class KnowledgeBase:
    """Community knowledge base for common issues and solutions."""

    def __init__(self, path: Path = KNOWLEDGE_BASE_PATH):
        self.path = path
        self.entries: List[KnowledgeEntry] = []
        self._load()

    def _load(self):
        """Load knowledge base from file."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                    self.entries = [KnowledgeEntry(**e) for e in data]
            except (json.JSONDecodeError, TypeError):
                self._init_default()
        else:
            self._init_default()

    def _init_default(self):
        """Initialize with default knowledge base."""
        self.entries = [KnowledgeEntry(**e) for e in DEFAULT_KNOWLEDGE_BASE]
        self.save()

    def save(self):
        """Save knowledge base to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def search(self,
               query: str = "",
               tags: Optional[List[str]] = None,
               category: Optional[str] = None,
               workflow: Optional[Dict] = None) -> List[KnowledgeEntry]:
        """Search knowledge base."""
        results = []
        query_lower = query.lower()

        for entry in self.entries:
            # Filter by category
            if category and entry.category != category:
                continue

            # Filter by tags
            if tags and not any(t in entry.tags for t in tags):
                continue

            # Filter by workflow relevance
            if workflow and not entry.matches_workflow(workflow):
                continue

            # Search by query
            if query:
                searchable = f"{entry.title} {entry.problem} {entry.solution}".lower()
                if query_lower not in searchable:
                    continue

            results.append(entry)

        return sorted(results, key=lambda e: e.votes, reverse=True)

    def get_relevant_for_workflow(self, workflow: Dict, limit: int = 5) -> List[KnowledgeEntry]:
        """Get knowledge entries relevant to a workflow."""
        return self.search(workflow=workflow)[:limit]

    def get_relevant_for_error(self, error_text: str, limit: int = 3) -> List[KnowledgeEntry]:
        """Get knowledge entries relevant to an error."""
        error_lower = error_text.lower()

        scored = []
        for entry in self.entries:
            score = 0

            # Check for tag matches
            for tag in entry.tags:
                if tag in error_lower:
                    score += 2

            # Check title/problem match
            if any(word in error_lower for word in entry.title.lower().split()):
                score += 1

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def add_entry(self, entry: KnowledgeEntry):
        """Add a new knowledge entry."""
        self.entries.append(entry)
        self.save()

    def export_for_llm(self, workflow: Optional[Dict] = None) -> str:
        """Export relevant knowledge for LLM context."""
        if workflow:
            relevant = self.get_relevant_for_workflow(workflow)
        else:
            relevant = self.entries[:10]

        if not relevant:
            return ""

        lines = ["# Relevant Knowledge Base Entries", ""]
        for entry in relevant:
            lines.append(entry.to_llm_context())
            lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW GRAPH EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowGraphExporter:
    """Exports workflow as ASCII graph visualization."""

    @staticmethod
    def export_ascii(workflow: Dict, max_width: int = 100) -> str:
        """Export workflow as ASCII diagram."""
        nodes = {n['id']: n for n in workflow.get('nodes', []) if 'id' in n}
        links = workflow.get('links', [])

        # Build adjacency list
        incoming: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        outgoing: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

        for link in links:
            if len(link) >= 5:
                link_id, src, src_slot, dst, dst_slot = link[:5]
                link_type = link[5] if len(link) > 5 else "?"
                outgoing[src].append((dst, link_type))
                incoming[dst].append((src, link_type))

        # Find root nodes (no incoming connections)
        roots = [nid for nid in nodes if nid not in incoming]
        if not roots:
            roots = list(nodes.keys())[:1]

        # Generate ASCII representation
        lines = [
            "# Workflow Graph",
            f"Nodes: {len(nodes)} | Connections: {len(links)}",
            ""
        ]

        # Simple text representation
        visited = set()

        def format_node(nid: int, depth: int = 0) -> List[str]:
            if nid in visited or nid not in nodes:
                return []
            visited.add(nid)

            node = nodes[nid]
            node_type = node.get('type', 'Unknown')
            prefix = "  " * depth

            result = [f"{prefix}[{nid}] {node_type}"]

            # Show connections
            for dst, link_type in outgoing.get(nid, []):
                if dst in nodes:
                    dst_type = nodes[dst].get('type', 'Unknown')
                    result.append(f"{prefix}  └─({link_type})─> [{dst}] {dst_type}")
                    result.extend(format_node(dst, depth + 2))

            return result

        for root in roots:
            lines.extend(format_node(root))
            lines.append("")

        # Show any unvisited nodes
        unvisited = set(nodes.keys()) - visited
        if unvisited:
            lines.append("# Disconnected Nodes")
            for nid in unvisited:
                node = nodes[nid]
                lines.append(f"  [{nid}] {node.get('type', 'Unknown')}")

        return "\n".join(lines)

    @staticmethod
    def export_mermaid(workflow: Dict) -> str:
        """Export workflow as Mermaid diagram."""
        nodes = {n['id']: n for n in workflow.get('nodes', []) if 'id' in n}
        links = workflow.get('links', [])

        lines = ["```mermaid", "graph LR"]

        # Add nodes
        for nid, node in nodes.items():
            node_type = node.get('type', 'Unknown').replace(' ', '_')
            lines.append(f"    {nid}[{node_type}]")

        # Add connections
        for link in links:
            if len(link) >= 5:
                _, src, _, dst, _ = link[:5]
                link_type = link[5] if len(link) > 5 else ""
                if src in nodes and dst in nodes:
                    lines.append(f"    {src} -->|{link_type}| {dst}")

        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def export_summary(workflow: Dict) -> str:
        """Export workflow structure summary."""
        nodes = workflow.get('nodes', [])
        links = workflow.get('links', [])

        # Count node types
        type_counts: Dict[str, int] = defaultdict(int)
        for node in nodes:
            type_counts[node.get('type', 'Unknown')] += 1

        # Identify key nodes
        input_nodes = []
        output_nodes = []
        sampling_nodes = []

        for node in nodes:
            node_type = node.get('type', '')
            if 'Load' in node_type or 'Input' in node_type:
                input_nodes.append(node_type)
            elif 'Save' in node_type or 'Preview' in node_type or 'Output' in node_type:
                output_nodes.append(node_type)
            elif 'Sampler' in node_type or 'KSampler' in node_type:
                sampling_nodes.append(node_type)

        lines = [
            "# Workflow Structure Summary",
            "",
            f"**Total Nodes**: {len(nodes)}",
            f"**Total Connections**: {len(links)}",
            "",
            "## Node Types",
        ]

        for node_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {node_type}: {count}")

        lines.extend([
            "",
            "## Key Components",
            f"  - Input nodes: {', '.join(input_nodes) or 'None identified'}",
            f"  - Sampling nodes: {', '.join(sampling_nodes) or 'None identified'}",
            f"  - Output nodes: {', '.join(output_nodes) or 'None identified'}",
        ])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ErrorEntry:
    """An error entry in history."""
    timestamp: str
    error_type: str
    message: str
    node_id: Optional[int] = None
    node_type: Optional[str] = None
    workflow_hash: Optional[str] = None
    resolved: bool = False
    resolution: str = ""


class ErrorHistory:
    """Tracks and manages error history."""

    def __init__(self, db_path: Path = MEMORY_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize error history table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    node_id INTEGER,
                    node_type TEXT,
                    workflow_hash TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolution TEXT DEFAULT ''
                )
            """)

    def add_error(self,
                  error_type: str,
                  message: str,
                  node_id: Optional[int] = None,
                  node_type: Optional[str] = None,
                  workflow: Optional[Dict] = None) -> int:
        """Add an error to history."""
        workflow_hash = None
        if workflow:
            workflow_hash = ConversationMemory.hash_workflow(workflow)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO error_history
                (timestamp, error_type, message, node_id, node_type, workflow_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), error_type, message,
                  node_id, node_type, workflow_hash))
            return cursor.lastrowid

    def get_recent(self, limit: int = 10, workflow: Optional[Dict] = None) -> List[ErrorEntry]:
        """Get recent errors."""
        with sqlite3.connect(self.db_path) as conn:
            if workflow:
                workflow_hash = ConversationMemory.hash_workflow(workflow)
                rows = conn.execute("""
                    SELECT timestamp, error_type, message, node_id, node_type,
                           workflow_hash, resolved, resolution
                    FROM error_history
                    WHERE workflow_hash = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (workflow_hash, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT timestamp, error_type, message, node_id, node_type,
                           workflow_hash, resolved, resolution
                    FROM error_history
                    ORDER BY timestamp DESC LIMIT ?
                """, (limit,)).fetchall()

        return [ErrorEntry(
            timestamp=r[0], error_type=r[1], message=r[2],
            node_id=r[3], node_type=r[4], workflow_hash=r[5],
            resolved=bool(r[6]), resolution=r[7]
        ) for r in rows]

    def mark_resolved(self, error_id: int, resolution: str = ""):
        """Mark an error as resolved."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE error_history
                SET resolved = 1, resolution = ?
                WHERE id = ?
            """, (resolution, error_id))

    def export_for_llm(self, workflow: Optional[Dict] = None, limit: int = 5) -> str:
        """Export error history for LLM context."""
        errors = self.get_recent(limit=limit, workflow=workflow)

        if not errors:
            return ""

        lines = ["# Recent Error History", ""]

        for error in errors:
            status = "✓ Resolved" if error.resolved else "✗ Unresolved"
            lines.append(f"## {error.timestamp} - {error.error_type}")
            lines.append(f"Status: {status}")
            lines.append(f"Message: {error.message}")
            if error.node_type:
                lines.append(f"Node: [{error.node_id}] {error.node_type}")
            if error.resolved and error.resolution:
                lines.append(f"Resolution: {error.resolution}")
            lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED LLM CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class LLMContextBuilder:
    """Builds comprehensive context for LLM interactions."""

    def __init__(self):
        self.node_catalog = NodeCatalog()
        self.specs_collector = SystemSpecsCollector()
        self.prompt_generator = PromptGenerator(self.node_catalog)
        self.validator = ModValidator(self.node_catalog)
        self.memory = ConversationMemory()
        self.knowledge_base = KnowledgeBase()
        self.error_history = ErrorHistory()
        self.graph_exporter = WorkflowGraphExporter()

    def build_full_context(self,
                           workflow: Dict,
                           goal: PromptGoal = PromptGoal.CUSTOM,
                           custom_request: str = "",
                           include_nodes: bool = True,
                           include_history: bool = True,
                           include_knowledge: bool = True,
                           include_errors: bool = True,
                           include_graph: bool = True,
                           metrics: Optional[Dict] = None) -> str:
        """Build comprehensive context for LLM."""

        sections = []

        # System specs (always include)
        specs = self.specs_collector.collect()
        sections.append(specs.to_llm_context())

        # Workflow graph
        if include_graph:
            sections.append(self.graph_exporter.export_summary(workflow))

        # Available nodes
        if include_nodes:
            self.node_catalog.fetch_catalog()
            sections.append(self.node_catalog.export_workflow_relevant_nodes(workflow))

        # Conversation history
        if include_history:
            history = self.memory.get_context_for_llm(workflow)
            if history:
                sections.append(history)

        # Knowledge base
        if include_knowledge:
            knowledge = self.knowledge_base.export_for_llm(workflow)
            if knowledge:
                sections.append(knowledge)

        # Error history
        if include_errors:
            errors = self.error_history.export_for_llm(workflow)
            if errors:
                sections.append(errors)

        # Generate goal-specific prompt
        prompt = self.prompt_generator.generate(
            goal=goal,
            workflow=workflow,
            metrics=metrics,
            custom_request=custom_request
        )

        return "\n\n".join(sections) + "\n\n" + prompt

    def validate_response(self,
                          llm_response: str,
                          original_workflow: Dict) -> ValidationResult:
        """Validate LLM response."""
        return self.validator.validate(llm_response, original_workflow)

    def record_interaction(self,
                          workflow: Dict,
                          goal: str,
                          request: str,
                          response: str,
                          applied: bool = False,
                          success: bool = False) -> int:
        """Record an LLM interaction to memory."""
        return self.memory.add_entry(
            workflow=workflow,
            goal=goal,
            user_message=request,
            llm_response=response,
            applied=applied,
            success=success
        )

    def record_error(self,
                     error_type: str,
                     message: str,
                     workflow: Optional[Dict] = None,
                     node_id: Optional[int] = None,
                     node_type: Optional[str] = None) -> int:
        """Record an error to history."""
        return self.error_history.add_error(
            error_type=error_type,
            message=message,
            node_id=node_id,
            node_type=node_type,
            workflow=workflow
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI interface for LLM Enhancer."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Enhancer for Performance Lab")
    parser.add_argument('command', choices=['catalog', 'specs', 'validate', 'prompt', 'knowledge', 'history'])
    parser.add_argument('--workflow', '-w', help='Path to workflow JSON file')
    parser.add_argument('--goal', '-g', choices=['debug', 'quality', 'speed', 'vram', 'explain', 'custom'],
                       default='custom', help='Optimization goal')
    parser.add_argument('--request', '-r', help='Custom request text')
    parser.add_argument('--mod', '-m', help='Path to mod JSON to validate')
    parser.add_argument('--search', '-s', help='Search query')

    args = parser.parse_args()

    if args.command == 'catalog':
        catalog = NodeCatalog()
        if catalog.fetch_catalog():
            print(catalog.export_for_llm(compact=True, max_nodes=100))
        else:
            print("Failed to fetch catalog. Is ComfyUI running?")

    elif args.command == 'specs':
        specs = SystemSpecsCollector.collect()
        print(specs.to_llm_context())

    elif args.command == 'validate':
        if not args.mod:
            print("Error: --mod required for validate command")
            return
        with open(args.mod) as f:
            mod_json = f.read()

        workflow = None
        if args.workflow:
            with open(args.workflow) as f:
                workflow = json.load(f)

        validator = ModValidator()
        result = validator.validate(mod_json, workflow)
        print(result.to_string())

    elif args.command == 'prompt':
        if not args.workflow:
            print("Error: --workflow required for prompt command")
            return

        with open(args.workflow) as f:
            workflow = json.load(f)

        goal = PromptGoal(args.goal)
        generator = PromptGenerator()
        prompt = generator.generate(goal=goal, workflow=workflow, custom_request=args.request or "")
        print(prompt)

    elif args.command == 'knowledge':
        kb = KnowledgeBase()
        if args.search:
            results = kb.search(args.search)
            for entry in results:
                print(entry.to_llm_context())
                print("-" * 40)
        else:
            print(kb.export_for_llm())

    elif args.command == 'history':
        memory = ConversationMemory()
        stats = memory.get_stats()
        print(f"Conversation Memory Stats:")
        print(f"  Total conversations: {stats['total_conversations']}")
        print(f"  Applied: {stats['applied']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Unique workflows: {stats['unique_workflows']}")


if __name__ == "__main__":
    main()
