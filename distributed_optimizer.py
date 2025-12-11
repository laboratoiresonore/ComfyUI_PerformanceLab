"""
Distributed Workflow Optimizer for Performance Lab.

Analyzes multi-machine AI pipelines to identify bottlenecks,
measure latencies, and provide LLM-driven optimization recommendations.

This module:
1. Detects network service nodes in workflows
2. Profiles machine specifications
3. Measures latencies across the network
4. Identifies bottlenecks and parallelization opportunities
5. Generates comprehensive prompts for LLM optimization advice
"""

import json
import logging
import time
import socket
import requests
import requests.adapters
import concurrent.futures
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set
from urllib.parse import urlparse
from enum import Enum
from pathlib import Path

# Setup module logger
logger = logging.getLogger("performance_lab.distributed")


# Configurable timeouts per service type
SERVICE_TIMEOUTS = {
    "llm": 120,
    "image_generation": 300,
    "video_generation": 600,
    "speech_to_text": 60,
    "text_to_speech": 60,
    "embeddings": 30,
    "health_check": 10,
    "default": 30,
}


def validate_endpoint(endpoint: str) -> Tuple[bool, str]:
    """
    Validate an endpoint URL.

    Args:
        endpoint: URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not endpoint:
        return False, "Endpoint cannot be empty"

    try:
        parsed = urlparse(endpoint)

        if parsed.scheme not in ('http', 'https'):
            return False, f"Invalid scheme '{parsed.scheme}'. Must be http or https"

        if not parsed.netloc:
            return False, "Missing host in endpoint URL"

        # Check for valid port if specified
        if parsed.port is not None and not (1 <= parsed.port <= 65535):
            return False, f"Invalid port number: {parsed.port}"

        return True, ""
    except Exception as e:
        return False, f"Invalid URL format: {e}"


def get_timeout_for_service(service_type: str) -> int:
    """Get appropriate timeout for a service type."""
    # Map service categories to timeout keys
    category_map = {
        "llm": "llm",
        "image_generation": "image_generation",
        "video_generation": "video_generation",
        "speech_to_text": "speech_to_text",
        "text_to_speech": "text_to_speech",
        "embeddings": "embeddings",
        "stt": "speech_to_text",
        "tts": "text_to_speech",
    }

    timeout_key = category_map.get(service_type, "default")
    return SERVICE_TIMEOUTS.get(timeout_key, SERVICE_TIMEOUTS["default"])


class ServiceCategory(Enum):
    """Categories of network services."""
    IMAGE_GEN = "image_generation"
    VIDEO_GEN = "video_generation"
    LLM = "llm"
    STT = "speech_to_text"
    TTS = "text_to_speech"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    AUDIO = "audio"
    UPSCALE = "upscale"
    TRAINING = "training"
    OTHER = "other"


@dataclass
class MachineProfile:
    """Profile of a machine on the network."""
    name: str
    endpoint: str
    specs: str = ""
    gpu_model: str = ""
    gpu_vram_gb: float = 0.0
    cpu_model: str = ""
    ram_gb: float = 0.0
    description: str = ""
    docs_url: str = ""
    services: List[str] = field(default_factory=list)
    models_loaded: List[str] = field(default_factory=list)
    context_size: int = 0
    quantization: str = ""
    notes: str = ""


@dataclass
class NetworkNodeInfo:
    """Information about a network service node in a workflow."""
    node_id: str
    node_type: str
    service_type: str
    endpoint: str
    category: ServiceCategory
    machine: Optional[MachineProfile] = None
    latency_ms: float = -1.0
    is_healthy: bool = False


@dataclass
class DistributedAnalysis:
    """Complete analysis of a distributed workflow."""
    network_nodes: List[NetworkNodeInfo]
    machines: Dict[str, MachineProfile]
    dependencies: Dict[str, List[str]]
    parallel_groups: List[List[str]]
    bottlenecks: List[str]
    total_latency_ms: float
    estimated_execution_time: float
    suggestions: List[str]


# Network node types and their categories
NETWORK_NODE_TYPES = {
    # LLM nodes
    "KoboldLLM": ServiceCategory.LLM,
    "KoboldLLMAdvanced": ServiceCategory.LLM,

    # ComfyUI nodes
    "RemoteComfyUI": ServiceCategory.IMAGE_GEN,
    "RemoteComfyUISimple": ServiceCategory.IMAGE_GEN,

    # Local Generator (multi-service)
    "LocalGenerator": ServiceCategory.OTHER,
    "LocalGeneratorBatch": ServiceCategory.OTHER,

    # Health check (meta)
    "EndpointHealthCheck": ServiceCategory.OTHER,
    "MultiEndpointHealthCheck": ServiceCategory.OTHER,
}


class DistributedWorkflowAnalyzer:
    """Analyze workflows that use network service nodes."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.machines: Dict[str, MachineProfile] = {}
        # Use session for connection pooling
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=requests.adapters.Retry(total=2, backoff_factor=0.1)
        )
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

    def detect_network_nodes(self, workflow: Dict) -> List[NetworkNodeInfo]:
        """Find all network service nodes in a workflow."""
        network_nodes = []

        nodes = workflow.get("nodes", [])

        # Handle both list and dict formats
        if isinstance(nodes, list):
            node_iter = enumerate(nodes)
        else:
            node_iter = nodes.items()

        for node_id, node in node_iter:
            if isinstance(node, dict):
                node_type = node.get("type") or node.get("class_type", "")

                if node_type in NETWORK_NODE_TYPES:
                    endpoint = self._extract_endpoint(node)
                    category = NETWORK_NODE_TYPES[node_type]

                    # Check for service_type in LocalGenerator
                    if node_type in ["LocalGenerator", "LocalGeneratorBatch"]:
                        service_type = self._extract_widget_value(node, "service_type", "custom")
                        category = self._get_category_for_service(service_type)
                    else:
                        service_type = node_type

                    network_nodes.append(NetworkNodeInfo(
                        node_id=str(node_id) if isinstance(node_id, int) else node_id,
                        node_type=node_type,
                        service_type=service_type,
                        endpoint=endpoint,
                        category=category
                    ))

        return network_nodes

    def _extract_endpoint(self, node: Dict) -> str:
        """Extract endpoint URL from node widgets."""
        # Check widgets_values array
        widgets = node.get("widgets_values", [])
        if isinstance(widgets, list):
            for val in widgets:
                if isinstance(val, str) and val.startswith("http"):
                    return val

        # Check inputs dict
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict):
            endpoint = inputs.get("endpoint", "")
            if endpoint:
                return endpoint

        return "http://localhost:8188"

    def _extract_widget_value(self, node: Dict, key: str, default: str = "") -> str:
        """Extract a specific widget value from a node."""
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict):
            return inputs.get(key, default)
        return default

    def _get_category_for_service(self, service_type: str) -> ServiceCategory:
        """Map service type to category."""
        category_map = {
            # Image
            "comfyui": ServiceCategory.IMAGE_GEN,
            "automatic1111": ServiceCategory.IMAGE_GEN,
            "forge": ServiceCategory.IMAGE_GEN,
            "invokeai": ServiceCategory.IMAGE_GEN,
            "fooocus": ServiceCategory.IMAGE_GEN,
            "swarmui": ServiceCategory.IMAGE_GEN,
            "stable_horde": ServiceCategory.IMAGE_GEN,

            # LLM
            "koboldcpp": ServiceCategory.LLM,
            "koboldai": ServiceCategory.LLM,
            "ollama": ServiceCategory.LLM,
            "llamacpp_server": ServiceCategory.LLM,
            "text_gen_webui": ServiceCategory.LLM,
            "vllm": ServiceCategory.LLM,
            "lmdeploy": ServiceCategory.LLM,
            "tgi": ServiceCategory.LLM,
            "localai": ServiceCategory.LLM,
            "jan": ServiceCategory.LLM,
            "lmstudio": ServiceCategory.LLM,
            "gpt4all": ServiceCategory.LLM,
            "exllama": ServiceCategory.LLM,
            "tabbyapi": ServiceCategory.LLM,
            "aphrodite": ServiceCategory.LLM,
            "openai_compatible": ServiceCategory.LLM,

            # STT
            "whisper": ServiceCategory.STT,
            "faster_whisper": ServiceCategory.STT,
            "whisper_cpp": ServiceCategory.STT,
            "whisperx": ServiceCategory.STT,
            "nemo_asr": ServiceCategory.STT,
            "vosk": ServiceCategory.STT,

            # TTS
            "coqui_tts": ServiceCategory.TTS,
            "xtts": ServiceCategory.TTS,
            "xtts_v2": ServiceCategory.TTS,
            "alltalk": ServiceCategory.TTS,
            "silero_tts": ServiceCategory.TTS,
            "piper": ServiceCategory.TTS,
            "bark": ServiceCategory.TTS,
            "tortoise_tts": ServiceCategory.TTS,
            "fish_speech": ServiceCategory.TTS,

            # Embeddings
            "embeddings_tei": ServiceCategory.EMBEDDINGS,
            "sentence_transformers": ServiceCategory.EMBEDDINGS,
            "infinity_emb": ServiceCategory.EMBEDDINGS,
            "fastembed": ServiceCategory.EMBEDDINGS,

            # Vision
            "llava": ServiceCategory.VISION,
            "cogvlm": ServiceCategory.VISION,
            "qwen_vl": ServiceCategory.VISION,
            "moondream": ServiceCategory.VISION,
            "florence2": ServiceCategory.VISION,

            # Video
            "animatediff": ServiceCategory.VIDEO_GEN,
            "svd": ServiceCategory.VIDEO_GEN,
            "mochi": ServiceCategory.VIDEO_GEN,
            "cogvideo": ServiceCategory.VIDEO_GEN,
            "hunyuan_video": ServiceCategory.VIDEO_GEN,
            "ltx_video": ServiceCategory.VIDEO_GEN,
            "wan": ServiceCategory.VIDEO_GEN,

            # Audio
            "audiocraft": ServiceCategory.AUDIO,
            "musicgen": ServiceCategory.AUDIO,
            "audioldm": ServiceCategory.AUDIO,
            "stable_audio": ServiceCategory.AUDIO,

            # Upscale
            "realesrgan": ServiceCategory.UPSCALE,
            "swinir": ServiceCategory.UPSCALE,
            "gfpgan": ServiceCategory.UPSCALE,
            "codeformer": ServiceCategory.UPSCALE,

            # Training
            "kohya_ss": ServiceCategory.TRAINING,
        }
        return category_map.get(service_type, ServiceCategory.OTHER)

    def get_unique_endpoints(self, workflow: Dict) -> Dict[str, List[str]]:
        """Get all unique endpoints grouped by category."""
        nodes = self.detect_network_nodes(workflow)
        endpoints: Dict[str, List[str]] = {}

        for node in nodes:
            category = node.category.value
            if category not in endpoints:
                endpoints[category] = []
            if node.endpoint not in endpoints[category]:
                endpoints[category].append(node.endpoint)

        return endpoints

    def check_endpoint_health(self, endpoint: str, service_type: str = "comfyui") -> Tuple[bool, float]:
        """Check if an endpoint is healthy and measure latency."""
        # Validate endpoint first
        is_valid, error_msg = validate_endpoint(endpoint)
        if not is_valid:
            logger.warning(f"Invalid endpoint '{endpoint}': {error_msg}")
            return (False, -1.0)

        endpoint = endpoint.rstrip('/')

        # Health check paths per service type
        health_paths = {
            "comfyui": "/system_stats",
            "automatic1111": "/sdapi/v1/memory",
            "koboldcpp": "/api/v1/model",
            "ollama": "/api/tags",
            "vllm": "/health",
        }

        path = health_paths.get(service_type, "/")
        url = f"{endpoint}{path}"
        timeout = get_timeout_for_service("health_check")

        try:
            start = time.time()
            response = self._session.get(url, timeout=timeout)
            latency = (time.time() - start) * 1000
            return (response.status_code == 200, latency)
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout checking endpoint: {endpoint}")
            return (False, -1.0)
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"Connection error for {endpoint}: {e}")
            return (False, -1.0)
        except Exception as e:
            logger.warning(f"Unexpected error checking endpoint {endpoint}: {e}")
            return (False, -1.0)

    def check_all_endpoints(self, workflow: Dict) -> Dict[str, Dict[str, Any]]:
        """Health check all endpoints in the workflow."""
        nodes = self.detect_network_nodes(workflow)
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for node in nodes:
                if node.endpoint not in futures:
                    futures[node.endpoint] = executor.submit(
                        self.check_endpoint_health,
                        node.endpoint,
                        node.service_type
                    )

            for endpoint, future in futures.items():
                try:
                    is_healthy, latency = future.result()
                    results[endpoint] = {
                        "healthy": is_healthy,
                        "latency_ms": round(latency, 2) if latency > 0 else None
                    }
                except Exception as e:
                    results[endpoint] = {
                        "healthy": False,
                        "error": str(e)
                    }

        return results

    def build_dependency_graph(self, workflow: Dict) -> Dict[str, List[str]]:
        """Build a dependency graph of nodes."""
        dependencies: Dict[str, List[str]] = {}

        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])

        # Initialize all nodes
        if isinstance(nodes, list):
            for i, node in enumerate(nodes):
                dependencies[str(i)] = []
        else:
            for node_id in nodes:
                dependencies[str(node_id)] = []

        # Process links
        if isinstance(links, list):
            for link in links:
                if isinstance(link, list) and len(link) >= 4:
                    source_node = str(link[1])
                    target_node = str(link[3])
                    if target_node in dependencies:
                        dependencies[target_node].append(source_node)

        return dependencies

    def find_parallel_opportunities(self, workflow: Dict) -> List[List[str]]:
        """Find nodes that can run in parallel."""
        deps = self.build_dependency_graph(workflow)
        network_nodes = self.detect_network_nodes(workflow)
        network_ids = {n.node_id for n in network_nodes}

        # Find network nodes with same dependencies (can run in parallel)
        dep_groups: Dict[tuple, List[str]] = {}

        for node in network_nodes:
            node_deps = tuple(sorted(deps.get(node.node_id, [])))
            if node_deps not in dep_groups:
                dep_groups[node_deps] = []
            dep_groups[node_deps].append(node.node_id)

        # Return groups with 2+ nodes
        return [group for group in dep_groups.values() if len(group) > 1]


class DistributedWorkflowOptimizer:
    """Optimize distributed workflows with LLM assistance."""

    def __init__(self):
        self.analyzer = DistributedWorkflowAnalyzer()
        self.machines: Dict[str, MachineProfile] = {}

    def register_machine(self, profile: MachineProfile):
        """Register a machine profile."""
        self.machines[profile.endpoint] = profile

    def analyze(self, workflow: Dict) -> DistributedAnalysis:
        """Full analysis of a distributed workflow."""
        nodes = self.analyzer.detect_network_nodes(workflow)
        endpoints = self.analyzer.get_unique_endpoints(workflow)
        health = self.analyzer.check_all_endpoints(workflow)
        deps = self.analyzer.build_dependency_graph(workflow)
        parallel = self.analyzer.find_parallel_opportunities(workflow)

        # Update nodes with health and machine info
        total_latency = 0.0
        for node in nodes:
            if node.endpoint in health:
                node.is_healthy = health[node.endpoint].get("healthy", False)
                node.latency_ms = health[node.endpoint].get("latency_ms", -1.0)
                if node.latency_ms > 0:
                    total_latency += node.latency_ms
            if node.endpoint in self.machines:
                node.machine = self.machines[node.endpoint]

        # Identify bottlenecks (highest latency nodes)
        sorted_nodes = sorted(nodes, key=lambda n: n.latency_ms, reverse=True)
        bottlenecks = [n.node_id for n in sorted_nodes[:3] if n.latency_ms > 0]

        # Generate suggestions
        suggestions = self._generate_suggestions(nodes, parallel, bottlenecks)

        return DistributedAnalysis(
            network_nodes=nodes,
            machines=self.machines,
            dependencies=deps,
            parallel_groups=parallel,
            bottlenecks=bottlenecks,
            total_latency_ms=total_latency,
            estimated_execution_time=total_latency / 1000,  # Very rough estimate
            suggestions=suggestions
        )

    def _generate_suggestions(
        self,
        nodes: List[NetworkNodeInfo],
        parallel: List[List[str]],
        bottlenecks: List[str]
    ) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []

        # Parallelization
        if parallel:
            for group in parallel:
                if len(group) > 1:
                    suggestions.append(
                        f"Nodes {', '.join(group)} can run in parallel - consider restructuring"
                    )

        # Bottlenecks
        for node_id in bottlenecks:
            node = next((n for n in nodes if n.node_id == node_id), None)
            if node and node.latency_ms > 100:
                suggestions.append(
                    f"Node {node_id} ({node.service_type}) has high latency ({node.latency_ms:.0f}ms)"
                )

        # Unhealthy endpoints
        for node in nodes:
            if not node.is_healthy:
                suggestions.append(
                    f"Node {node.node_id} endpoint {node.endpoint} is not responding"
                )

        # Single points of failure
        endpoint_count: Dict[str, int] = {}
        for node in nodes:
            endpoint_count[node.endpoint] = endpoint_count.get(node.endpoint, 0) + 1

        for endpoint, count in endpoint_count.items():
            if count > 2:
                suggestions.append(
                    f"Multiple nodes ({count}) use {endpoint} - consider load balancing"
                )

        return suggestions

    def generate_llm_prompt(
        self,
        workflow: Dict,
        analysis: DistributedAnalysis,
        goal: str = "optimization"
    ) -> str:
        """Generate a comprehensive prompt for LLM optimization advice."""

        # Build machine specs section
        machine_specs = []
        for endpoint, machine in analysis.machines.items():
            spec = f"### {machine.name or endpoint}\n"
            spec += f"- Endpoint: {endpoint}\n"
            if machine.specs:
                spec += f"- Specs: {machine.specs}\n"
            if machine.gpu_model:
                spec += f"- GPU: {machine.gpu_model}"
                if machine.gpu_vram_gb > 0:
                    spec += f" ({machine.gpu_vram_gb}GB VRAM)"
                spec += "\n"
            if machine.cpu_model:
                spec += f"- CPU: {machine.cpu_model}\n"
            if machine.ram_gb > 0:
                spec += f"- RAM: {machine.ram_gb}GB\n"
            if machine.models_loaded:
                spec += f"- Models: {', '.join(machine.models_loaded)}\n"
            if machine.context_size > 0:
                spec += f"- Context: {machine.context_size}\n"
            if machine.quantization:
                spec += f"- Quantization: {machine.quantization}\n"
            if machine.docs_url:
                spec += f"- Docs: {machine.docs_url}\n"
            if machine.description:
                spec += f"- Description: {machine.description}\n"
            machine_specs.append(spec)

        # Build network nodes section
        nodes_section = []
        for node in analysis.network_nodes:
            status = "healthy" if node.is_healthy else "UNHEALTHY"
            latency = f"{node.latency_ms:.0f}ms" if node.latency_ms > 0 else "unknown"
            nodes_section.append(
                f"- [{node.node_id}] {node.node_type} -> {node.endpoint} "
                f"({status}, latency: {latency})"
            )

        # Build bottlenecks section
        bottleneck_section = []
        for node_id in analysis.bottlenecks:
            node = next((n for n in analysis.network_nodes if n.node_id == node_id), None)
            if node:
                bottleneck_section.append(
                    f"- Node {node_id}: {node.service_type} at {node.endpoint} "
                    f"({node.latency_ms:.0f}ms latency)"
                )

        prompt = f"""# Distributed AI Pipeline Optimization Request

## Goal
{goal}

## Network Architecture

### Machines on Network
{chr(10).join(machine_specs) if machine_specs else "No machine profiles registered. User should add machine specs."}

### Network Service Nodes in Workflow
{chr(10).join(nodes_section)}

### Dependencies
```json
{json.dumps(analysis.dependencies, indent=2)}
```

### Parallelization Opportunities
{json.dumps(analysis.parallel_groups, indent=2) if analysis.parallel_groups else "None detected"}

### Current Bottlenecks
{chr(10).join(bottleneck_section) if bottleneck_section else "None identified"}

### Current Suggestions
{chr(10).join('- ' + s for s in analysis.suggestions) if analysis.suggestions else "None"}

## Performance Metrics
- Total network latency: {analysis.total_latency_ms:.0f}ms
- Estimated execution time: {analysis.estimated_execution_time:.1f}s

## Request
Based on this distributed AI pipeline analysis, please provide:

1. **Bottleneck Analysis**: Which machine/service is slowing down the pipeline and why?

2. **Hardware Recommendations**: Based on the machine specs, what changes would help?
   - Should any service move to a different machine?
   - Are there VRAM/RAM constraints causing issues?
   - Would different quantization help?

3. **Configuration Recommendations**: Specific settings to adjust
   - Context sizes for LLMs
   - Batch sizes for image generation
   - Quality vs speed tradeoffs

4. **Architecture Recommendations**: Pipeline restructuring
   - Which operations can run in parallel?
   - Should any operations be combined or split?
   - Would caching help anywhere?

5. **Optimized Configuration**: Provide specific parameter recommendations

Please be specific and actionable. Reference the actual machine specs and endpoints provided.
"""
        return prompt


# Convenience functions for CLI usage
def analyze_workflow_file(workflow_path: str) -> DistributedAnalysis:
    """Analyze a workflow file."""
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    optimizer = DistributedWorkflowOptimizer()
    return optimizer.analyze(workflow)


def generate_optimization_prompt(
    workflow_path: str,
    machines: Optional[List[Dict]] = None
) -> str:
    """Generate an LLM prompt for a workflow file."""
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    optimizer = DistributedWorkflowOptimizer()

    # Register machines if provided
    if machines:
        for m in machines:
            profile = MachineProfile(
                name=m.get("name", ""),
                endpoint=m.get("endpoint", ""),
                specs=m.get("specs", ""),
                gpu_model=m.get("gpu_model", ""),
                gpu_vram_gb=m.get("gpu_vram_gb", 0.0),
                cpu_model=m.get("cpu_model", ""),
                ram_gb=m.get("ram_gb", 0.0),
                description=m.get("description", ""),
                docs_url=m.get("docs_url", ""),
                models_loaded=m.get("models_loaded", []),
                context_size=m.get("context_size", 0),
                quantization=m.get("quantization", ""),
                notes=m.get("notes", ""),
            )
            optimizer.register_machine(profile)

    analysis = optimizer.analyze(workflow)
    return optimizer.generate_llm_prompt(workflow, analysis)
