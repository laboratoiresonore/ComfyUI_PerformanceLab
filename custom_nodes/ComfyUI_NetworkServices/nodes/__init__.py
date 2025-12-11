"""
Network Service Nodes for ComfyUI.

Provides flexible nodes for connecting to network services:
- KoboldLLM: Comprehensive LLM generation via Kobold API
- RemoteComfyUI: Execute workflows on remote ComfyUI instances
- LocalGenerator: General-purpose REST API node for 70+ AI services
- EndpointHealthCheck: Health check and monitor network endpoints

Part of the Performance Lab ecosystem for multi-machine AI orchestration.
"""

from .kobold_llm import KoboldLLM, KoboldLLMAdvanced
from .remote_comfyui import RemoteComfyUI, RemoteComfyUISimple
from .network_api import NetworkAPI, NetworkAPIBatch, SERVICE_DESCRIPTIONS
from .health_check import EndpointHealthCheck, MultiEndpointHealthCheck

NODE_CLASS_MAPPINGS = {
    # LLM Nodes
    "KoboldLLM": KoboldLLM,
    "KoboldLLMAdvanced": KoboldLLMAdvanced,

    # Remote ComfyUI
    "RemoteComfyUI": RemoteComfyUI,
    "RemoteComfyUISimple": RemoteComfyUISimple,

    # Local Generator (70+ service types)
    "LocalGenerator": NetworkAPI,
    "LocalGeneratorBatch": NetworkAPIBatch,

    # Health Check
    "EndpointHealthCheck": EndpointHealthCheck,
    "MultiEndpointHealthCheck": MultiEndpointHealthCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # LLM Nodes
    "KoboldLLM": "Kobold LLM",
    "KoboldLLMAdvanced": "Kobold LLM (Advanced)",

    # Remote ComfyUI
    "RemoteComfyUI": "Remote ComfyUI",
    "RemoteComfyUISimple": "Remote ComfyUI (Quick)",

    # Local Generator
    "LocalGenerator": "Local Generator (70+ Services)",
    "LocalGeneratorBatch": "Local Generator Batch",

    # Health Check
    "EndpointHealthCheck": "Endpoint Health Check",
    "MultiEndpointHealthCheck": "Multi-Endpoint Health Check",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'SERVICE_DESCRIPTIONS'
]
