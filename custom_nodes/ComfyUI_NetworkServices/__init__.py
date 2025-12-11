"""
ComfyUI Network Services - Custom nodes for distributed workflow orchestration.

This node pack provides nodes that connect to network services, enabling
ComfyUI to act as an orchestrator for multi-machine AI pipelines.

Part of the Performance Lab ecosystem.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web"

__version__ = "0.1.0"
