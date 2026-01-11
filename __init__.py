"""
Performance Lab v2.0 - ComfyUI Custom Nodes Package
===================================================
LLM-guided workflow optimization for ComfyUI.

Installation:
1. Copy or symlink this directory to ComfyUI/custom_nodes/performance_lab/
2. Install dependencies: pip install litellm
3. Set API keys in environment:
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY (optional)
4. Restart ComfyUI
"""

from .comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
