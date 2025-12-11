"""
Pytest configuration and shared fixtures for Performance Lab tests.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "custom_nodes" / "ComfyUI_NetworkServices" / "nodes"))


@pytest.fixture
def sample_sd15_workflow():
    """Sample SD 1.5 workflow for testing."""
    return {
        "nodes": [
            {
                "id": 1,
                "type": "CheckpointLoaderSimple",
                "widgets_values": ["v1-5-pruned-emaonly.safetensors"]
            },
            {
                "id": 2,
                "type": "KSampler",
                "widgets_values": [12345, "fixed", 20, 7.5, "euler", "normal", 1.0]
            },
            {
                "id": 3,
                "type": "CLIPTextEncode",
                "widgets_values": ["a beautiful landscape, highly detailed, 4k"]
            },
            {
                "id": 4,
                "type": "CLIPTextEncode",
                "widgets_values": ["blurry, low quality"]
            },
            {
                "id": 5,
                "type": "EmptyLatentImage",
                "widgets_values": [512, 512, 1]
            },
            {
                "id": 6,
                "type": "VAEDecode"
            },
            {
                "id": 7,
                "type": "SaveImage",
                "widgets_values": ["output/test"]
            }
        ],
        "links": [
            [1, 1, 0, 2, 0, "MODEL"],
            [2, 1, 1, 3, 0, "CLIP"],
            [3, 1, 1, 4, 0, "CLIP"],
            [4, 3, 0, 2, 1, "CONDITIONING"],
            [5, 4, 0, 2, 2, "CONDITIONING"],
            [6, 5, 0, 2, 3, "LATENT"],
            [7, 2, 0, 6, 0, "LATENT"],
            [8, 1, 2, 6, 1, "VAE"],
            [9, 6, 0, 7, 0, "IMAGE"]
        ]
    }


@pytest.fixture
def sample_distributed_workflow():
    """Sample distributed workflow with network nodes."""
    return {
        "nodes": [
            {
                "id": "1",
                "type": "KoboldLLM",
                "widgets_values": [
                    "http://192.168.1.100:5001",
                    "Write a prompt for a beautiful landscape",
                    512, 0.7, 1.0
                ]
            },
            {
                "id": "2",
                "type": "RemoteComfyUI",
                "widgets_values": [
                    "http://192.168.1.101:8188",
                    "{}",
                    300
                ]
            },
            {
                "id": "3",
                "type": "EndpointHealthCheck",
                "widgets_values": ["http://192.168.1.100:5001", "koboldcpp"]
            },
            {
                "id": "4",
                "type": "SaveImage",
                "widgets_values": ["distributed_output"]
            }
        ],
        "links": [
            [1, 1, 0, 2, 0, "STRING"],
            [2, 2, 0, 4, 0, "IMAGE"]
        ]
    }


@pytest.fixture
def sample_sdxl_workflow():
    """Sample SDXL workflow for testing."""
    return {
        "nodes": [
            {
                "id": 1,
                "type": "CheckpointLoaderSimple",
                "widgets_values": ["sd_xl_base_1.0.safetensors"]
            },
            {
                "id": 2,
                "type": "KSampler",
                "widgets_values": [12345, "fixed", 25, 7.0, "dpmpp_2m", "karras", 1.0]
            },
            {
                "id": 3,
                "type": "EmptyLatentImage",
                "widgets_values": [1024, 1024, 1]
            }
        ],
        "links": []
    }


@pytest.fixture
def mock_comfyui_response():
    """Mock response from ComfyUI API."""
    return {
        "prompt_id": "test-prompt-id-123",
        "number": 1,
        "node_errors": {}
    }


@pytest.fixture
def mock_kobold_response():
    """Mock response from Kobold API."""
    return {
        "results": [
            {"text": "A majestic mountain landscape with snow-capped peaks"}
        ]
    }


@pytest.fixture
def mock_system_stats():
    """Mock ComfyUI system stats response."""
    return {
        "system": {
            "os": "linux",
            "python_version": "3.10.12",
            "pytorch_version": "2.1.0+cu118"
        },
        "devices": [
            {
                "name": "NVIDIA GeForce RTX 4090",
                "type": "cuda",
                "vram_total": 25769803776,
                "vram_free": 20000000000
            }
        ]
    }
