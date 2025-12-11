"""
Local Generator Node - Connect to any AI generator on your local network.

Universal node for ComfyUI, Automatic1111, Kohya, Ollama, TTS, STT, or any REST API.
Designed for multi-machine setups optimized by Performance Lab.
"""

import json
import time
import requests
import io
import base64
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
import numpy as np


class NetworkAPI:
    """
    Local Generator - Connect to any AI service on your network.

    Works with:
    - ComfyUI (image/video generation)
    - Automatic1111/Forge (SD WebUI API)
    - Kohya (training servers)
    - Ollama (LLM)
    - Whisper/faster-whisper (STT)
    - Coqui/XTTS/AllTalk (TTS)
    - Embeddings servers
    - Any custom REST API

    The LLM optimizer uses the service description and docs URL
    to understand your setup and provide tailored recommendations.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "endpoint": ("STRING", {
                    "default": "http://localhost:8188",
                    "placeholder": "http://192.168.1.100:8188"
                }),
                "service_type": ([
                    # === Image/Video Generation ===
                    "comfyui",
                    "automatic1111",
                    "forge",
                    "invokeai",
                    "fooocus",
                    "swarmui",
                    "kohya_ss",
                    "stable_horde",

                    # === LLM Inference ===
                    "koboldcpp",
                    "koboldai",
                    "ollama",
                    "llamacpp_server",
                    "text_gen_webui",
                    "vllm",
                    "lmdeploy",
                    "tgi",  # Text Generation Inference
                    "localai",
                    "jan",
                    "lmstudio",
                    "gpt4all",
                    "exllama",
                    "tabbyapi",
                    "aphrodite",

                    # === OpenAI Compatible ===
                    "openai_compatible",
                    "litellm",
                    "openrouter",

                    # === Speech-to-Text ===
                    "whisper",
                    "faster_whisper",
                    "whisper_cpp",
                    "whisperx",
                    "insanely_fast_whisper",
                    "whisper_jax",
                    "nemo_asr",
                    "vosk",
                    "deepspeech",

                    # === Text-to-Speech ===
                    "coqui_tts",
                    "xtts",
                    "xtts_v2",
                    "alltalk",
                    "silero_tts",
                    "piper",
                    "bark",
                    "tortoise_tts",
                    "valle",
                    "styletts2",
                    "openvoice",
                    "rvc",
                    "so_vits_svc",
                    "fish_speech",

                    # === Embeddings ===
                    "embeddings_tei",  # Text Embeddings Inference
                    "sentence_transformers",
                    "infinity_emb",
                    "fastembed",

                    # === Vision/Multimodal ===
                    "llava",
                    "cogvlm",
                    "qwen_vl",
                    "moondream",
                    "bakllava",

                    # === Video Generation ===
                    "animatediff",
                    "svd",  # Stable Video Diffusion
                    "mochi",
                    "cogvideo",
                    "hunyuan_video",
                    "ltx_video",
                    "wan",  # Alibaba WAN

                    # === Audio Generation ===
                    "audiocraft",
                    "musicgen",
                    "audioldm",
                    "riffusion",
                    "stable_audio",

                    # === Upscaling/Enhancement ===
                    "realesrgan",
                    "swinir",
                    "gfpgan",
                    "codeformer",

                    # === Other AI Services ===
                    "controlnet",
                    "ip_adapter",
                    "segment_anything",
                    "grounding_dino",
                    "depth_anything",
                    "marigold",
                    "florence2",

                    # === Custom ===
                    "custom"
                ], {"default": "comfyui"}),
            },
            "optional": {
                # Request configuration
                "api_path": ("STRING", {
                    "default": "",
                    "placeholder": "/api/v1/generate (auto-detected if empty)"
                }),
                "method": (["POST", "GET", "PUT", "DELETE"], {"default": "POST"}),
                "payload": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": '{"prompt": "...", "steps": 20}'
                }),
                "headers": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": '{"Authorization": "Bearer xxx"}'
                }),
                "timeout": ("INT", {"default": 300, "min": 10, "max": 3600}),

                # Response handling
                "response_type": (["auto", "json", "image", "audio", "text", "binary"], {"default": "auto"}),
                "extract_path": ("STRING", {
                    "default": "",
                    "placeholder": "results.0.text or images.0"
                }),

                # Machine/Service Description for LLM Optimizer
                "machine_name": ("STRING", {
                    "default": "",
                    "placeholder": "GPU Server 1"
                }),
                "machine_specs": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "RTX 4090 24GB VRAM\n64GB RAM\nRyzen 9 5950X\nNVMe SSD"
                }),
                "service_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Primary SD generation server running SDXL with 4-bit quantization"
                }),
                "docs_url": ("STRING", {
                    "default": "",
                    "placeholder": "https://github.com/comfyanonymous/ComfyUI/wiki"
                }),
                "model_loaded": ("STRING", {
                    "default": "",
                    "placeholder": "SDXL Base + RealVisXL refiner"
                }),
                "context_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "placeholder": "For LLMs: context window size"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "placeholder": "Current batch size setting"
                }),
                "quantization": ("STRING", {
                    "default": "",
                    "placeholder": "fp16, int8, int4, q4_k_m, etc."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("response_text", "response_image", "execution_time", "service_info")
    FUNCTION = "call_api"
    CATEGORY = "NetworkServices/LocalGenerator"
    OUTPUT_NODE = True

    NETWORK_SERVICE = True
    SERVICE_TYPE = "local_generator"
    ENDPOINT_PARAM = "endpoint"

    # Default API paths per service type
    DEFAULT_PATHS = {
        # Image/Video Generation
        "comfyui": "/prompt",
        "automatic1111": "/sdapi/v1/txt2img",
        "forge": "/sdapi/v1/txt2img",
        "invokeai": "/api/v1/images",
        "fooocus": "/v1/generation/text-to-image",
        "swarmui": "/API/GenerateText2Image",
        "kohya_ss": "/train",
        "stable_horde": "/api/v2/generate/async",

        # LLM Inference
        "koboldcpp": "/api/v1/generate",
        "koboldai": "/api/v1/generate",
        "ollama": "/api/generate",
        "llamacpp_server": "/completion",
        "text_gen_webui": "/api/v1/generate",
        "vllm": "/v1/completions",
        "lmdeploy": "/v1/chat/completions",
        "tgi": "/generate",
        "localai": "/v1/completions",
        "jan": "/v1/chat/completions",
        "lmstudio": "/v1/chat/completions",
        "gpt4all": "/v1/completions",
        "exllama": "/api/v1/generate",
        "tabbyapi": "/v1/completions",
        "aphrodite": "/v1/completions",

        # OpenAI Compatible
        "openai_compatible": "/v1/chat/completions",
        "litellm": "/v1/chat/completions",
        "openrouter": "/api/v1/chat/completions",

        # Speech-to-Text
        "whisper": "/asr",
        "faster_whisper": "/transcribe",
        "whisper_cpp": "/inference",
        "whisperx": "/asr",
        "insanely_fast_whisper": "/transcribe",
        "whisper_jax": "/transcribe",
        "nemo_asr": "/asr",
        "vosk": "/asr",
        "deepspeech": "/stt",

        # Text-to-Speech
        "coqui_tts": "/api/tts",
        "xtts": "/tts_to_audio",
        "xtts_v2": "/tts_to_audio",
        "alltalk": "/api/tts-generate",
        "silero_tts": "/tts",
        "piper": "/api/tts",
        "bark": "/generate",
        "tortoise_tts": "/generate",
        "valle": "/synthesize",
        "styletts2": "/synthesize",
        "openvoice": "/synthesize",
        "rvc": "/convert",
        "so_vits_svc": "/convert",
        "fish_speech": "/v1/tts",

        # Embeddings
        "embeddings_tei": "/embed",
        "sentence_transformers": "/embeddings",
        "infinity_emb": "/embeddings",
        "fastembed": "/embed",

        # Vision/Multimodal
        "llava": "/api/generate",
        "cogvlm": "/v1/chat/completions",
        "qwen_vl": "/v1/chat/completions",
        "moondream": "/answer",
        "bakllava": "/api/generate",

        # Video Generation
        "animatediff": "/generate",
        "svd": "/generate",
        "mochi": "/generate",
        "cogvideo": "/generate",
        "hunyuan_video": "/generate",
        "ltx_video": "/generate",
        "wan": "/generate",

        # Audio Generation
        "audiocraft": "/generate",
        "musicgen": "/generate",
        "audioldm": "/generate",
        "riffusion": "/generate",
        "stable_audio": "/generate",

        # Upscaling/Enhancement
        "realesrgan": "/upscale",
        "swinir": "/upscale",
        "gfpgan": "/restore",
        "codeformer": "/restore",

        # Other AI Services
        "controlnet": "/detect",
        "ip_adapter": "/generate",
        "segment_anything": "/segment",
        "grounding_dino": "/detect",
        "depth_anything": "/predict",
        "marigold": "/predict",
        "florence2": "/process",

        "custom": "",
    }

    def call_api(
        self,
        endpoint: str,
        service_type: str = "comfyui",
        api_path: str = "",
        method: str = "POST",
        payload: str = "",
        headers: str = "",
        timeout: int = 300,
        response_type: str = "auto",
        extract_path: str = "",
        machine_name: str = "",
        machine_specs: str = "",
        service_description: str = "",
        docs_url: str = "",
        model_loaded: str = "",
        context_size: int = 0,
        batch_size: int = 1,
        quantization: str = ""
    ) -> Tuple[str, Any, float, str]:
        """Call any local network API."""

        endpoint = endpoint.rstrip('/')

        # Determine API path
        if not api_path:
            api_path = self.DEFAULT_PATHS.get(service_type, "")

        url = f"{endpoint}{api_path}"

        # Parse payload
        request_data = None
        if payload.strip():
            try:
                request_data = json.loads(payload)
            except json.JSONDecodeError:
                # Treat as raw string payload
                request_data = payload

        # Parse headers
        request_headers = {"Content-Type": "application/json"}
        if headers.strip():
            try:
                request_headers.update(json.loads(headers))
            except json.JSONDecodeError:
                pass

        # Make request
        start_time = time.time()

        try:
            if method == "GET":
                response = requests.get(url, headers=request_headers, timeout=timeout)
            elif method == "POST":
                if isinstance(request_data, dict):
                    response = requests.post(url, json=request_data, headers=request_headers, timeout=timeout)
                else:
                    response = requests.post(url, data=request_data, headers=request_headers, timeout=timeout)
            elif method == "PUT":
                response = requests.put(url, json=request_data, headers=request_headers, timeout=timeout)
            elif method == "DELETE":
                response = requests.delete(url, headers=request_headers, timeout=timeout)

            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout after {timeout}s connecting to {url}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to {url}")
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = response.text[:500]
            except:
                pass
            raise RuntimeError(f"API error: {e}\n{error_detail}")

        execution_time = time.time() - start_time

        # Determine response type
        if response_type == "auto":
            content_type = response.headers.get("Content-Type", "")
            if "image" in content_type:
                response_type = "image"
            elif "audio" in content_type:
                response_type = "audio"
            elif "json" in content_type:
                response_type = "json"
            else:
                response_type = "text"

        # Parse response
        response_text = ""
        response_image = None

        if response_type == "json":
            result = response.json()
            response_text = json.dumps(result, indent=2)

            # Extract specific path if requested
            if extract_path:
                extracted = self._extract_json_path(result, extract_path)
                if isinstance(extracted, str):
                    response_text = extracted
                else:
                    response_text = json.dumps(extracted, indent=2)

            # Check for base64 images in response (A1111 style)
            if "images" in result and isinstance(result["images"], list):
                response_image = self._decode_base64_images(result["images"])

        elif response_type == "image":
            response_image = self._bytes_to_image(response.content)
            response_text = f"Image received: {len(response.content)} bytes"

        elif response_type == "text":
            response_text = response.text

        elif response_type == "binary":
            response_text = f"Binary data: {len(response.content)} bytes"

        # Build comprehensive service info for Performance Lab
        service_info = json.dumps({
            "endpoint": endpoint,
            "service_type": service_type,
            "api_path": api_path,
            "execution_time": execution_time,
            "machine_name": machine_name,
            "machine_specs": machine_specs,
            "service_description": service_description,
            "docs_url": docs_url,
            "model_loaded": model_loaded,
            "context_size": context_size if context_size > 0 else None,
            "batch_size": batch_size,
            "quantization": quantization if quantization else None,
            "response_size_bytes": len(response.content),
        }, indent=2)

        return (response_text, response_image, execution_time, service_info)

    def _extract_json_path(self, data: Any, path: str) -> Any:
        """Extract value from nested JSON using dot notation."""
        parts = path.split(".")
        result = data
        for part in parts:
            try:
                if isinstance(result, dict):
                    result = result[part]
                elif isinstance(result, list):
                    result = result[int(part)]
            except (KeyError, IndexError, ValueError):
                return None
        return result

    def _decode_base64_images(self, images: List[str]) -> Any:
        """Decode base64 images to numpy array."""
        decoded = []
        for img_data in images:
            try:
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img).astype(np.float32) / 255.0
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
                decoded.append(img_array)
            except:
                pass
        if decoded:
            return np.stack(decoded)
        return None

    def _bytes_to_image(self, data: bytes) -> Any:
        """Convert raw bytes to image tensor."""
        try:
            img = Image.open(io.BytesIO(data))
            img_array = np.array(img).astype(np.float32) / 255.0
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            return np.stack([img_array])
        except:
            return None


class NetworkAPIBatch:
    """
    Batch Local Generator - Call multiple services in parallel or sequence.

    Useful for:
    - Calling the same endpoint with multiple inputs
    - Chaining different services
    - Load balancing across multiple machines
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "endpoints": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "http://192.168.1.100:8188\nhttp://192.168.1.101:8188"
                }),
                "service_type": ([
                    "comfyui", "automatic1111", "forge", "ollama",
                    "kobold", "whisper", "coqui_tts", "custom"
                ], {"default": "comfyui"}),
            },
            "optional": {
                "payloads": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "One JSON payload per line, or single payload for all"
                }),
                "mode": (["parallel", "sequential", "round_robin"], {"default": "parallel"}),
                "timeout": ("INT", {"default": 300, "min": 10, "max": 3600}),

                # Batch service info
                "cluster_name": ("STRING", {
                    "default": "",
                    "placeholder": "GPU Cluster Alpha"
                }),
                "cluster_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "3x RTX 4090 servers for SD generation"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("responses", "total_time", "cluster_info")
    FUNCTION = "batch_call"
    CATEGORY = "NetworkServices/LocalGenerator"

    NETWORK_SERVICE = True
    SERVICE_TYPE = "local_generator_batch"

    def batch_call(
        self,
        endpoints: str,
        service_type: str = "comfyui",
        payloads: str = "",
        mode: str = "parallel",
        timeout: int = 300,
        cluster_name: str = "",
        cluster_description: str = ""
    ) -> Tuple[str, float, str]:
        """Call multiple endpoints."""

        import concurrent.futures

        endpoint_list = [e.strip() for e in endpoints.strip().split("\n") if e.strip()]
        payload_list = [p.strip() for p in payloads.strip().split("\n") if p.strip()] if payloads.strip() else [""]

        # Expand payloads to match endpoints if single payload
        if len(payload_list) == 1 and len(endpoint_list) > 1:
            payload_list = payload_list * len(endpoint_list)

        start_time = time.time()
        results = []

        if mode == "parallel":
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoint_list)) as executor:
                futures = []
                for ep, pl in zip(endpoint_list, payload_list):
                    node = NetworkAPI()
                    futures.append(executor.submit(
                        node.call_api,
                        endpoint=ep,
                        service_type=service_type,
                        payload=pl,
                        timeout=timeout
                    ))
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append({"success": True, "response": result[0]})
                    except Exception as e:
                        results.append({"success": False, "error": str(e)})

        elif mode == "sequential":
            for ep, pl in zip(endpoint_list, payload_list):
                try:
                    node = NetworkAPI()
                    result = node.call_api(
                        endpoint=ep,
                        service_type=service_type,
                        payload=pl,
                        timeout=timeout
                    )
                    results.append({"success": True, "response": result[0]})
                except Exception as e:
                    results.append({"success": False, "error": str(e)})

        total_time = time.time() - start_time

        cluster_info = json.dumps({
            "cluster_name": cluster_name,
            "cluster_description": cluster_description,
            "endpoints": endpoint_list,
            "mode": mode,
            "total_time": total_time,
            "results_count": len(results),
            "successful": sum(1 for r in results if r.get("success")),
        }, indent=2)

        return (json.dumps(results, indent=2), total_time, cluster_info)


# Service type descriptions for LLM context
SERVICE_DESCRIPTIONS = {
    # === Image/Video Generation ===
    "comfyui": {
        "name": "ComfyUI",
        "docs": "https://github.com/comfyanonymous/ComfyUI",
        "description": "Node-based Stable Diffusion interface for images and video",
        "category": "image_generation"
    },
    "automatic1111": {
        "name": "Automatic1111 WebUI",
        "docs": "https://github.com/AUTOMATIC1111/stable-diffusion-webui",
        "description": "Popular SD WebUI with extensive extension ecosystem",
        "category": "image_generation"
    },
    "forge": {
        "name": "SD WebUI Forge",
        "docs": "https://github.com/lllyasviel/stable-diffusion-webui-forge",
        "description": "Optimized fork of A1111 with better memory management",
        "category": "image_generation"
    },
    "invokeai": {
        "name": "InvokeAI",
        "docs": "https://github.com/invoke-ai/InvokeAI",
        "description": "Creative engine for SD with unified canvas and node editor",
        "category": "image_generation"
    },
    "fooocus": {
        "name": "Fooocus",
        "docs": "https://github.com/lllyasviel/Fooocus",
        "description": "Simplified SD interface inspired by Midjourney",
        "category": "image_generation"
    },
    "swarmui": {
        "name": "SwarmUI",
        "docs": "https://github.com/mcmonkeyprojects/SwarmUI",
        "description": "Modular SD WebUI with ComfyUI backend",
        "category": "image_generation"
    },
    "kohya_ss": {
        "name": "Kohya SS",
        "docs": "https://github.com/kohya-ss/sd-scripts",
        "description": "Training scripts for SD/SDXL/Flux LoRA and fine-tuning",
        "category": "training"
    },
    "stable_horde": {
        "name": "Stable Horde",
        "docs": "https://stablehorde.net/",
        "description": "Distributed SD generation network",
        "category": "image_generation"
    },

    # === LLM Inference ===
    "koboldcpp": {
        "name": "KoboldCpp",
        "docs": "https://github.com/LostRuins/koboldcpp",
        "description": "llama.cpp with Kobold API, GGUF/GGML models",
        "category": "llm"
    },
    "koboldai": {
        "name": "KoboldAI",
        "docs": "https://github.com/KoboldAI/KoboldAI-Client",
        "description": "Browser-based LLM interface with story features",
        "category": "llm"
    },
    "ollama": {
        "name": "Ollama",
        "docs": "https://ollama.ai",
        "description": "Local LLM runner with easy model management",
        "category": "llm"
    },
    "llamacpp_server": {
        "name": "llama.cpp Server",
        "docs": "https://github.com/ggerganov/llama.cpp",
        "description": "Pure C++ LLM inference, GGUF format",
        "category": "llm"
    },
    "text_gen_webui": {
        "name": "Text Generation WebUI (Oobabooga)",
        "docs": "https://github.com/oobabooga/text-generation-webui",
        "description": "Gradio web UI for running LLMs with multiple backends",
        "category": "llm"
    },
    "vllm": {
        "name": "vLLM",
        "docs": "https://github.com/vllm-project/vllm",
        "description": "High-throughput LLM serving with PagedAttention",
        "category": "llm"
    },
    "lmdeploy": {
        "name": "LMDeploy",
        "docs": "https://github.com/InternLM/lmdeploy",
        "description": "Efficient LLM deployment with TurboMind engine",
        "category": "llm"
    },
    "tgi": {
        "name": "Text Generation Inference (TGI)",
        "docs": "https://github.com/huggingface/text-generation-inference",
        "description": "HuggingFace production LLM serving",
        "category": "llm"
    },
    "localai": {
        "name": "LocalAI",
        "docs": "https://localai.io/",
        "description": "OpenAI-compatible local AI server, multi-model",
        "category": "llm"
    },
    "jan": {
        "name": "Jan",
        "docs": "https://jan.ai/",
        "description": "Open-source ChatGPT alternative with local LLMs",
        "category": "llm"
    },
    "lmstudio": {
        "name": "LM Studio",
        "docs": "https://lmstudio.ai/",
        "description": "Desktop app for running local LLMs",
        "category": "llm"
    },
    "gpt4all": {
        "name": "GPT4All",
        "docs": "https://gpt4all.io/",
        "description": "Local LLM ecosystem with cross-platform support",
        "category": "llm"
    },
    "exllama": {
        "name": "ExLlamaV2",
        "docs": "https://github.com/turboderp/exllamav2",
        "description": "Fast GPTQ/EXL2 inference for consumer GPUs",
        "category": "llm"
    },
    "tabbyapi": {
        "name": "TabbyAPI",
        "docs": "https://github.com/theroyallab/tabbyAPI",
        "description": "ExLlamaV2 API server with OpenAI compatibility",
        "category": "llm"
    },
    "aphrodite": {
        "name": "Aphrodite Engine",
        "docs": "https://github.com/PygmalionAI/aphrodite-engine",
        "description": "vLLM fork optimized for roleplay and creative writing",
        "category": "llm"
    },

    # === OpenAI Compatible ===
    "openai_compatible": {
        "name": "OpenAI-Compatible API",
        "docs": "https://platform.openai.com/docs/api-reference",
        "description": "Any server implementing OpenAI API spec",
        "category": "llm"
    },
    "litellm": {
        "name": "LiteLLM",
        "docs": "https://github.com/BerriAI/litellm",
        "description": "Unified API for 100+ LLM providers",
        "category": "llm"
    },
    "openrouter": {
        "name": "OpenRouter",
        "docs": "https://openrouter.ai/",
        "description": "Unified API gateway for many LLM providers",
        "category": "llm"
    },

    # === Speech-to-Text ===
    "whisper": {
        "name": "Whisper",
        "docs": "https://github.com/openai/whisper",
        "description": "OpenAI speech recognition model",
        "category": "stt"
    },
    "faster_whisper": {
        "name": "Faster Whisper",
        "docs": "https://github.com/SYSTRAN/faster-whisper",
        "description": "CTranslate2 optimized Whisper, 4x faster",
        "category": "stt"
    },
    "whisper_cpp": {
        "name": "whisper.cpp",
        "docs": "https://github.com/ggerganov/whisper.cpp",
        "description": "C++ port of Whisper, CPU optimized",
        "category": "stt"
    },
    "whisperx": {
        "name": "WhisperX",
        "docs": "https://github.com/m-bain/whisperX",
        "description": "Whisper with word-level timestamps and diarization",
        "category": "stt"
    },
    "insanely_fast_whisper": {
        "name": "Insanely Fast Whisper",
        "docs": "https://github.com/Vaibhavs10/insanely-fast-whisper",
        "description": "Batched Whisper inference with flash attention",
        "category": "stt"
    },
    "whisper_jax": {
        "name": "Whisper JAX",
        "docs": "https://github.com/sanchit-gandhi/whisper-jax",
        "description": "JAX implementation of Whisper, 70x faster",
        "category": "stt"
    },
    "nemo_asr": {
        "name": "NVIDIA NeMo ASR",
        "docs": "https://github.com/NVIDIA/NeMo",
        "description": "NVIDIA's ASR models including Canary and Parakeet",
        "category": "stt"
    },
    "vosk": {
        "name": "Vosk",
        "docs": "https://alphacephei.com/vosk/",
        "description": "Offline speech recognition toolkit",
        "category": "stt"
    },
    "deepspeech": {
        "name": "DeepSpeech",
        "docs": "https://github.com/mozilla/DeepSpeech",
        "description": "Mozilla's open source speech-to-text",
        "category": "stt"
    },

    # === Text-to-Speech ===
    "coqui_tts": {
        "name": "Coqui TTS",
        "docs": "https://github.com/coqui-ai/TTS",
        "description": "Deep learning TTS library with many models",
        "category": "tts"
    },
    "xtts": {
        "name": "XTTS",
        "docs": "https://github.com/coqui-ai/TTS",
        "description": "Cross-lingual voice cloning TTS",
        "category": "tts"
    },
    "xtts_v2": {
        "name": "XTTS v2",
        "docs": "https://docs.coqui.ai/en/latest/models/xtts.html",
        "description": "Improved XTTS with better prosody",
        "category": "tts"
    },
    "alltalk": {
        "name": "AllTalk TTS",
        "docs": "https://github.com/erew123/alltalk_tts",
        "description": "Multi-engine TTS with web interface",
        "category": "tts"
    },
    "silero_tts": {
        "name": "Silero TTS",
        "docs": "https://github.com/snakers4/silero-models",
        "description": "Fast and lightweight TTS models",
        "category": "tts"
    },
    "piper": {
        "name": "Piper",
        "docs": "https://github.com/rhasspy/piper",
        "description": "Fast local neural TTS system",
        "category": "tts"
    },
    "bark": {
        "name": "Bark",
        "docs": "https://github.com/suno-ai/bark",
        "description": "Text-to-audio with speech, music, sound effects",
        "category": "tts"
    },
    "tortoise_tts": {
        "name": "Tortoise TTS",
        "docs": "https://github.com/neonbjb/tortoise-tts",
        "description": "Multi-voice TTS with voice cloning",
        "category": "tts"
    },
    "valle": {
        "name": "VALL-E",
        "docs": "https://github.com/microsoft/unilm/tree/master/valle",
        "description": "Microsoft's neural codec language model for TTS",
        "category": "tts"
    },
    "styletts2": {
        "name": "StyleTTS 2",
        "docs": "https://github.com/yl4579/StyleTTS2",
        "description": "Style-based TTS with diffusion and adversarial training",
        "category": "tts"
    },
    "openvoice": {
        "name": "OpenVoice",
        "docs": "https://github.com/myshell-ai/OpenVoice",
        "description": "Instant voice cloning with emotion and accent control",
        "category": "tts"
    },
    "rvc": {
        "name": "RVC (Retrieval-based Voice Conversion)",
        "docs": "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
        "description": "AI voice conversion with minimal training",
        "category": "voice_conversion"
    },
    "so_vits_svc": {
        "name": "So-VITS-SVC",
        "docs": "https://github.com/svc-develop-team/so-vits-svc",
        "description": "Singing voice conversion with VITS",
        "category": "voice_conversion"
    },
    "fish_speech": {
        "name": "Fish Speech",
        "docs": "https://github.com/fishaudio/fish-speech",
        "description": "Fast zero-shot multilingual TTS",
        "category": "tts"
    },

    # === Embeddings ===
    "embeddings_tei": {
        "name": "Text Embeddings Inference",
        "docs": "https://github.com/huggingface/text-embeddings-inference",
        "description": "HuggingFace high-performance embeddings server",
        "category": "embeddings"
    },
    "sentence_transformers": {
        "name": "Sentence Transformers",
        "docs": "https://www.sbert.net/",
        "description": "Multilingual sentence embeddings",
        "category": "embeddings"
    },
    "infinity_emb": {
        "name": "Infinity Embeddings",
        "docs": "https://github.com/michaelfeil/infinity",
        "description": "High-throughput embedding inference server",
        "category": "embeddings"
    },
    "fastembed": {
        "name": "FastEmbed",
        "docs": "https://github.com/qdrant/fastembed",
        "description": "Lightweight ONNX-based embeddings",
        "category": "embeddings"
    },

    # === Vision/Multimodal ===
    "llava": {
        "name": "LLaVA",
        "docs": "https://github.com/haotian-liu/LLaVA",
        "description": "Large Language and Vision Assistant",
        "category": "vision"
    },
    "cogvlm": {
        "name": "CogVLM",
        "docs": "https://github.com/THUDM/CogVLM",
        "description": "Visual expert for pretrained language models",
        "category": "vision"
    },
    "qwen_vl": {
        "name": "Qwen-VL",
        "docs": "https://github.com/QwenLM/Qwen-VL",
        "description": "Alibaba's multimodal LLM",
        "category": "vision"
    },
    "moondream": {
        "name": "Moondream",
        "docs": "https://github.com/vikhyat/moondream",
        "description": "Tiny vision language model",
        "category": "vision"
    },
    "bakllava": {
        "name": "BakLLaVA",
        "docs": "https://github.com/SkunkworksAI/BakLLaVA",
        "description": "LLaVA variant with Mistral backbone",
        "category": "vision"
    },

    # === Video Generation ===
    "animatediff": {
        "name": "AnimateDiff",
        "docs": "https://github.com/guoyww/AnimateDiff",
        "description": "Animate SD images with motion modules",
        "category": "video"
    },
    "svd": {
        "name": "Stable Video Diffusion",
        "docs": "https://stability.ai/stable-video",
        "description": "Image-to-video generation from Stability AI",
        "category": "video"
    },
    "mochi": {
        "name": "Mochi 1",
        "docs": "https://www.genmo.ai/",
        "description": "Genmo's high-quality video generation model",
        "category": "video"
    },
    "cogvideo": {
        "name": "CogVideo",
        "docs": "https://github.com/THUDM/CogVideo",
        "description": "Text-to-video generation from Tsinghua",
        "category": "video"
    },
    "hunyuan_video": {
        "name": "Hunyuan Video",
        "docs": "https://github.com/Tencent/HunyuanVideo",
        "description": "Tencent's video generation foundation model",
        "category": "video"
    },
    "ltx_video": {
        "name": "LTX-Video",
        "docs": "https://github.com/Lightricks/LTX-Video",
        "description": "Lightricks video generation model",
        "category": "video"
    },
    "wan": {
        "name": "Alibaba Wan",
        "docs": "https://github.com/alibaba/wan",
        "description": "Alibaba's text-to-video model",
        "category": "video"
    },

    # === Audio Generation ===
    "audiocraft": {
        "name": "AudioCraft",
        "docs": "https://github.com/facebookresearch/audiocraft",
        "description": "Meta's audio generation library (MusicGen, AudioGen)",
        "category": "audio"
    },
    "musicgen": {
        "name": "MusicGen",
        "docs": "https://github.com/facebookresearch/audiocraft",
        "description": "Text-to-music generation from Meta",
        "category": "audio"
    },
    "audioldm": {
        "name": "AudioLDM",
        "docs": "https://github.com/haoheliu/AudioLDM",
        "description": "Text-to-audio with latent diffusion",
        "category": "audio"
    },
    "riffusion": {
        "name": "Riffusion",
        "docs": "https://github.com/riffusion/riffusion",
        "description": "Music generation via spectrogram diffusion",
        "category": "audio"
    },
    "stable_audio": {
        "name": "Stable Audio",
        "docs": "https://stability.ai/stable-audio",
        "description": "Stability AI's audio generation model",
        "category": "audio"
    },

    # === Upscaling/Enhancement ===
    "realesrgan": {
        "name": "Real-ESRGAN",
        "docs": "https://github.com/xinntao/Real-ESRGAN",
        "description": "Practical image restoration with pure-synthetic training",
        "category": "upscale"
    },
    "swinir": {
        "name": "SwinIR",
        "docs": "https://github.com/JingyunLiang/SwinIR",
        "description": "Image restoration with Swin Transformer",
        "category": "upscale"
    },
    "gfpgan": {
        "name": "GFPGAN",
        "docs": "https://github.com/TencentARC/GFPGAN",
        "description": "Face restoration with generative priors",
        "category": "face_restore"
    },
    "codeformer": {
        "name": "CodeFormer",
        "docs": "https://github.com/sczhou/CodeFormer",
        "description": "Robust face restoration with codebook lookup",
        "category": "face_restore"
    },

    # === Other AI Services ===
    "controlnet": {
        "name": "ControlNet",
        "docs": "https://github.com/lllyasviel/ControlNet",
        "description": "Adding conditional control to SD models",
        "category": "control"
    },
    "ip_adapter": {
        "name": "IP-Adapter",
        "docs": "https://github.com/tencent-ailab/IP-Adapter",
        "description": "Image prompt adapter for SD",
        "category": "control"
    },
    "segment_anything": {
        "name": "Segment Anything (SAM)",
        "docs": "https://github.com/facebookresearch/segment-anything",
        "description": "Foundation model for image segmentation",
        "category": "segmentation"
    },
    "grounding_dino": {
        "name": "Grounding DINO",
        "docs": "https://github.com/IDEA-Research/GroundingDINO",
        "description": "Open-set object detection with language",
        "category": "detection"
    },
    "depth_anything": {
        "name": "Depth Anything",
        "docs": "https://github.com/LiheYoung/Depth-Anything",
        "description": "Foundation model for monocular depth estimation",
        "category": "depth"
    },
    "marigold": {
        "name": "Marigold",
        "docs": "https://github.com/prs-eth/Marigold",
        "description": "Diffusion-based depth estimation",
        "category": "depth"
    },
    "florence2": {
        "name": "Florence-2",
        "docs": "https://huggingface.co/microsoft/Florence-2-large",
        "description": "Microsoft's unified vision foundation model",
        "category": "vision"
    },

    "custom": {
        "name": "Custom Service",
        "docs": "",
        "description": "User-defined API endpoint",
        "category": "custom"
    },
}
