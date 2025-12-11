"""
RemoteComfyUI Node - Execute any workflow on remote ComfyUI instances.

Comprehensive node for distributed generation across multiple machines.
Supports SD, SDXL, Flux, video generation, and any other ComfyUI workflow.
"""

import json
import time
import uuid
import base64
import requests
import io
from typing import Dict, Any, Tuple, Optional, List, Union
from PIL import Image
import numpy as np


class RemoteComfyUI:
    """
    Execute any workflow on a remote ComfyUI instance.

    Comprehensive node supporting:
    - SD 1.5, SDXL, Flux image generation
    - Video generation (AnimateDiff, SVD, Mochi, etc.)
    - Any custom workflow
    - Multiple output types (images, video, audio, text)
    - Dynamic parameter injection
    - Real-time progress tracking
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "endpoint": ("STRING", {
                    "default": "http://localhost:8188",
                    "placeholder": "http://192.168.1.101:8188"
                }),
                "mode": (["workflow_json", "workflow_file", "api_json"], {
                    "default": "workflow_json"
                }),
            },
            "optional": {
                # Workflow input (one of these based on mode)
                "workflow_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste workflow JSON or API format..."
                }),
                "workflow_file": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/workflow.json"
                }),

                # Parameter injection
                "inject_params": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": '{"3": {"inputs": {"seed": 12345}}}'
                }),

                # Common overrides (inject into detected nodes)
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 150}),
                "cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0}),
                "width": ("INT", {"default": -1, "min": -1, "max": 8192}),
                "height": ("INT", {"default": -1, "min": -1, "max": 8192}),
                "denoise": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0}),

                # Video-specific
                "frames": ("INT", {"default": -1, "min": -1, "max": 1000}),
                "fps": ("INT", {"default": -1, "min": -1, "max": 120}),

                # Execution control
                "output_node_id": ("STRING", {"default": ""}),
                "output_type": (["auto", "images", "video", "audio", "latent", "text"], {"default": "auto"}),
                "timeout": ("INT", {"default": 600, "min": 30, "max": 7200}),
                "poll_interval": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),

                # Machine info for Performance Lab
                "machine_name": ("STRING", {"default": "", "placeholder": "GPU Server 1"}),
                "machine_specs": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "RTX 4090 24GB, 64GB RAM, i9-13900K"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("images", "video_path", "execution_time", "endpoint_info", "raw_output")
    FUNCTION = "execute_remote"
    CATEGORY = "NetworkServices/ComfyUI"
    OUTPUT_NODE = True

    NETWORK_SERVICE = True
    SERVICE_TYPE = "comfyui"
    ENDPOINT_PARAM = "endpoint"

    def execute_remote(
        self,
        endpoint: str,
        mode: str = "workflow_json",
        workflow_json: str = "",
        workflow_file: str = "",
        inject_params: str = "",
        positive_prompt: str = "",
        negative_prompt: str = "",
        seed: int = -1,
        steps: int = -1,
        cfg: float = -1.0,
        width: int = -1,
        height: int = -1,
        denoise: float = -1.0,
        frames: int = -1,
        fps: int = -1,
        output_node_id: str = "",
        output_type: str = "auto",
        timeout: int = 600,
        poll_interval: float = 1.0,
        machine_name: str = "",
        machine_specs: str = ""
    ) -> Tuple[Any, str, float, str, str]:
        """Execute workflow on remote ComfyUI."""

        endpoint = endpoint.rstrip('/')
        client_id = str(uuid.uuid4())

        # Load workflow based on mode
        workflow = self._load_workflow(mode, workflow_json, workflow_file)

        # Inject parameters
        workflow = self._inject_common_params(
            workflow, positive_prompt, negative_prompt,
            seed, steps, cfg, width, height, denoise, frames, fps
        )

        # Inject custom params
        if inject_params.strip():
            try:
                custom_params = json.loads(inject_params)
                workflow = self._deep_merge(workflow, custom_params)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse inject_params: {e}")

        # Auto-detect output node if not specified
        if not output_node_id:
            output_node_id, detected_type = self._find_output_node(workflow)
            if output_type == "auto":
                output_type = detected_type

        start_time = time.time()

        # Get remote system info before execution
        system_stats = self._get_system_stats(endpoint)

        # Queue the workflow
        prompt_id = self._queue_workflow(endpoint, workflow, client_id)

        # Wait for completion with progress tracking
        result = self._wait_for_completion(
            endpoint, prompt_id, output_node_id,
            client_id, output_type, timeout, poll_interval
        )

        execution_time = time.time() - start_time

        # Build comprehensive endpoint info
        endpoint_info = json.dumps({
            "endpoint": endpoint,
            "service_type": "remote_comfyui",
            "execution_time": execution_time,
            "prompt_id": prompt_id,
            "output_type": output_type,
            "machine_name": machine_name,
            "machine_specs": machine_specs,
            "system_stats": system_stats,
        }, indent=2)

        # Parse results based on type
        images = None
        video_path = ""
        raw_output = json.dumps(result, indent=2) if result else "{}"

        if output_type == "images" or output_type == "auto":
            if "images" in result:
                images = self._fetch_images(endpoint, result["images"])
        elif output_type == "video":
            if "gifs" in result or "videos" in result:
                video_data = result.get("gifs", result.get("videos", []))
                if video_data:
                    video_path = self._fetch_video(endpoint, video_data[0])

        return (images, video_path, execution_time, endpoint_info, raw_output)

    def _load_workflow(self, mode: str, workflow_json: str, workflow_file: str) -> Dict:
        """Load workflow from JSON string or file."""
        if mode == "workflow_file" and workflow_file:
            with open(workflow_file, 'r') as f:
                return json.load(f)
        elif workflow_json:
            return json.loads(workflow_json)
        else:
            raise ValueError("No workflow provided. Supply workflow_json or workflow_file.")

    def _inject_common_params(
        self,
        workflow: Dict,
        positive: str,
        negative: str,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
        denoise: float,
        frames: int,
        fps: int
    ) -> Dict:
        """Inject common parameters into detected nodes."""

        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue

            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            # Prompts
            if positive and class_type in ["CLIPTextEncode", "CLIPTextEncodeSDXL"]:
                if "text" in inputs or "text_g" in inputs:
                    # Try to detect if positive or negative by connection analysis
                    # For now, use naming conventions in title
                    title = node.get("_meta", {}).get("title", "").lower()
                    if "positive" in title or "pos" in title:
                        if "text" in inputs:
                            inputs["text"] = positive
                        if "text_g" in inputs:
                            inputs["text_g"] = positive
                            inputs["text_l"] = positive

            if negative and class_type in ["CLIPTextEncode", "CLIPTextEncodeSDXL"]:
                title = node.get("_meta", {}).get("title", "").lower()
                if "negative" in title or "neg" in title:
                    if "text" in inputs:
                        inputs["text"] = negative
                    if "text_g" in inputs:
                        inputs["text_g"] = negative
                        inputs["text_l"] = negative

            # KSampler parameters
            if class_type in ["KSampler", "KSamplerAdvanced", "SamplerCustom"]:
                if seed >= 0 and "seed" in inputs:
                    inputs["seed"] = seed
                if steps > 0 and "steps" in inputs:
                    inputs["steps"] = steps
                if cfg > 0 and "cfg" in inputs:
                    inputs["cfg"] = cfg
                if denoise >= 0 and "denoise" in inputs:
                    inputs["denoise"] = denoise

            # Latent dimensions
            if class_type in ["EmptyLatentImage", "EmptySD3LatentImage"]:
                if width > 0 and "width" in inputs:
                    inputs["width"] = width
                if height > 0 and "height" in inputs:
                    inputs["height"] = height

            # Video-specific (AnimateDiff, SVD, etc.)
            if class_type in ["EmptyLatentImageHD", "SVD_img2vid_Conditioning", "AnimateDiffLoaderWithContext"]:
                if frames > 0:
                    if "batch_size" in inputs:
                        inputs["batch_size"] = frames
                    if "video_frames" in inputs:
                        inputs["video_frames"] = frames
                if fps > 0 and "fps" in inputs:
                    inputs["fps"] = fps

        return workflow

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge override into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _find_output_node(self, workflow: Dict) -> Tuple[str, str]:
        """Find output node and detect type."""

        # Priority order for output detection
        output_types = [
            # Video outputs
            (["VHS_VideoCombine", "SaveAnimatedWEBP", "SaveAnimatedPNG"], "video"),
            # Image outputs
            (["SaveImage", "PreviewImage", "SaveImageWebsocket"], "images"),
            # Audio outputs
            (["SaveAudio", "PreviewAudio"], "audio"),
            # Text outputs
            (["ShowText", "SaveText"], "text"),
        ]

        for class_types, out_type in output_types:
            for node_id, node in workflow.items():
                if isinstance(node, dict) and node.get("class_type") in class_types:
                    return str(node_id), out_type

        raise ValueError("No output node found. Please specify output_node_id.")

    def _get_system_stats(self, endpoint: str) -> Optional[Dict]:
        """Get system stats from remote ComfyUI."""
        try:
            response = requests.get(f"{endpoint}/system_stats", timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

    def _queue_workflow(self, endpoint: str, workflow: Dict, client_id: str) -> str:
        """Queue workflow on remote ComfyUI."""
        payload = {"prompt": workflow, "client_id": client_id}

        try:
            response = requests.post(f"{endpoint}/prompt", json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("prompt_id")
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                pass
            raise RuntimeError(f"Failed to queue workflow: {e}\n{error_detail}")
        except Exception as e:
            raise RuntimeError(f"Failed to queue workflow: {e}")

    def _wait_for_completion(
        self,
        endpoint: str,
        prompt_id: str,
        output_node_id: str,
        client_id: str,
        output_type: str,
        timeout: int,
        poll_interval: float
    ) -> Dict:
        """Wait for workflow completion and return outputs."""

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{endpoint}/history/{prompt_id}", timeout=10)
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        if output_node_id in outputs:
                            return outputs[output_node_id]
            except:
                pass

            time.sleep(poll_interval)

        raise TimeoutError(f"Workflow timed out after {timeout}s")

    def _fetch_images(self, endpoint: str, image_refs: List[Dict]) -> Any:
        """Fetch images from remote ComfyUI."""
        images = []

        for img_ref in image_refs:
            params = {
                "filename": img_ref.get("filename"),
                "subfolder": img_ref.get("subfolder", ""),
                "type": img_ref.get("type", "output")
            }

            try:
                response = requests.get(f"{endpoint}/view", params=params, timeout=60)
                response.raise_for_status()

                img = Image.open(io.BytesIO(response.content))
                img_array = np.array(img).astype(np.float32) / 255.0

                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]

                images.append(img_array)
            except Exception as e:
                print(f"Warning: Failed to fetch image: {e}")

        if images:
            return np.stack(images)
        return None

    def _fetch_video(self, endpoint: str, video_ref: Dict) -> str:
        """Fetch video from remote ComfyUI and save locally."""
        import tempfile
        import os

        params = {
            "filename": video_ref.get("filename"),
            "subfolder": video_ref.get("subfolder", ""),
            "type": video_ref.get("type", "output")
        }

        try:
            response = requests.get(f"{endpoint}/view", params=params, timeout=120)
            response.raise_for_status()

            # Save to temp file
            ext = os.path.splitext(params["filename"])[1] or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(response.content)
                return f.name
        except Exception as e:
            print(f"Warning: Failed to fetch video: {e}")
            return ""


class RemoteComfyUISimple:
    """
    Simplified Remote ComfyUI for quick SD/Flux generation.

    Auto-builds a basic workflow - no JSON needed.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "endpoint": ("STRING", {"default": "http://localhost:8188"}),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_type": (["sd15", "sdxl", "flux_dev", "flux_schnell"], {"default": "sdxl"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality"}),
                "checkpoint": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "sampler": (["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim"], {"default": "euler"}),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform"], {"default": "normal"}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 3600}),
                "machine_name": ("STRING", {"default": ""}),
                "machine_specs": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "execution_time", "endpoint_info")
    FUNCTION = "generate"
    CATEGORY = "NetworkServices/ComfyUI"

    NETWORK_SERVICE = True
    SERVICE_TYPE = "comfyui"
    ENDPOINT_PARAM = "endpoint"

    def generate(
        self,
        endpoint: str,
        positive_prompt: str,
        model_type: str = "sdxl",
        negative_prompt: str = "blurry, low quality",
        checkpoint: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = -1,
        sampler: str = "euler",
        scheduler: str = "normal",
        timeout: int = 300,
        machine_name: str = "",
        machine_specs: str = ""
    ) -> Tuple[Any, float, str]:
        """Generate using auto-built workflow."""

        import random
        if seed < 0:
            seed = random.randint(0, 2147483647)

        # Build workflow based on model type
        workflow = self._build_workflow(
            model_type, positive_prompt, negative_prompt,
            checkpoint, width, height, steps, cfg, seed, sampler, scheduler
        )

        executor = RemoteComfyUI()
        images, _, exec_time, endpoint_info, _ = executor.execute_remote(
            endpoint=endpoint,
            workflow_json=json.dumps(workflow),
            timeout=timeout,
            machine_name=machine_name,
            machine_specs=machine_specs
        )

        return (images, exec_time, endpoint_info)

    def _build_workflow(
        self,
        model_type: str,
        positive: str,
        negative: str,
        checkpoint: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        sampler: str,
        scheduler: str
    ) -> Dict:
        """Build workflow for specified model type."""

        # Default checkpoints
        default_checkpoints = {
            "sd15": "v1-5-pruned-emaonly.safetensors",
            "sdxl": "sd_xl_base_1.0.safetensors",
            "flux_dev": "flux1-dev.safetensors",
            "flux_schnell": "flux1-schnell.safetensors"
        }

        ckpt = checkpoint if checkpoint else default_checkpoints.get(model_type, "")

        if model_type in ["flux_dev", "flux_schnell"]:
            return self._build_flux_workflow(positive, ckpt, width, height, steps, seed, sampler, scheduler)
        else:
            return self._build_sd_workflow(positive, negative, ckpt, width, height, steps, cfg, seed, sampler, scheduler)

    def _build_sd_workflow(self, positive, negative, ckpt, w, h, steps, cfg, seed, sampler, scheduler):
        """Build SD/SDXL workflow."""
        return {
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive"}, "inputs": {"text": positive, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "_meta": {"title": "Negative"}, "inputs": {"text": negative, "clip": ["4", 1]}},
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
                "scheduler": scheduler, "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]
            }},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "remote", "images": ["8", 0]}}
        }

    def _build_flux_workflow(self, positive, ckpt, w, h, steps, seed, sampler, scheduler):
        """Build Flux workflow (simplified)."""
        return {
            "4": {"class_type": "UNETLoader", "inputs": {"unet_name": ckpt, "weight_dtype": "default"}},
            "11": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5xxl_fp16.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}},
            "12": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
            "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["11", 0]}},
            "25": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": sampler,
                "scheduler": scheduler, "denoise": 1.0,
                "model": ["4", 0], "positive": ["25", 0], "negative": ["6", 0], "latent_image": ["5", 0]
            }},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["12", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "flux", "images": ["8", 0]}}
        }


# Helper functions for Performance Lab integration
def get_remote_system_stats(endpoint: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Get system stats from a remote ComfyUI instance."""
    try:
        response = requests.get(f"{endpoint}/system_stats", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_remote_queue_status(endpoint: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Get queue status from a remote ComfyUI instance."""
    try:
        response = requests.get(f"{endpoint}/queue", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_remote_object_info(endpoint: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Get available nodes from a remote ComfyUI instance."""
    try:
        response = requests.get(f"{endpoint}/object_info", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None
