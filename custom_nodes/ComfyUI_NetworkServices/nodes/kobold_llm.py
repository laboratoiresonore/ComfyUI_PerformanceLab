"""
KoboldLLM Node - Connect to Kobold AI instances on the network.

Supports KoboldAI and KoboldCpp API endpoints.
"""

import json
import time
import requests
from typing import Dict, Any, Tuple, Optional


class KoboldLLM:
    """
    Generate text using a Kobold instance on the network.

    Connects to KoboldAI or KoboldCpp servers using the standard Kobold API.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt here..."
                }),
                "endpoint": ("STRING", {
                    "default": "http://localhost:5001",
                    "placeholder": "http://192.168.1.100:5001"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 64
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional system prompt..."
                }),
                "stop_sequences": ("STRING", {
                    "default": "",
                    "placeholder": "Comma-separated stop sequences"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 200
                }),
                "rep_pen": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.05
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 10,
                    "max": 600
                }),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("generated_text", "generation_time", "endpoint_info")
    FUNCTION = "generate"
    CATEGORY = "NetworkServices/LLM"

    # Metadata for Performance Lab detection
    NETWORK_SERVICE = True
    SERVICE_TYPE = "llm"
    ENDPOINT_PARAM = "endpoint"

    def generate(
        self,
        prompt: str,
        endpoint: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str = "",
        stop_sequences: str = "",
        top_p: float = 0.9,
        top_k: int = 40,
        rep_pen: float = 1.1,
        timeout: int = 120
    ) -> Tuple[str, float, str]:
        """Generate text using Kobold API."""

        # Clean endpoint URL
        endpoint = endpoint.rstrip('/')

        # Build the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Parse stop sequences
        stop_list = []
        if stop_sequences:
            stop_list = [s.strip() for s in stop_sequences.split(',') if s.strip()]

        # Build request payload (Kobold API format)
        payload = {
            "prompt": full_prompt,
            "max_length": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "rep_pen": rep_pen,
        }

        if stop_list:
            payload["stop_sequence"] = stop_list

        # Make the request
        start_time = time.time()
        try:
            response = requests.post(
                f"{endpoint}/api/v1/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout after {timeout}s connecting to {endpoint}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Kobold at {endpoint}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Kobold API error: {e}")

        generation_time = time.time() - start_time

        # Extract generated text
        if "results" in result and len(result["results"]) > 0:
            generated_text = result["results"][0].get("text", "")
        else:
            generated_text = result.get("text", "")

        # Build endpoint info for Performance Lab
        endpoint_info = json.dumps({
            "endpoint": endpoint,
            "service_type": "kobold_llm",
            "generation_time": generation_time,
            "tokens_requested": max_tokens,
        })

        return (generated_text, generation_time, endpoint_info)


class KoboldLLMAdvanced:
    """
    Advanced Kobold LLM node with full parameter control.

    Exposes all Kobold API parameters including sampler settings,
    grammar constraints, and more.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "endpoint": ("STRING", {"default": "http://localhost:5001"}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
            },
            "optional": {
                # Temperature and sampling
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 500}),
                "top_a": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tfs": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Repetition penalty
                "rep_pen": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 3.0, "step": 0.01}),
                "rep_pen_range": ("INT", {"default": 256, "min": 0, "max": 4096}),
                "rep_pen_slope": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),

                # Mirostat
                "mirostat_mode": (["disabled", "mirostat_1", "mirostat_2"], {"default": "disabled"}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),

                # Grammar (GBNF format for KoboldCpp)
                "grammar": ("STRING", {"multiline": True, "default": ""}),

                # Control
                "stop_sequences": ("STRING", {"default": ""}),
                "trim_stop": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "timeout": ("INT", {"default": 300, "min": 10, "max": 1800}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("generated_text", "generation_time", "endpoint_info", "tokens_generated")
    FUNCTION = "generate"
    CATEGORY = "NetworkServices/LLM"

    NETWORK_SERVICE = True
    SERVICE_TYPE = "llm"
    ENDPOINT_PARAM = "endpoint"

    def generate(
        self,
        prompt: str,
        endpoint: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        top_a: float = 0.0,
        typical: float = 1.0,
        tfs: float = 1.0,
        rep_pen: float = 1.1,
        rep_pen_range: int = 256,
        rep_pen_slope: float = 1.0,
        mirostat_mode: str = "disabled",
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        grammar: str = "",
        stop_sequences: str = "",
        trim_stop: bool = True,
        seed: int = -1,
        timeout: int = 300
    ) -> Tuple[str, float, str, int]:
        """Generate text with advanced Kobold API parameters."""

        endpoint = endpoint.rstrip('/')

        # Build payload
        payload = {
            "prompt": prompt,
            "max_length": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "top_a": top_a,
            "typical": typical,
            "tfs": tfs,
            "rep_pen": rep_pen,
            "rep_pen_range": rep_pen_range,
            "rep_pen_slope": rep_pen_slope,
            "trim_stop": trim_stop,
        }

        # Mirostat
        if mirostat_mode == "mirostat_1":
            payload["mirostat"] = 1
            payload["mirostat_tau"] = mirostat_tau
            payload["mirostat_eta"] = mirostat_eta
        elif mirostat_mode == "mirostat_2":
            payload["mirostat"] = 2
            payload["mirostat_tau"] = mirostat_tau
            payload["mirostat_eta"] = mirostat_eta

        # Grammar
        if grammar.strip():
            payload["grammar"] = grammar

        # Stop sequences
        if stop_sequences:
            payload["stop_sequence"] = [s.strip() for s in stop_sequences.split(',') if s.strip()]

        # Seed
        if seed >= 0:
            payload["sampler_seed"] = seed

        # Make request
        start_time = time.time()
        try:
            response = requests.post(
                f"{endpoint}/api/v1/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            raise RuntimeError(f"Kobold API error: {e}")

        generation_time = time.time() - start_time

        # Extract text
        if "results" in result and len(result["results"]) > 0:
            generated_text = result["results"][0].get("text", "")
        else:
            generated_text = result.get("text", "")

        # Estimate tokens (rough)
        tokens_generated = len(generated_text.split()) * 1.3  # Rough estimate

        endpoint_info = json.dumps({
            "endpoint": endpoint,
            "service_type": "kobold_llm_advanced",
            "generation_time": generation_time,
            "tokens_requested": max_tokens,
            "tokens_generated_est": int(tokens_generated),
        })

        return (generated_text, generation_time, endpoint_info, int(tokens_generated))


def get_kobold_model_info(endpoint: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Get model information from a Kobold endpoint."""
    try:
        response = requests.get(f"{endpoint}/api/v1/model", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_kobold_max_context(endpoint: str, timeout: int = 10) -> Optional[int]:
    """Get max context length from a Kobold endpoint."""
    try:
        response = requests.get(f"{endpoint}/api/v1/config/max_context_length", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return data.get("value")
    except:
        pass
    return None
