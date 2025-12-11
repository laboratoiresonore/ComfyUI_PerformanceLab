"""
Endpoint Health Check Nodes - Monitor and verify network service availability.

Provides health checking, latency measurement, and service discovery
for multi-machine AI pipelines.
"""

import json
import logging
import time
import socket
import requests
import concurrent.futures
from typing import Dict, Any, Tuple, Optional, List
from .network_api import SERVICE_DESCRIPTIONS

# Setup module logger
logger = logging.getLogger("performance_lab.nodes.health_check")


class EndpointHealthCheck:
    """
    Health Check for a single endpoint.

    Verifies connectivity, measures latency, and optionally retrieves
    system information from supported services.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        service_types = list(SERVICE_DESCRIPTIONS.keys())
        return {
            "required": {
                "endpoint": ("STRING", {
                    "default": "http://localhost:8188",
                    "placeholder": "http://192.168.1.100:8188"
                }),
            },
            "optional": {
                "service_type": (service_types, {"default": "comfyui"}),
                "timeout": ("INT", {"default": 10, "min": 1, "max": 60}),
                "check_path": ("STRING", {
                    "default": "",
                    "placeholder": "/health or /api/v1/model (auto if empty)"
                }),
                "expected_status": ("INT", {"default": 200, "min": 100, "max": 599}),
                "measure_latency": ("BOOLEAN", {"default": True}),
                "latency_samples": ("INT", {"default": 3, "min": 1, "max": 10}),
                "get_system_info": ("BOOLEAN", {"default": True}),

                # Machine info
                "machine_name": ("STRING", {"default": ""}),
                "machine_specs": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("is_healthy", "latency_ms", "system_info", "full_report")
    FUNCTION = "check"
    CATEGORY = "NetworkServices/HealthCheck"

    NETWORK_SERVICE = True
    SERVICE_TYPE = "health_check"
    ENDPOINT_PARAM = "endpoint"

    # Health check paths per service type
    HEALTH_PATHS = {
        "comfyui": "/system_stats",
        "automatic1111": "/sdapi/v1/memory",
        "forge": "/sdapi/v1/memory",
        "invokeai": "/api/v1/app/version",
        "fooocus": "/",
        "swarmui": "/API/GetCurrentStatus",
        "koboldcpp": "/api/v1/model",
        "koboldai": "/api/v1/model",
        "ollama": "/api/tags",
        "llamacpp_server": "/health",
        "text_gen_webui": "/api/v1/model",
        "vllm": "/health",
        "lmdeploy": "/v1/models",
        "tgi": "/health",
        "localai": "/readyz",
        "whisper": "/",
        "faster_whisper": "/",
        "coqui_tts": "/api/tts/languages",
        "xtts": "/languages",
        "alltalk": "/api/voices",
        "openai_compatible": "/v1/models",
    }

    def check(
        self,
        endpoint: str,
        service_type: str = "comfyui",
        timeout: int = 10,
        check_path: str = "",
        expected_status: int = 200,
        measure_latency: bool = True,
        latency_samples: int = 3,
        get_system_info: bool = True,
        machine_name: str = "",
        machine_specs: str = ""
    ) -> Tuple[bool, float, str, str]:
        """Perform health check on endpoint."""

        endpoint = endpoint.rstrip('/')
        is_healthy = False
        latency_ms = -1.0
        system_info = "{}"
        report_data = {
            "endpoint": endpoint,
            "service_type": service_type,
            "machine_name": machine_name,
            "machine_specs": machine_specs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Determine health check path
        if not check_path:
            check_path = self.HEALTH_PATHS.get(service_type, "/")

        check_url = f"{endpoint}{check_path}"

        # 1. Basic connectivity check
        try:
            start = time.time()
            response = requests.get(check_url, timeout=timeout)
            initial_latency = (time.time() - start) * 1000

            is_healthy = response.status_code == expected_status
            report_data["status_code"] = response.status_code
            report_data["initial_latency_ms"] = round(initial_latency, 2)

            # Try to parse response for system info
            if get_system_info:
                try:
                    system_info = json.dumps(response.json(), indent=2)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Could not parse system info as JSON: {e}")
                    system_info = "{}"

        except requests.exceptions.Timeout:
            report_data["error"] = f"Timeout after {timeout}s"
        except requests.exceptions.ConnectionError as e:
            report_data["error"] = f"Connection failed: {str(e)[:100]}"
        except Exception as e:
            report_data["error"] = f"Error: {str(e)[:100]}"

        # 2. Measure latency with multiple samples
        if is_healthy and measure_latency:
            latencies = []
            for sample_num in range(latency_samples):
                try:
                    start = time.time()
                    requests.get(check_url, timeout=timeout)
                    latencies.append((time.time() - start) * 1000)
                except requests.exceptions.Timeout:
                    logger.debug(f"Latency sample {sample_num} timed out")
                except requests.exceptions.ConnectionError as e:
                    logger.debug(f"Latency sample {sample_num} connection error: {e}")
                except Exception as e:
                    logger.debug(f"Latency sample {sample_num} failed: {e}")

            if latencies:
                latency_ms = sum(latencies) / len(latencies)
                report_data["latency_samples"] = [round(l, 2) for l in latencies]
                report_data["latency_avg_ms"] = round(latency_ms, 2)
                report_data["latency_min_ms"] = round(min(latencies), 2)
                report_data["latency_max_ms"] = round(max(latencies), 2)

        # 3. Get additional system info for known services
        if is_healthy and get_system_info:
            extra_info = self._get_service_specific_info(endpoint, service_type, timeout)
            if extra_info:
                report_data["service_info"] = extra_info

        report_data["is_healthy"] = is_healthy
        full_report = json.dumps(report_data, indent=2)

        return (is_healthy, latency_ms, system_info, full_report)

    def _get_service_specific_info(
        self,
        endpoint: str,
        service_type: str,
        timeout: int
    ) -> Optional[Dict]:
        """Get service-specific information."""

        info = {}

        try:
            if service_type == "comfyui":
                # Get system stats and object info
                stats = requests.get(f"{endpoint}/system_stats", timeout=timeout)
                if stats.status_code == 200:
                    info["system_stats"] = stats.json()

            elif service_type in ["automatic1111", "forge"]:
                # Get memory and options
                memory = requests.get(f"{endpoint}/sdapi/v1/memory", timeout=timeout)
                options = requests.get(f"{endpoint}/sdapi/v1/options", timeout=timeout)
                if memory.status_code == 200:
                    info["memory"] = memory.json()
                if options.status_code == 200:
                    opts = options.json()
                    info["model"] = opts.get("sd_model_checkpoint", "unknown")

            elif service_type in ["koboldcpp", "koboldai"]:
                # Get model info
                model = requests.get(f"{endpoint}/api/v1/model", timeout=timeout)
                config = requests.get(f"{endpoint}/api/v1/config/max_context_length", timeout=timeout)
                if model.status_code == 200:
                    info["model"] = model.json()
                if config.status_code == 200:
                    info["max_context"] = config.json().get("value")

            elif service_type == "ollama":
                # Get running models
                models = requests.get(f"{endpoint}/api/tags", timeout=timeout)
                if models.status_code == 200:
                    info["models"] = models.json()

            elif service_type in ["vllm", "lmdeploy", "tgi"]:
                # OpenAI-compatible models endpoint
                models = requests.get(f"{endpoint}/v1/models", timeout=timeout)
                if models.status_code == 200:
                    info["models"] = models.json()

        except requests.exceptions.Timeout:
            logger.debug(f"Timeout getting service-specific info from {endpoint}")
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"Connection error getting service info: {e}")
        except Exception as e:
            logger.warning(f"Error getting service-specific info from {endpoint}: {e}")

        return info if info else None


class MultiEndpointHealthCheck:
    """
    Health Check for multiple endpoints simultaneously.

    Useful for monitoring a cluster of services or verifying
    all machines in a distributed setup.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "endpoints": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "http://192.168.1.100:8188\nhttp://192.168.1.101:5001\nhttp://192.168.1.102:9000"
                }),
            },
            "optional": {
                "service_types": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "comfyui\nkoboldcpp\nfaster_whisper\n(one per line, or single for all)"
                }),
                "timeout": ("INT", {"default": 10, "min": 1, "max": 60}),
                "parallel": ("BOOLEAN", {"default": True}),
                "fail_fast": ("BOOLEAN", {"default": False}),

                # Cluster info
                "cluster_name": ("STRING", {"default": "", "placeholder": "Production Cluster"}),
                "cluster_description": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("all_healthy", "healthy_count", "total_count", "summary", "full_report")
    FUNCTION = "check_all"
    CATEGORY = "NetworkServices/HealthCheck"

    def check_all(
        self,
        endpoints: str,
        service_types: str = "",
        timeout: int = 10,
        parallel: bool = True,
        fail_fast: bool = False,
        cluster_name: str = "",
        cluster_description: str = ""
    ) -> Tuple[bool, int, int, str, str]:
        """Check health of multiple endpoints."""

        endpoint_list = [e.strip() for e in endpoints.strip().split("\n") if e.strip()]
        type_list = [t.strip() for t in service_types.strip().split("\n") if t.strip()]

        # Expand types to match endpoints
        if len(type_list) == 0:
            type_list = ["comfyui"] * len(endpoint_list)
        elif len(type_list) == 1:
            type_list = type_list * len(endpoint_list)
        elif len(type_list) < len(endpoint_list):
            type_list = type_list + ["comfyui"] * (len(endpoint_list) - len(type_list))

        results = []
        checker = EndpointHealthCheck()

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoint_list)) as executor:
                futures = {}
                for ep, st in zip(endpoint_list, type_list):
                    futures[executor.submit(checker.check, ep, st, timeout)] = (ep, st)

                for future in concurrent.futures.as_completed(futures):
                    ep, st = futures[future]
                    try:
                        is_healthy, latency, _, report = future.result()
                        results.append({
                            "endpoint": ep,
                            "service_type": st,
                            "healthy": is_healthy,
                            "latency_ms": round(latency, 2) if latency > 0 else None,
                            "report": json.loads(report)
                        })
                        if fail_fast and not is_healthy:
                            break
                    except Exception as e:
                        results.append({
                            "endpoint": ep,
                            "service_type": st,
                            "healthy": False,
                            "error": str(e)
                        })
        else:
            for ep, st in zip(endpoint_list, type_list):
                try:
                    is_healthy, latency, _, report = checker.check(ep, st, timeout)
                    results.append({
                        "endpoint": ep,
                        "service_type": st,
                        "healthy": is_healthy,
                        "latency_ms": round(latency, 2) if latency > 0 else None,
                        "report": json.loads(report)
                    })
                    if fail_fast and not is_healthy:
                        break
                except Exception as e:
                    results.append({
                        "endpoint": ep,
                        "service_type": st,
                        "healthy": False,
                        "error": str(e)
                    })

        # Calculate summary
        healthy_count = sum(1 for r in results if r.get("healthy"))
        total_count = len(endpoint_list)
        all_healthy = healthy_count == total_count

        # Build summary
        summary_lines = [
            f"Cluster: {cluster_name}" if cluster_name else "Endpoint Health Check",
            f"Status: {'ALL HEALTHY' if all_healthy else 'DEGRADED'}",
            f"Healthy: {healthy_count}/{total_count}",
            "",
            "Endpoints:",
        ]

        for r in results:
            status = "OK" if r.get("healthy") else "FAIL"
            latency = f" ({r['latency_ms']}ms)" if r.get("latency_ms") else ""
            summary_lines.append(f"  [{status}] {r['endpoint']} - {r['service_type']}{latency}")

        summary = "\n".join(summary_lines)

        # Full report
        full_report = json.dumps({
            "cluster_name": cluster_name,
            "cluster_description": cluster_description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "all_healthy": all_healthy,
            "healthy_count": healthy_count,
            "total_count": total_count,
            "results": results
        }, indent=2)

        return (all_healthy, healthy_count, total_count, summary, full_report)


def ping_host(host: str, port: int = None, timeout: float = 5.0) -> Tuple[bool, float]:
    """
    Check if a host is reachable via TCP.

    Returns (is_reachable, latency_ms)
    """
    try:
        # Parse host and port if port not specified
        if port is None:
            if "://" in host:
                from urllib.parse import urlparse
                parsed = urlparse(host)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
            else:
                port = 80

        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        latency = (time.time() - start) * 1000
        sock.close()

        return (result == 0, latency if result == 0 else -1)
    except socket.timeout:
        logger.debug(f"Socket timeout connecting to {host}:{port}")
        return (False, -1)
    except socket.gaierror as e:
        logger.debug(f"DNS resolution failed for {host}: {e}")
        return (False, -1)
    except OSError as e:
        logger.debug(f"OS error connecting to {host}:{port}: {e}")
        return (False, -1)
    except Exception as e:
        logger.warning(f"Unexpected error pinging {host}:{port}: {e}")
        return (False, -1)
