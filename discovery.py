"""
Performance Lab Service Discovery

Discovers AI services on local network (ComfyUI, KoboldCPP, Ollama).
Adapted from WhimWeaver's comfyui_manager.py discovery logic.
"""

import socket
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


@dataclass
class ServiceEndpoint:
    """Represents a discovered AI service."""
    host: str
    port: int
    service_type: str  # "comfyui", "kobold", "ollama"
    url: str
    status: str = "unknown"  # "online", "offline", "unknown"
    health_score: float = 1.0  # 0.0-1.0, higher = healthier
    response_time_ms: float = 0.0
    last_check: float = 0.0
    capabilities: List[str] = field(default_factory=list)  # ["sd", "flux", "video"]
    version: str = ""

    @property
    def is_online(self) -> bool:
        return self.status == "online"


# Known service ports
SERVICE_PORTS = {
    "comfyui": [8188, 8189, 8190],
    "kobold": [5001, 5000],
    "ollama": [11434],
}

# Health check endpoints for each service type
HEALTH_ENDPOINTS = {
    "comfyui": ["/system_stats", "/queue"],
    "kobold": ["/api/v1/info", "/api/v1/model"],
    "ollama": ["/api/tags", "/"],
}


def check_port(host: str, port: int, timeout: float = 0.1) -> bool:
    """Check if a port is open on a host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


def http_get(url: str, timeout: float = 2.0) -> Tuple[bool, Optional[Dict], float]:
    """
    Make HTTP GET request and measure response time.

    Returns:
        (success, response_data, response_time_ms)
    """
    start = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PerformanceLab/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            elapsed = (time.time() - start) * 1000
            return True, data, elapsed
    except:
        elapsed = (time.time() - start) * 1000
        return False, None, elapsed


def detect_service_type(host: str, port: int) -> Optional[str]:
    """Try to detect what type of service is running on a port."""

    # Try each service type's endpoints
    for service_type, endpoints in HEALTH_ENDPOINTS.items():
        for endpoint in endpoints:
            url = f"http://{host}:{port}{endpoint}"
            success, data, _ = http_get(url, timeout=1.0)
            if success:
                return service_type

    # Fallback: guess by port number
    for service_type, ports in SERVICE_PORTS.items():
        if port in ports:
            return service_type

    return None


def get_service_capabilities(endpoint: ServiceEndpoint) -> List[str]:
    """Detect capabilities of a service."""
    capabilities = []

    if endpoint.service_type == "comfyui":
        # Try to get object_info for node types
        success, data, _ = http_get(f"{endpoint.url}/object_info", timeout=3.0)
        if success and data:
            node_types = list(data.keys()) if isinstance(data, dict) else []
            node_str = " ".join(node_types).lower()

            if "flux" in node_str:
                capabilities.append("flux")
            if "sdxl" in node_str or "xl" in node_str:
                capabilities.append("sdxl")
            if "sd3" in node_str:
                capabilities.append("sd3")
            if "video" in node_str or "animatediff" in node_str or "wan" in node_str:
                capabilities.append("video")
            if "image" in node_str:
                capabilities.append("image")

    elif endpoint.service_type == "kobold":
        # Get model info
        success, data, _ = http_get(f"{endpoint.url}/api/v1/info", timeout=2.0)
        if success and data:
            model = data.get("model", "").lower()
            if "llama" in model:
                capabilities.append("llama")
            if "mistral" in model:
                capabilities.append("mistral")
            if "qwen" in model:
                capabilities.append("qwen")
            capabilities.append("llm")

    elif endpoint.service_type == "ollama":
        # Get available models
        success, data, _ = http_get(f"{endpoint.url}/api/tags", timeout=2.0)
        if success and data:
            models = data.get("models", [])
            for model in models:
                name = model.get("name", "").lower()
                if "llama" in name:
                    capabilities.append("llama")
                if "mistral" in name:
                    capabilities.append("mistral")
                if "qwen" in name:
                    capabilities.append("qwen")
            if models:
                capabilities.append("llm")

    return list(set(capabilities))  # Deduplicate


def health_check(endpoint: ServiceEndpoint) -> ServiceEndpoint:
    """Check health of a service endpoint and update its status."""
    endpoint.last_check = time.time()

    endpoints_to_try = HEALTH_ENDPOINTS.get(endpoint.service_type, ["/"])

    for path in endpoints_to_try:
        url = f"{endpoint.url}{path}"
        success, data, response_time = http_get(url, timeout=2.0)

        if success:
            endpoint.status = "online"
            endpoint.response_time_ms = response_time

            # Calculate health score based on response time
            if response_time < 100:
                endpoint.health_score = 1.0
            elif response_time < 500:
                endpoint.health_score = 0.8
            elif response_time < 1000:
                endpoint.health_score = 0.5
            else:
                endpoint.health_score = 0.3

            # Update capabilities
            if not endpoint.capabilities:
                endpoint.capabilities = get_service_capabilities(endpoint)

            return endpoint

    endpoint.status = "offline"
    endpoint.health_score = 0.0
    return endpoint


def parse_cidr(cidr: str) -> List[str]:
    """Parse CIDR notation to list of IPs (supports /24 only for simplicity)."""
    if "/" not in cidr:
        return [cidr]

    base, prefix = cidr.split("/")
    prefix = int(prefix)

    if prefix != 24:
        # Only support /24 for simplicity
        return [base]

    parts = base.split(".")
    if len(parts) != 4:
        return [base]

    # Generate all IPs in /24 range
    return [f"{parts[0]}.{parts[1]}.{parts[2]}.{i}" for i in range(1, 255)]


class ServiceDiscovery:
    """
    Discovers and manages AI services on local network.

    Usage:
        discovery = ServiceDiscovery()
        endpoints = discovery.scan()
        for ep in endpoints:
            print(f"{ep.service_type} at {ep.url} - {ep.status}")
    """

    def __init__(self, timeout_ms: int = 100, max_workers: int = 50):
        """
        Initialize service discovery.

        Args:
            timeout_ms: Timeout for port scans in milliseconds
            max_workers: Maximum parallel scan threads
        """
        self.timeout = timeout_ms / 1000.0
        self.max_workers = max_workers
        self.endpoints: List[ServiceEndpoint] = []
        self.last_scan: float = 0.0

    def _scan_host(self, host: str) -> List[ServiceEndpoint]:
        """Scan a single host for all known service ports."""
        found = []

        for service_type, ports in SERVICE_PORTS.items():
            for port in ports:
                if check_port(host, port, self.timeout):
                    endpoint = ServiceEndpoint(
                        host=host,
                        port=port,
                        service_type=service_type,
                        url=f"http://{host}:{port}",
                        status="unknown"
                    )
                    found.append(endpoint)

        return found

    def scan_localhost(self) -> List[ServiceEndpoint]:
        """Scan localhost for services."""
        return self._scan_host("127.0.0.1")

    def scan_network(self, cidr: str = "192.168.1.0/24", skip_localhost: bool = True) -> List[ServiceEndpoint]:
        """
        Scan network range for services.

        Args:
            cidr: Network range in CIDR notation
            skip_localhost: Skip 127.0.0.1 (usually scanned separately)
        """
        hosts = parse_cidr(cidr)
        all_found = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for host in hosts:
                if skip_localhost and host == "127.0.0.1":
                    continue
                futures[executor.submit(self._scan_host, host)] = host

            for future in as_completed(futures):
                try:
                    found = future.result()
                    all_found.extend(found)
                except:
                    pass

        return all_found

    def scan(self, network_range: Optional[str] = None, include_localhost: bool = True) -> List[ServiceEndpoint]:
        """
        Full scan of localhost and optionally network.

        Args:
            network_range: CIDR range to scan (None = localhost only)
            include_localhost: Include localhost in scan
        """
        self.endpoints = []
        self.last_scan = time.time()

        # Scan localhost first
        if include_localhost:
            self.endpoints.extend(self.scan_localhost())

        # Scan network if specified
        if network_range:
            self.endpoints.extend(self.scan_network(network_range))

        # Health check all endpoints
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(health_check, ep) for ep in self.endpoints]
            for future in as_completed(futures):
                try:
                    future.result()
                except:
                    pass

        return self.endpoints

    def get_online(self, service_type: Optional[str] = None) -> List[ServiceEndpoint]:
        """Get online endpoints, optionally filtered by type."""
        online = [ep for ep in self.endpoints if ep.is_online]
        if service_type:
            online = [ep for ep in online if ep.service_type == service_type]
        return online

    def get_best(self, service_type: str) -> Optional[ServiceEndpoint]:
        """Get the best (healthiest) endpoint of a given type."""
        online = self.get_online(service_type)
        if not online:
            return None
        return max(online, key=lambda ep: ep.health_score)

    def generate_litellm_config(self) -> str:
        """Generate LiteLLM config YAML from discovered endpoints."""
        kobold = self.get_online("kobold")
        ollama = self.get_online("ollama")

        lines = [
            "# LiteLLM Configuration",
            "# Generated by Performance Lab",
            f"# Scan time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "model_list:"
        ]

        for ep in kobold:
            lines.extend([
                f"  - model_name: local-llm",
                f"    litellm_params:",
                f"      model: openai/kobold",
                f"      api_base: {ep.url}/v1",
                f"      rpm: 10"
            ])

        for ep in ollama:
            lines.extend([
                f"  - model_name: local-llm",
                f"    litellm_params:",
                f"      model: ollama/llama3.2",
                f"      api_base: {ep.url}",
                f"      rpm: 10"
            ])

        if kobold or ollama:
            lines.extend([
                "",
                "router_settings:",
                "  routing_strategy: least-busy",
                "  num_retries: 2",
                "  allowed_fails: 3",
                "  cooldown_time: 60"
            ])

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export discovered endpoints as JSON."""
        data = []
        for ep in self.endpoints:
            data.append({
                "host": ep.host,
                "port": ep.port,
                "service_type": ep.service_type,
                "url": ep.url,
                "status": ep.status,
                "health_score": ep.health_score,
                "response_time_ms": ep.response_time_ms,
                "capabilities": ep.capabilities,
            })
        return json.dumps(data, indent=2)

    def get_status_report(self) -> str:
        """Generate human-readable status report."""
        lines = ["═══ Service Discovery Report ═══", ""]

        by_type = {}
        for ep in self.endpoints:
            if ep.service_type not in by_type:
                by_type[ep.service_type] = []
            by_type[ep.service_type].append(ep)

        for service_type, endpoints in by_type.items():
            online = [ep for ep in endpoints if ep.is_online]
            lines.append(f"{service_type.upper()}: {len(online)}/{len(endpoints)} online")

            for ep in endpoints:
                status_icon = "🟢" if ep.is_online else "🔴"
                health = f"({ep.health_score:.0%})" if ep.is_online else ""
                caps = f" [{', '.join(ep.capabilities)}]" if ep.capabilities else ""
                lines.append(f"  {status_icon} {ep.url} {health}{caps}")

            lines.append("")

        total_online = len([ep for ep in self.endpoints if ep.is_online])
        lines.append(f"Total: {total_online}/{len(self.endpoints)} services online")

        return "\n".join(lines)


# Global discovery instance
_discovery: Optional[ServiceDiscovery] = None


def get_discovery() -> ServiceDiscovery:
    """Get the global service discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = ServiceDiscovery()
    return _discovery


def quick_scan(network_range: Optional[str] = None) -> List[ServiceEndpoint]:
    """Convenience function for quick scan."""
    discovery = get_discovery()
    return discovery.scan(network_range)
