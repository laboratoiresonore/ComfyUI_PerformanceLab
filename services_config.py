"""
Services Configuration Module for Performance Lab

Manages service endpoint configurations from YAML/JSON files for easier
management of multi-machine setups.

Example configuration file (services.yaml):

services:
  - name: "Main GPU Server"
    endpoint: "http://192.168.1.100:8188"
    type: "comfyui"
    specs:
      gpu: "RTX 4090"
      vram_gb: 24
      cpu: "Ryzen 9 5950X"
      ram_gb: 64
    description: "Primary SD generation server"

  - name: "LLM Server"
    endpoint: "http://192.168.1.101:5001"
    type: "koboldcpp"
    specs:
      gpu: "RTX 3090"
      vram_gb: 24
      model: "Mixtral-8x7B"
      context_size: 32768
    description: "Text generation with Kobold"

clusters:
  production:
    description: "Production inference cluster"
    services:
      - "Main GPU Server"
      - "LLM Server"
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger("performance_lab.services_config")

# Try to import YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.debug("PyYAML not available, using JSON-only mode")


@dataclass
class ServiceSpecs:
    """Hardware and configuration specs for a service."""
    gpu: str = ""
    vram_gb: int = 0
    cpu: str = ""
    ram_gb: int = 0
    model: str = ""
    context_size: int = 0
    quantization: str = ""
    batch_size: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ServiceSpecs':
        """Create ServiceSpecs from dictionary."""
        known_fields = {'gpu', 'vram_gb', 'cpu', 'ram_gb', 'model',
                        'context_size', 'quantization', 'batch_size'}
        known = {k: v for k, v in data.items() if k in known_fields}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        return cls(**known, extra=extra)


@dataclass
class ServiceConfig:
    """Configuration for a single service endpoint."""
    name: str
    endpoint: str
    type: str
    description: str = ""
    docs_url: str = ""
    specs: ServiceSpecs = field(default_factory=ServiceSpecs)
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ServiceConfig':
        """Create ServiceConfig from dictionary."""
        specs_data = data.pop('specs', {})
        specs = ServiceSpecs.from_dict(specs_data) if specs_data else ServiceSpecs()
        return cls(
            name=data.get('name', ''),
            endpoint=data.get('endpoint', ''),
            type=data.get('type', 'custom'),
            description=data.get('description', ''),
            docs_url=data.get('docs_url', ''),
            specs=specs,
            enabled=data.get('enabled', True),
            tags=data.get('tags', [])
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'endpoint': self.endpoint,
            'type': self.type,
            'description': self.description,
            'docs_url': self.docs_url,
            'specs': asdict(self.specs),
            'enabled': self.enabled,
            'tags': self.tags
        }


@dataclass
class ClusterConfig:
    """Configuration for a cluster of services."""
    name: str
    description: str = ""
    service_names: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: Dict) -> 'ClusterConfig':
        """Create ClusterConfig from dictionary."""
        return cls(
            name=name,
            description=data.get('description', ''),
            service_names=data.get('services', [])
        )


class ServicesConfigManager:
    """
    Manages service configurations from files.

    Supports both YAML and JSON configuration files.
    """

    DEFAULT_CONFIG_PATHS = [
        Path.home() / ".performance_lab" / "services.yaml",
        Path.home() / ".performance_lab" / "services.json",
        Path("services.yaml"),
        Path("services.json"),
    ]

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Optional path to configuration file.
                         If not provided, searches default locations.
        """
        self.config_path = config_path
        self.services: Dict[str, ServiceConfig] = {}
        self.clusters: Dict[str, ClusterConfig] = {}
        self._loaded = False

        if config_path:
            self.load(config_path)
        else:
            self._try_load_defaults()

    def _try_load_defaults(self):
        """Try to load from default configuration paths."""
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                try:
                    self.load(path)
                    logger.info(f"Loaded services configuration from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

    def load(self, path: Path) -> bool:
        """
        Load configuration from a file.

        Args:
            path: Path to YAML or JSON configuration file

        Returns:
            True if successful, False otherwise
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return False

        try:
            content = path.read_text()

            if path.suffix in ('.yaml', '.yml'):
                if not YAML_AVAILABLE:
                    logger.error("PyYAML not installed. Install with: pip install pyyaml")
                    return False
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)

            self._parse_config(data)
            self.config_path = path
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            return False

    def _parse_config(self, data: Dict):
        """Parse configuration data."""
        # Parse services
        services_data = data.get('services', [])
        for svc_data in services_data:
            service = ServiceConfig.from_dict(svc_data)
            self.services[service.name] = service

        # Parse clusters
        clusters_data = data.get('clusters', {})
        for name, cluster_data in clusters_data.items():
            cluster = ClusterConfig.from_dict(name, cluster_data)
            self.clusters[name] = cluster

    def save(self, path: Optional[Path] = None):
        """
        Save configuration to a file.

        Args:
            path: Path to save to. Uses original path if not specified.
        """
        path = Path(path) if path else self.config_path
        if not path:
            path = Path.home() / ".performance_lab" / "services.json"
            path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'services': [svc.to_dict() for svc in self.services.values()],
            'clusters': {
                name: {
                    'description': cluster.description,
                    'services': cluster.service_names
                }
                for name, cluster in self.clusters.items()
            }
        }

        if path.suffix in ('.yaml', '.yml') and YAML_AVAILABLE:
            content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        else:
            content = json.dumps(data, indent=2)

        path.write_text(content)
        logger.info(f"Saved services configuration to {path}")

    def add_service(self, service: ServiceConfig):
        """Add or update a service configuration."""
        self.services[service.name] = service

    def remove_service(self, name: str) -> bool:
        """Remove a service by name."""
        if name in self.services:
            del self.services[name]
            return True
        return False

    def get_service(self, name: str) -> Optional[ServiceConfig]:
        """Get a service by name."""
        return self.services.get(name)

    def get_service_by_endpoint(self, endpoint: str) -> Optional[ServiceConfig]:
        """Find a service by its endpoint URL."""
        endpoint = endpoint.rstrip('/')
        for service in self.services.values():
            if service.endpoint.rstrip('/') == endpoint:
                return service
        return None

    def get_services_by_type(self, service_type: str) -> List[ServiceConfig]:
        """Get all services of a specific type."""
        return [
            svc for svc in self.services.values()
            if svc.type == service_type and svc.enabled
        ]

    def get_services_by_tag(self, tag: str) -> List[ServiceConfig]:
        """Get all services with a specific tag."""
        return [
            svc for svc in self.services.values()
            if tag in svc.tags and svc.enabled
        ]

    def get_cluster(self, name: str) -> Optional[ClusterConfig]:
        """Get a cluster by name."""
        return self.clusters.get(name)

    def get_cluster_services(self, cluster_name: str) -> List[ServiceConfig]:
        """Get all services in a cluster."""
        cluster = self.clusters.get(cluster_name)
        if not cluster:
            return []
        return [
            self.services[name]
            for name in cluster.service_names
            if name in self.services
        ]

    def get_all_endpoints(self) -> List[str]:
        """Get all configured endpoints."""
        return [svc.endpoint for svc in self.services.values() if svc.enabled]

    def to_machine_profiles(self) -> List['MachineProfile']:
        """
        Convert service configs to MachineProfile objects.

        Returns:
            List of MachineProfile objects for use with DistributedWorkflowAnalyzer
        """
        from distributed_optimizer import MachineProfile

        profiles = []
        for svc in self.services.values():
            if not svc.enabled:
                continue

            profile = MachineProfile(
                name=svc.name,
                endpoint=svc.endpoint,
                gpu_model=svc.specs.gpu,
                vram_gb=svc.specs.vram_gb,
                cpu_model=svc.specs.cpu,
                ram_gb=svc.specs.ram_gb,
                services=[svc.type],
                notes=svc.description
            )
            profiles.append(profile)

        return profiles

    @property
    def is_loaded(self) -> bool:
        """Check if configuration has been loaded."""
        return self._loaded

    def __len__(self) -> int:
        return len(self.services)

    def __iter__(self):
        return iter(self.services.values())


def create_example_config(path: Optional[Path] = None) -> Path:
    """
    Create an example configuration file.

    Args:
        path: Path to create the file at

    Returns:
        Path to created file
    """
    if path is None:
        path = Path.home() / ".performance_lab" / "services.example.yaml"

    path.parent.mkdir(parents=True, exist_ok=True)

    example = """# Performance Lab Services Configuration
# Copy this file to services.yaml and customize for your setup

services:
  # ComfyUI instance for image/video generation
  - name: "Main ComfyUI"
    endpoint: "http://192.168.1.100:8188"
    type: "comfyui"
    description: "Primary SD/SDXL/Flux generation server"
    specs:
      gpu: "RTX 4090"
      vram_gb: 24
      cpu: "Ryzen 9 5950X"
      ram_gb: 64
    tags:
      - production
      - gpu

  # KoboldCpp for LLM inference
  - name: "LLM Server"
    endpoint: "http://192.168.1.101:5001"
    type: "koboldcpp"
    description: "Text generation with Mixtral"
    docs_url: "https://github.com/LostRuins/koboldcpp"
    specs:
      gpu: "RTX 3090"
      vram_gb: 24
      model: "Mixtral-8x7B-Instruct"
      context_size: 32768
      quantization: "Q4_K_M"
    tags:
      - production
      - llm

  # Whisper for speech-to-text
  - name: "STT Server"
    endpoint: "http://192.168.1.102:9000"
    type: "faster_whisper"
    description: "Speech transcription with Faster Whisper"
    specs:
      gpu: "RTX 3080"
      vram_gb: 10
      model: "large-v3"
    tags:
      - production
      - audio

  # TTS service
  - name: "TTS Server"
    endpoint: "http://192.168.1.102:5500"
    type: "xtts"
    description: "Voice synthesis with XTTS v2"
    specs:
      gpu: "RTX 3080"
      vram_gb: 10
    tags:
      - production
      - audio

# Clusters group services together
clusters:
  production:
    description: "Full production inference pipeline"
    services:
      - "Main ComfyUI"
      - "LLM Server"
      - "STT Server"
      - "TTS Server"

  image_gen:
    description: "Image generation only"
    services:
      - "Main ComfyUI"

  audio:
    description: "Audio processing services"
    services:
      - "STT Server"
      - "TTS Server"
"""

    path.write_text(example)
    logger.info(f"Created example configuration at {path}")
    return path


# Convenience function for quick access
_global_config: Optional[ServicesConfigManager] = None


def get_services_config() -> ServicesConfigManager:
    """Get the global services configuration manager."""
    global _global_config
    if _global_config is None:
        _global_config = ServicesConfigManager()
    return _global_config


def load_services_config(path: Path) -> ServicesConfigManager:
    """Load services configuration from a specific path."""
    global _global_config
    _global_config = ServicesConfigManager(path)
    return _global_config
