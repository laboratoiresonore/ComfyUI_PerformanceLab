"""
Tests for distributed_optimizer.py

Run with: pytest tests/test_distributed_optimizer.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

# Import module under test
import sys
sys.path.insert(0, '..')
from distributed_optimizer import (
    validate_endpoint,
    get_timeout_for_service,
    SERVICE_TIMEOUTS,
    ServiceCategory,
    NetworkNodeInfo,
    MachineProfile,
    DistributedWorkflowAnalyzer,
)


class TestValidateEndpoint:
    """Tests for endpoint validation."""

    def test_valid_http_endpoint(self):
        is_valid, error = validate_endpoint("http://localhost:8188")
        assert is_valid is True
        assert error == ""

    def test_valid_https_endpoint(self):
        is_valid, error = validate_endpoint("https://192.168.1.100:8188")
        assert is_valid is True
        assert error == ""

    def test_valid_endpoint_with_path(self):
        is_valid, error = validate_endpoint("http://localhost:8188/api/v1")
        assert is_valid is True
        assert error == ""

    def test_empty_endpoint(self):
        is_valid, error = validate_endpoint("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_invalid_scheme(self):
        is_valid, error = validate_endpoint("ftp://localhost:8188")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_missing_host(self):
        is_valid, error = validate_endpoint("http://")
        assert is_valid is False
        assert "host" in error.lower()

    def test_no_scheme(self):
        is_valid, error = validate_endpoint("localhost:8188")
        assert is_valid is False


class TestGetTimeoutForService:
    """Tests for service timeout configuration."""

    def test_llm_timeout(self):
        timeout = get_timeout_for_service("llm")
        assert timeout == SERVICE_TIMEOUTS["llm"]
        assert timeout == 120

    def test_image_generation_timeout(self):
        timeout = get_timeout_for_service("image_generation")
        assert timeout == SERVICE_TIMEOUTS["image_generation"]
        assert timeout == 300

    def test_video_generation_timeout(self):
        timeout = get_timeout_for_service("video_generation")
        assert timeout == 600

    def test_stt_alias(self):
        timeout = get_timeout_for_service("stt")
        assert timeout == SERVICE_TIMEOUTS["speech_to_text"]

    def test_tts_alias(self):
        timeout = get_timeout_for_service("tts")
        assert timeout == SERVICE_TIMEOUTS["text_to_speech"]

    def test_unknown_service_returns_default(self):
        timeout = get_timeout_for_service("unknown_service")
        assert timeout == SERVICE_TIMEOUTS["default"]


class TestNetworkNodeInfo:
    """Tests for NetworkNodeInfo dataclass."""

    def test_create_network_node_info(self):
        node = NetworkNodeInfo(
            node_id="123",
            node_type="KoboldLLM",
            service_type="llm",
            endpoint="http://localhost:5001",
            category=ServiceCategory.LLM
        )
        assert node.node_id == "123"
        assert node.node_type == "KoboldLLM"
        assert node.endpoint == "http://localhost:5001"

    def test_network_node_with_machine_info(self):
        node = NetworkNodeInfo(
            node_id="456",
            node_type="RemoteComfyUI",
            service_type="image_generation",
            endpoint="http://192.168.1.100:8188",
            category=ServiceCategory.IMAGE_GENERATION,
            machine_name="GPU Server 1",
            machine_specs="RTX 4090 24GB"
        )
        assert node.machine_name == "GPU Server 1"
        assert node.machine_specs == "RTX 4090 24GB"


class TestMachineProfile:
    """Tests for MachineProfile dataclass."""

    def test_create_machine_profile(self):
        profile = MachineProfile(
            name="GPU Server",
            endpoint="http://192.168.1.100:8188",
            gpu_model="RTX 4090",
            vram_gb=24,
            cpu_model="Ryzen 9 5950X",
            ram_gb=64
        )
        assert profile.name == "GPU Server"
        assert profile.vram_gb == 24
        assert profile.ram_gb == 64

    def test_machine_profile_to_dict(self):
        profile = MachineProfile(
            name="Test",
            endpoint="http://localhost:8188"
        )
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "Test"


class TestDistributedWorkflowAnalyzer:
    """Tests for DistributedWorkflowAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return DistributedWorkflowAnalyzer(timeout=5)

    @pytest.fixture
    def sample_workflow(self):
        return {
            "nodes": [
                {
                    "id": "1",
                    "type": "KoboldLLM",
                    "widgets_values": ["http://192.168.1.100:5001", "prompt text", 512]
                },
                {
                    "id": "2",
                    "type": "RemoteComfyUI",
                    "widgets_values": ["http://192.168.1.101:8188", "{}", 300]
                },
                {
                    "id": "3",
                    "type": "KSampler",
                    "widgets_values": [20, 7.5, "euler"]
                }
            ],
            "links": [
                [1, 1, 0, 2, 0, "STRING"]
            ]
        }

    def test_detect_network_nodes(self, analyzer, sample_workflow):
        nodes = analyzer.detect_network_nodes(sample_workflow)
        # Should find KoboldLLM and RemoteComfyUI, but not KSampler
        network_types = [n.node_type for n in nodes]
        assert "KoboldLLM" in network_types
        assert "RemoteComfyUI" in network_types
        assert "KSampler" not in network_types

    def test_detect_empty_workflow(self, analyzer):
        nodes = analyzer.detect_network_nodes({})
        assert len(nodes) == 0

    def test_detect_workflow_no_network_nodes(self, analyzer):
        workflow = {
            "nodes": [
                {"id": "1", "type": "KSampler"},
                {"id": "2", "type": "VAEDecode"}
            ]
        }
        nodes = analyzer.detect_network_nodes(workflow)
        assert len(nodes) == 0

    def test_register_machine(self, analyzer):
        profile = MachineProfile(
            name="Test Machine",
            endpoint="http://localhost:8188"
        )
        analyzer.register_machine(profile)
        assert "http://localhost:8188" in analyzer.machines
        assert analyzer.machines["http://localhost:8188"].name == "Test Machine"

    @patch('distributed_optimizer.requests.Session.get')
    def test_check_endpoint_health_success(self, mock_get, analyzer):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        is_healthy, latency = analyzer.check_endpoint_health("http://localhost:8188", "comfyui")
        assert is_healthy is True
        assert latency > 0

    @patch('distributed_optimizer.requests.Session.get')
    def test_check_endpoint_health_failure(self, mock_get, analyzer):
        mock_get.side_effect = Exception("Connection refused")

        is_healthy, latency = analyzer.check_endpoint_health("http://localhost:8188", "comfyui")
        assert is_healthy is False
        assert latency == -1.0

    def test_check_invalid_endpoint(self, analyzer):
        is_healthy, latency = analyzer.check_endpoint_health("not-a-valid-url", "comfyui")
        assert is_healthy is False
        assert latency == -1.0

    def test_build_dependency_graph(self, analyzer, sample_workflow):
        graph = analyzer.build_dependency_graph(sample_workflow)
        assert isinstance(graph, dict)

    def test_get_unique_endpoints(self, analyzer, sample_workflow):
        endpoints = analyzer.get_unique_endpoints(sample_workflow)
        assert isinstance(endpoints, dict)


class TestServiceCategory:
    """Tests for ServiceCategory enum."""

    def test_service_categories_exist(self):
        assert ServiceCategory.LLM is not None
        assert ServiceCategory.IMAGE_GENERATION is not None
        assert ServiceCategory.VIDEO_GENERATION is not None
        assert ServiceCategory.SPEECH_TO_TEXT is not None
        assert ServiceCategory.TEXT_TO_SPEECH is not None
        assert ServiceCategory.EMBEDDINGS is not None

    def test_category_values(self):
        assert ServiceCategory.LLM.value == "llm"
        assert ServiceCategory.IMAGE_GENERATION.value == "image_generation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
