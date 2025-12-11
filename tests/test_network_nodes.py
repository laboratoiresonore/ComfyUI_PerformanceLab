"""
Tests for ComfyUI Network Service Nodes

Run with: pytest tests/test_network_nodes.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np

# Import modules under test
import sys
sys.path.insert(0, '../custom_nodes/ComfyUI_NetworkServices/nodes')


class TestKoboldLLM:
    """Tests for KoboldLLM node."""

    @pytest.fixture
    def kobold_node(self):
        from kobold_llm import KoboldLLM
        return KoboldLLM()

    def test_input_types(self, kobold_node):
        input_types = kobold_node.INPUT_TYPES()
        assert "required" in input_types
        assert "prompt" in input_types["required"]
        assert "endpoint" in input_types["required"]

    def test_return_types(self, kobold_node):
        assert hasattr(kobold_node, 'RETURN_TYPES')
        assert "STRING" in kobold_node.RETURN_TYPES

    def test_function_name(self, kobold_node):
        assert kobold_node.FUNCTION == "generate"

    def test_category(self, kobold_node):
        assert "NetworkServices" in kobold_node.CATEGORY

    def test_network_service_flag(self, kobold_node):
        assert kobold_node.NETWORK_SERVICE is True
        assert kobold_node.SERVICE_TYPE == "llm"

    @patch('kobold_llm.requests.post')
    def test_generate_success(self, mock_post, kobold_node):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"text": "Generated text response"}]
        }
        mock_post.return_value = mock_response

        result = kobold_node.generate(
            prompt="Test prompt",
            endpoint="http://localhost:5001",
            max_tokens=100,
            temperature=0.7
        )

        assert result[0] == "Generated text response"
        assert result[1] > 0  # execution time

    @patch('kobold_llm.requests.post')
    def test_generate_connection_error(self, mock_post, kobold_node):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(RuntimeError):
            kobold_node.generate(
                prompt="Test",
                endpoint="http://localhost:5001",
                max_tokens=100,
                temperature=0.7
            )


class TestRemoteComfyUI:
    """Tests for RemoteComfyUI node."""

    @pytest.fixture
    def remote_node(self):
        from remote_comfyui import RemoteComfyUI
        return RemoteComfyUI()

    def test_input_types(self, remote_node):
        input_types = remote_node.INPUT_TYPES()
        assert "required" in input_types
        assert "endpoint" in input_types["required"]
        assert "workflow_json" in input_types["required"]

    def test_return_types(self, remote_node):
        assert hasattr(remote_node, 'RETURN_TYPES')

    def test_function_name(self, remote_node):
        assert remote_node.FUNCTION == "execute"

    def test_network_service_flag(self, remote_node):
        assert remote_node.NETWORK_SERVICE is True
        assert remote_node.SERVICE_TYPE == "comfyui"


class TestNetworkAPI:
    """Tests for NetworkAPI (LocalGenerator) node."""

    @pytest.fixture
    def api_node(self):
        from network_api import NetworkAPI
        return NetworkAPI()

    def test_input_types(self, api_node):
        input_types = api_node.INPUT_TYPES()
        assert "required" in input_types
        assert "endpoint" in input_types["required"]
        assert "service_type" in input_types["required"]

    def test_service_types_available(self, api_node):
        input_types = api_node.INPUT_TYPES()
        service_types = input_types["required"]["service_type"][0]
        # Should have many service types
        assert len(service_types) > 50
        assert "comfyui" in service_types
        assert "ollama" in service_types
        assert "whisper" in service_types

    def test_default_paths_defined(self, api_node):
        assert hasattr(api_node, 'DEFAULT_PATHS')
        assert "comfyui" in api_node.DEFAULT_PATHS
        assert "ollama" in api_node.DEFAULT_PATHS

    def test_extract_json_path(self, api_node):
        data = {"results": [{"text": "hello"}]}
        result = api_node._extract_json_path(data, "results.0.text")
        assert result == "hello"

    def test_extract_json_path_missing(self, api_node):
        data = {"results": []}
        result = api_node._extract_json_path(data, "results.0.text")
        assert result is None

    @patch('network_api.requests.post')
    def test_call_api_json_response(self, mock_post, api_node):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"status": "ok"}
        mock_response.content = b'{"status": "ok"}'
        mock_post.return_value = mock_response

        result = api_node.call_api(
            endpoint="http://localhost:8188",
            service_type="comfyui",
            payload='{"test": true}'
        )

        assert "status" in result[0]  # response_text


class TestEndpointHealthCheck:
    """Tests for EndpointHealthCheck node."""

    @pytest.fixture
    def health_node(self):
        from health_check import EndpointHealthCheck
        return EndpointHealthCheck()

    def test_input_types(self, health_node):
        input_types = health_node.INPUT_TYPES()
        assert "required" in input_types
        assert "endpoint" in input_types["required"]

    def test_return_types(self, health_node):
        assert "BOOLEAN" in health_node.RETURN_TYPES
        assert "FLOAT" in health_node.RETURN_TYPES

    def test_health_paths_defined(self, health_node):
        assert hasattr(health_node, 'HEALTH_PATHS')
        assert "comfyui" in health_node.HEALTH_PATHS
        assert "ollama" in health_node.HEALTH_PATHS

    @patch('health_check.requests.get')
    def test_check_healthy_endpoint(self, mock_get, health_node):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        is_healthy, latency, system_info, report = health_node.check(
            endpoint="http://localhost:8188",
            service_type="comfyui",
            measure_latency=False,
            get_system_info=False
        )

        assert is_healthy is True

    @patch('health_check.requests.get')
    def test_check_unhealthy_endpoint(self, mock_get, health_node):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()

        is_healthy, latency, system_info, report = health_node.check(
            endpoint="http://localhost:8188",
            service_type="comfyui",
            measure_latency=False,
            get_system_info=False
        )

        assert is_healthy is False


class TestServiceDescriptions:
    """Tests for SERVICE_DESCRIPTIONS catalog."""

    def test_service_descriptions_import(self):
        from network_api import SERVICE_DESCRIPTIONS
        assert isinstance(SERVICE_DESCRIPTIONS, dict)

    def test_service_descriptions_structure(self):
        from network_api import SERVICE_DESCRIPTIONS

        for service_type, info in SERVICE_DESCRIPTIONS.items():
            assert "name" in info, f"Missing 'name' for {service_type}"
            assert "docs" in info, f"Missing 'docs' for {service_type}"
            assert "description" in info, f"Missing 'description' for {service_type}"
            assert "category" in info, f"Missing 'category' for {service_type}"

    def test_key_services_present(self):
        from network_api import SERVICE_DESCRIPTIONS

        key_services = [
            "comfyui", "automatic1111", "forge", "ollama",
            "koboldcpp", "whisper", "coqui_tts", "vllm"
        ]
        for service in key_services:
            assert service in SERVICE_DESCRIPTIONS, f"Missing key service: {service}"


class TestHelperFunctions:
    """Tests for helper functions in network nodes."""

    def test_get_kobold_model_info(self):
        from kobold_llm import get_kobold_model_info
        # Should handle connection errors gracefully
        result = get_kobold_model_info("http://invalid-endpoint:9999", timeout=1)
        assert result is None

    def test_get_remote_system_stats(self):
        from remote_comfyui import get_remote_system_stats
        # Should handle connection errors gracefully
        result = get_remote_system_stats("http://invalid-endpoint:9999", timeout=1)
        assert result is None

    def test_ping_host_invalid(self):
        from health_check import ping_host
        is_reachable, latency = ping_host("invalid.host.that.does.not.exist", timeout=1)
        assert is_reachable is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
