"""
Tests for workflow_utils.py

Run with: pytest tests/test_workflow_utils.py -v
"""

import pytest
import json
from unittest.mock import Mock, patch

# Import module under test
import sys
sys.path.insert(0, '..')
from workflow_utils import (
    WorkflowFingerprinter,
    WorkflowFingerprint,
    ModelFamily,
    WorkflowBeautifier,
    BeautifyMode,
    WorkflowDiff,
    is_safe_to_overwrite,
    suggest_filename,
    is_distributed_workflow,
    get_network_nodes_summary,
    extract_endpoints_from_workflow,
    NETWORK_NODE_TYPES,
)


class TestWorkflowFingerprinter:
    """Tests for WorkflowFingerprinter class."""

    @pytest.fixture
    def fingerprinter(self):
        return WorkflowFingerprinter()

    @pytest.fixture
    def sample_workflow(self):
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
                    "widgets_values": ["a beautiful landscape"]
                }
            ],
            "links": [[1, 1, 0, 2, 0, "MODEL"]]
        }

    def test_fingerprint_workflow(self, fingerprinter, sample_workflow):
        fp = fingerprinter.fingerprint(sample_workflow)
        assert isinstance(fp, WorkflowFingerprint)
        assert fp.node_count == 3
        assert fp.hash is not None

    def test_fingerprint_empty_workflow(self, fingerprinter):
        fp = fingerprinter.fingerprint({})
        assert fp.node_count == 0

    def test_fingerprint_detects_model_family(self, fingerprinter, sample_workflow):
        fp = fingerprinter.fingerprint(sample_workflow)
        # Should detect SD15 from the checkpoint name
        assert fp.model_family is not None

    def test_same_workflow_same_hash(self, fingerprinter, sample_workflow):
        fp1 = fingerprinter.fingerprint(sample_workflow)
        fp2 = fingerprinter.fingerprint(sample_workflow)
        assert fp1.hash == fp2.hash

    def test_different_workflow_different_hash(self, fingerprinter, sample_workflow):
        fp1 = fingerprinter.fingerprint(sample_workflow)

        modified = sample_workflow.copy()
        modified["nodes"] = sample_workflow["nodes"] + [{"id": 4, "type": "VAEDecode"}]
        fp2 = fingerprinter.fingerprint(modified)

        assert fp1.hash != fp2.hash


class TestWorkflowBeautifier:
    """Tests for WorkflowBeautifier class."""

    @pytest.fixture
    def beautifier(self):
        return WorkflowBeautifier()

    @pytest.fixture
    def messy_workflow(self):
        return {
            "nodes": [
                {"id": 1, "type": "KSampler", "pos": [100, 100]},
                {"id": 2, "type": "CheckpointLoaderSimple", "pos": [500, 500]},
                {"id": 3, "type": "VAEDecode", "pos": [200, 200]}
            ],
            "links": [
                [1, 2, 0, 1, 0, "MODEL"],
                [2, 1, 0, 3, 0, "LATENT"]
            ]
        }

    def test_beautify_workflow(self, beautifier, messy_workflow):
        result = beautifier.beautify(messy_workflow)
        assert "nodes" in result
        assert len(result["nodes"]) == 3

    def test_beautify_modes_exist(self):
        assert BeautifyMode.COMPACT is not None
        assert BeautifyMode.ORGANIZED is not None
        assert BeautifyMode.EXPANDED is not None


class TestWorkflowDiff:
    """Tests for WorkflowDiff class."""

    @pytest.fixture
    def workflow_a(self):
        return {
            "nodes": [
                {"id": 1, "type": "KSampler", "widgets_values": [20, 7.5]},
                {"id": 2, "type": "VAEDecode"}
            ]
        }

    @pytest.fixture
    def workflow_b(self):
        return {
            "nodes": [
                {"id": 1, "type": "KSampler", "widgets_values": [30, 8.0]},
                {"id": 2, "type": "VAEDecode"},
                {"id": 3, "type": "SaveImage"}
            ]
        }

    def test_diff_workflows(self, workflow_a, workflow_b):
        diff = WorkflowDiff.diff(workflow_a, workflow_b)
        assert isinstance(diff, dict)
        assert "added_nodes" in diff or "modified_nodes" in diff or "removed_nodes" in diff

    def test_summarize_diff(self, workflow_a, workflow_b):
        diff = WorkflowDiff.diff(workflow_a, workflow_b)
        summary = WorkflowDiff.summarize(diff)
        assert isinstance(summary, str)

    def test_identical_workflows(self, workflow_a):
        diff = WorkflowDiff.diff(workflow_a, workflow_a)
        summary = WorkflowDiff.summarize(diff)
        # Should indicate no changes or minimal changes
        assert isinstance(summary, str)


class TestModelFamily:
    """Tests for ModelFamily enum."""

    def test_model_families_exist(self):
        assert ModelFamily.SD15 is not None
        assert ModelFamily.SDXL is not None
        assert ModelFamily.FLUX is not None
        assert ModelFamily.VIDEO is not None

    def test_model_family_values(self):
        assert ModelFamily.SD15.value == "sd15"
        assert ModelFamily.SDXL.value == "sdxl"
        assert ModelFamily.FLUX.value == "flux"


class TestNetworkNodeDetection:
    """Tests for network node detection functions."""

    @pytest.fixture
    def distributed_workflow(self):
        return {
            "nodes": [
                {"id": "1", "type": "KoboldLLM", "widgets_values": ["http://192.168.1.100:5001"]},
                {"id": "2", "type": "RemoteComfyUI", "widgets_values": ["http://192.168.1.101:8188"]},
                {"id": "3", "type": "KSampler"}
            ]
        }

    @pytest.fixture
    def local_workflow(self):
        return {
            "nodes": [
                {"id": "1", "type": "KSampler"},
                {"id": "2", "type": "VAEDecode"},
                {"id": "3", "type": "SaveImage"}
            ]
        }

    def test_network_node_types_defined(self):
        assert "KoboldLLM" in NETWORK_NODE_TYPES
        assert "RemoteComfyUI" in NETWORK_NODE_TYPES
        assert "LocalGenerator" in NETWORK_NODE_TYPES

    def test_is_distributed_workflow_true(self, distributed_workflow):
        assert is_distributed_workflow(distributed_workflow) is True

    def test_is_distributed_workflow_false(self, local_workflow):
        assert is_distributed_workflow(local_workflow) is False

    def test_get_network_nodes_summary(self, distributed_workflow):
        summary = get_network_nodes_summary(distributed_workflow)
        assert isinstance(summary, str)
        assert "KoboldLLM" in summary or "network" in summary.lower()

    def test_extract_endpoints(self, distributed_workflow):
        endpoints = extract_endpoints_from_workflow(distributed_workflow)
        assert isinstance(endpoints, list)
        # Should find at least one endpoint
        assert len(endpoints) >= 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_safe_to_overwrite_same_file(self):
        # Same content should be safe
        result = is_safe_to_overwrite('{"test": 1}', '{"test": 1}')
        assert result is True

    def test_suggest_filename_basic(self):
        workflow = {"nodes": [{"type": "KSampler"}]}
        filename = suggest_filename(workflow)
        assert isinstance(filename, str)
        assert filename.endswith(".json")

    def test_suggest_filename_with_model(self):
        workflow = {
            "nodes": [
                {"type": "CheckpointLoaderSimple", "widgets_values": ["sdxl_base.safetensors"]}
            ]
        }
        filename = suggest_filename(workflow)
        assert isinstance(filename, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
