#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🔧 WORKFLOW UTILITIES - Fingerprinting & Beautification         ║
║         Smart change detection, organization, and visual improvements        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import copy
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW FINGERPRINTING - Detect key changes
# ═══════════════════════════════════════════════════════════════════════════════

class ModelFamily(Enum):
    """Model family classification for compatibility checking."""
    SD15 = "sd15"
    SD21 = "sd21"
    SDXL = "sdxl"
    SD3 = "sd3"
    FLUX = "flux"
    CASCADE = "cascade"
    PIXART = "pixart"
    HUNYUAN = "hunyuan"
    KOLORS = "kolors"
    VIDEO = "video"  # AnimateDiff, SVD, etc.
    UNKNOWN = "unknown"


@dataclass
class WorkflowFingerprint:
    """Unique fingerprint of a workflow's key characteristics."""
    model_family: ModelFamily
    model_names: List[str]
    resolution: Tuple[int, int]
    has_video: bool
    has_controlnet: bool
    has_ipadapter: bool
    has_upscaler: bool
    lora_count: int
    node_count: int
    hash_key: str  # Short hash for quick comparison

    def is_compatible_with(self, other: 'WorkflowFingerprint') -> Tuple[bool, List[str]]:
        """
        Check if this fingerprint is compatible with another.
        Returns (is_compatible, list_of_warnings).
        """
        warnings = []

        # Critical: Model family mismatch
        if self.model_family != other.model_family:
            if self.model_family != ModelFamily.UNKNOWN and other.model_family != ModelFamily.UNKNOWN:
                warnings.append(
                    f"MODEL FAMILY CHANGED: {other.model_family.value} -> {self.model_family.value}"
                )
                return False, warnings

        # Warning: Resolution changed significantly
        if self.resolution != other.resolution:
            old_pixels = other.resolution[0] * other.resolution[1]
            new_pixels = self.resolution[0] * self.resolution[1]
            if abs(new_pixels - old_pixels) / old_pixels > 0.5:
                warnings.append(
                    f"Resolution changed significantly: {other.resolution} -> {self.resolution}"
                )

        # Warning: Video mode changed
        if self.has_video != other.has_video:
            if self.has_video:
                warnings.append("Video generation ADDED")
            else:
                warnings.append("Video generation REMOVED")

        # Warning: ControlNet changed
        if self.has_controlnet != other.has_controlnet:
            warnings.append(f"ControlNet {'ADDED' if self.has_controlnet else 'REMOVED'}")

        # Warning: IPAdapter changed
        if self.has_ipadapter != other.has_ipadapter:
            warnings.append(f"IPAdapter {'ADDED' if self.has_ipadapter else 'REMOVED'}")

        # Info: LoRA count changed
        if abs(self.lora_count - other.lora_count) > 2:
            warnings.append(f"LoRA count changed: {other.lora_count} -> {self.lora_count}")

        return True, warnings

    def to_dict(self) -> Dict:
        return {
            "model_family": self.model_family.value,
            "model_names": self.model_names,
            "resolution": list(self.resolution),
            "has_video": self.has_video,
            "has_controlnet": self.has_controlnet,
            "has_ipadapter": self.has_ipadapter,
            "has_upscaler": self.has_upscaler,
            "lora_count": self.lora_count,
            "node_count": self.node_count,
            "hash_key": self.hash_key,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowFingerprint':
        return cls(
            model_family=ModelFamily(data.get("model_family", "unknown")),
            model_names=data.get("model_names", []),
            resolution=tuple(data.get("resolution", [512, 512])),
            has_video=data.get("has_video", False),
            has_controlnet=data.get("has_controlnet", False),
            has_ipadapter=data.get("has_ipadapter", False),
            has_upscaler=data.get("has_upscaler", False),
            lora_count=data.get("lora_count", 0),
            node_count=data.get("node_count", 0),
            hash_key=data.get("hash_key", ""),
        )


class WorkflowFingerprinter:
    """Generate fingerprints from workflows for change detection."""

    MODEL_PATTERNS = {
        ModelFamily.FLUX: [r"flux", r"FluxGuidance", r"UNETLoaderFlux"],
        ModelFamily.SD3: [r"sd3", r"stable.*diffusion.*3"],
        ModelFamily.SDXL: [r"sdxl", r"sd_xl", r"stable.*xl"],
        ModelFamily.CASCADE: [r"cascade", r"stage.*[bc]"],
        ModelFamily.PIXART: [r"pixart"],
        ModelFamily.HUNYUAN: [r"hunyuan"],
        ModelFamily.KOLORS: [r"kolors"],
        ModelFamily.SD21: [r"sd.*2\.1", r"768.*v"],
        ModelFamily.SD15: [r"sd.*1\.5", r"v1-5"],
        ModelFamily.VIDEO: [r"animatediff", r"svd", r"videodiff", r"wan"],
    }

    @classmethod
    def fingerprint(cls, workflow: Dict) -> WorkflowFingerprint:
        """Generate a fingerprint from a workflow."""
        nodes = workflow.get("nodes", [])

        # Detect model family
        model_family = ModelFamily.UNKNOWN
        model_names = []

        for node in nodes:
            node_type = node.get("type", "").lower()
            widgets = node.get("widgets_values", [])

            # Check node type
            for family, patterns in cls.MODEL_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, node_type, re.IGNORECASE):
                        if family == ModelFamily.VIDEO:
                            # Video is additive, not a model family
                            pass
                        elif model_family == ModelFamily.UNKNOWN:
                            model_family = family
                        break

            # Check widget values for model names
            for widget in widgets:
                if isinstance(widget, str) and len(widget) > 5:
                    # Check for model file patterns
                    if any(ext in widget.lower() for ext in ['.safetensors', '.ckpt', '.pt']):
                        model_names.append(widget)
                        # Try to detect family from model name
                        for family, patterns in cls.MODEL_PATTERNS.items():
                            for pattern in patterns:
                                if re.search(pattern, widget, re.IGNORECASE):
                                    if family != ModelFamily.VIDEO and model_family == ModelFamily.UNKNOWN:
                                        model_family = family
                                    break

        # Detect features
        has_video = False
        has_controlnet = False
        has_ipadapter = False
        has_upscaler = False
        lora_count = 0
        resolutions = []

        for node in nodes:
            node_type = node.get("type", "").lower()

            if any(p in node_type for p in ["video", "animatediff", "vhs", "svd"]):
                has_video = True
            if "controlnet" in node_type:
                has_controlnet = True
            if any(p in node_type for p in ["ipadapter", "instantid"]):
                has_ipadapter = True
            if any(p in node_type for p in ["upscale", "esrgan", "4x", "8x"]):
                has_upscaler = True
            if "lora" in node_type:
                lora_count += 1

            # Extract resolutions
            for widget in node.get("widgets_values", []):
                if isinstance(widget, int) and 256 <= widget <= 4096 and widget % 8 == 0:
                    resolutions.append(widget)

        # Determine primary resolution
        if resolutions:
            # Most common resolution pair
            sorted_res = sorted(set(resolutions), reverse=True)
            if len(sorted_res) >= 2:
                resolution = (sorted_res[0], sorted_res[1])
            else:
                resolution = (sorted_res[0], sorted_res[0])
        else:
            resolution = (512, 512)

        # Generate hash
        hash_input = f"{model_family.value}:{','.join(model_names[:3])}:{resolution}"
        hash_key = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        return WorkflowFingerprint(
            model_family=model_family,
            model_names=model_names[:5],  # Keep top 5
            resolution=resolution,
            has_video=has_video,
            has_controlnet=has_controlnet,
            has_ipadapter=has_ipadapter,
            has_upscaler=has_upscaler,
            lora_count=lora_count,
            node_count=len(nodes),
            hash_key=hash_key,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW BEAUTIFICATION - Organization & Visual Improvements
# ═══════════════════════════════════════════════════════════════════════════════

class BeautifyMode(Enum):
    """Different beautification approaches."""
    ORGANIZE = "organize"          # Logical grouping
    ALIGN_GRID = "align_grid"      # Snap to grid
    FLOW_LEFT_RIGHT = "flow_lr"    # Left-to-right flow
    FLOW_TOP_DOWN = "flow_td"      # Top-to-bottom flow
    COMPACT = "compact"            # Minimize space
    EXPAND = "expand"              # Add breathing room
    COLOR_CODE = "color_code"      # Add colors by function


class WorkflowBeautifier:
    """Organize and beautify ComfyUI workflows."""

    # Node categories for organization
    NODE_CATEGORIES = {
        "input": ["load", "input", "image", "mask", "latent.*empty"],
        "model": ["checkpoint", "model", "unet", "vae", "clip", "lora"],
        "conditioning": ["prompt", "text", "conditioning", "encode", "clip.*text"],
        "sampling": ["sampler", "ksampler", "sample", "noise"],
        "decode": ["decode", "vae.*decode"],
        "output": ["save", "preview", "output", "display"],
        "controlnet": ["controlnet", "preprocessor", "openpose", "depth", "canny"],
        "ipadapter": ["ipadapter", "instantid", "faceid"],
        "upscale": ["upscale", "esrgan", "4x", "8x", "ultimate"],
        "video": ["video", "animate", "vhs", "frame"],
        "utility": ["switch", "mux", "preview", "string", "int", "float"],
    }

    # Colors for different categories (ComfyUI color format)
    CATEGORY_COLORS = {
        "input": "#2a363b",
        "model": "#3d5a80",
        "conditioning": "#5c4d7d",
        "sampling": "#8b4513",
        "decode": "#355e3b",
        "output": "#228b22",
        "controlnet": "#4a90d9",
        "ipadapter": "#9370db",
        "upscale": "#ff8c00",
        "video": "#dc143c",
        "utility": "#708090",
    }

    GRID_SIZE = 50  # Snap to this grid size

    @classmethod
    def beautify(cls, workflow: Dict, mode: BeautifyMode) -> Dict:
        """Apply beautification to workflow."""
        workflow = copy.deepcopy(workflow)

        if mode == BeautifyMode.ORGANIZE:
            workflow = cls._organize_by_category(workflow)
        elif mode == BeautifyMode.ALIGN_GRID:
            workflow = cls._align_to_grid(workflow)
        elif mode == BeautifyMode.FLOW_LEFT_RIGHT:
            workflow = cls._arrange_flow(workflow, horizontal=True)
        elif mode == BeautifyMode.FLOW_TOP_DOWN:
            workflow = cls._arrange_flow(workflow, horizontal=False)
        elif mode == BeautifyMode.COMPACT:
            workflow = cls._compact_layout(workflow)
        elif mode == BeautifyMode.EXPAND:
            workflow = cls._expand_layout(workflow)
        elif mode == BeautifyMode.COLOR_CODE:
            workflow = cls._apply_color_coding(workflow)

        return workflow

    @classmethod
    def _categorize_node(cls, node: Dict) -> str:
        """Determine category for a node."""
        node_type = node.get("type", "").lower()

        for category, patterns in cls.NODE_CATEGORIES.items():
            for pattern in patterns:
                if re.search(pattern, node_type, re.IGNORECASE):
                    return category

        return "utility"

    @classmethod
    def _organize_by_category(cls, workflow: Dict) -> Dict:
        """Group nodes by category and arrange in columns."""
        nodes = workflow.get("nodes", [])
        if not nodes:
            return workflow

        # Categorize nodes
        categorized: Dict[str, List[Dict]] = {}
        for node in nodes:
            cat = cls._categorize_node(node)
            if cat not in categorized:
                categorized[cat] = []
            categorized[cat].append(node)

        # Define category order (processing flow)
        category_order = [
            "input", "model", "conditioning", "controlnet", "ipadapter",
            "sampling", "decode", "upscale", "video", "output", "utility"
        ]

        # Arrange nodes
        x_offset = 100
        column_width = 400
        y_spacing = 150
        node_height = 120

        for category in category_order:
            if category not in categorized:
                continue

            cat_nodes = categorized[category]
            y_offset = 100

            for node in cat_nodes:
                pos = node.get("pos", [0, 0])
                if isinstance(pos, dict):
                    pos = [pos.get("0", 0), pos.get("1", 0)]

                node["pos"] = [x_offset, y_offset]
                y_offset += node_height + y_spacing

            x_offset += column_width

        return workflow

    @classmethod
    def _align_to_grid(cls, workflow: Dict) -> Dict:
        """Snap all nodes to grid."""
        for node in workflow.get("nodes", []):
            pos = node.get("pos", [0, 0])
            if isinstance(pos, dict):
                pos = [pos.get("0", 0), pos.get("1", 0)]

            # Snap to grid
            pos[0] = round(pos[0] / cls.GRID_SIZE) * cls.GRID_SIZE
            pos[1] = round(pos[1] / cls.GRID_SIZE) * cls.GRID_SIZE
            node["pos"] = pos

        return workflow

    @classmethod
    def _arrange_flow(cls, workflow: Dict, horizontal: bool = True) -> Dict:
        """Arrange nodes in a flow based on connections."""
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])

        if not nodes:
            return workflow

        # Build connection graph
        node_map = {n.get("id"): n for n in nodes}
        incoming: Dict[int, List[int]] = {n.get("id"): [] for n in nodes}
        outgoing: Dict[int, List[int]] = {n.get("id"): [] for n in nodes}

        for link in links:
            if len(link) >= 4:
                # link format: [link_id, source_node, source_slot, dest_node, dest_slot, type]
                src_node = link[1]
                dst_node = link[3]
                if src_node in outgoing and dst_node in incoming:
                    outgoing[src_node].append(dst_node)
                    incoming[dst_node].append(src_node)

        # Find root nodes (no incoming connections)
        roots = [n.get("id") for n in nodes if not incoming.get(n.get("id"), [])]
        if not roots:
            roots = [nodes[0].get("id")]

        # BFS to assign levels
        levels: Dict[int, int] = {}
        queue = [(r, 0) for r in roots]
        visited = set()

        while queue:
            node_id, level = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            levels[node_id] = max(levels.get(node_id, 0), level)

            for child in outgoing.get(node_id, []):
                if child not in visited:
                    queue.append((child, level + 1))

        # Group by level
        level_groups: Dict[int, List[int]] = {}
        for node_id, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node_id)

        # Position nodes
        x_spacing = 400
        y_spacing = 200
        node_height = 120

        for level, node_ids in level_groups.items():
            for i, node_id in enumerate(node_ids):
                if node_id in node_map:
                    if horizontal:
                        node_map[node_id]["pos"] = [
                            100 + level * x_spacing,
                            100 + i * (node_height + y_spacing)
                        ]
                    else:
                        node_map[node_id]["pos"] = [
                            100 + i * x_spacing,
                            100 + level * (node_height + y_spacing)
                        ]

        return workflow

    @classmethod
    def _compact_layout(cls, workflow: Dict) -> Dict:
        """Reduce spacing between nodes."""
        nodes = workflow.get("nodes", [])
        if not nodes:
            return workflow

        # Find bounding box
        min_x = min(n.get("pos", [0, 0])[0] if isinstance(n.get("pos"), list) else 0 for n in nodes)
        min_y = min(n.get("pos", [0, 0])[1] if isinstance(n.get("pos"), list) else 0 for n in nodes)

        # Reposition with tighter spacing (0.7x)
        for node in nodes:
            pos = node.get("pos", [0, 0])
            if isinstance(pos, dict):
                pos = [pos.get("0", 0), pos.get("1", 0)]

            new_x = 50 + (pos[0] - min_x) * 0.7
            new_y = 50 + (pos[1] - min_y) * 0.7
            node["pos"] = [new_x, new_y]

        return cls._align_to_grid(workflow)

    @classmethod
    def _expand_layout(cls, workflow: Dict) -> Dict:
        """Add breathing room between nodes."""
        nodes = workflow.get("nodes", [])
        if not nodes:
            return workflow

        # Find center
        avg_x = sum(n.get("pos", [0, 0])[0] if isinstance(n.get("pos"), list) else 0 for n in nodes) / len(nodes)
        avg_y = sum(n.get("pos", [0, 0])[1] if isinstance(n.get("pos"), list) else 0 for n in nodes) / len(nodes)

        # Expand from center (1.4x)
        for node in nodes:
            pos = node.get("pos", [0, 0])
            if isinstance(pos, dict):
                pos = [pos.get("0", 0), pos.get("1", 0)]

            new_x = avg_x + (pos[0] - avg_x) * 1.4
            new_y = avg_y + (pos[1] - avg_y) * 1.4
            node["pos"] = [new_x, new_y]

        return cls._align_to_grid(workflow)

    @classmethod
    def _apply_color_coding(cls, workflow: Dict) -> Dict:
        """Apply colors to nodes based on their category."""
        for node in workflow.get("nodes", []):
            category = cls._categorize_node(node)
            if category in cls.CATEGORY_COLORS:
                node["bgcolor"] = cls.CATEGORY_COLORS[category]

        return workflow

    @classmethod
    def create_groups_by_category(cls, workflow: Dict) -> Dict:
        """Create visual groups for node categories."""
        workflow = copy.deepcopy(workflow)
        nodes = workflow.get("nodes", [])

        # Categorize nodes and find bounds
        category_bounds: Dict[str, Dict] = {}

        for node in nodes:
            cat = cls._categorize_node(node)
            pos = node.get("pos", [0, 0])
            if isinstance(pos, dict):
                pos = [pos.get("0", 0), pos.get("1", 0)]

            size = node.get("size", [200, 100])
            if isinstance(size, dict):
                size = [size.get("0", 200), size.get("1", 100)]

            if cat not in category_bounds:
                category_bounds[cat] = {
                    "min_x": pos[0],
                    "min_y": pos[1],
                    "max_x": pos[0] + size[0],
                    "max_y": pos[1] + size[1],
                }
            else:
                category_bounds[cat]["min_x"] = min(category_bounds[cat]["min_x"], pos[0])
                category_bounds[cat]["min_y"] = min(category_bounds[cat]["min_y"], pos[1])
                category_bounds[cat]["max_x"] = max(category_bounds[cat]["max_x"], pos[0] + size[0])
                category_bounds[cat]["max_y"] = max(category_bounds[cat]["max_y"], pos[1] + size[1])

        # Create groups
        groups = workflow.get("groups", [])
        padding = 30

        for cat, bounds in category_bounds.items():
            if cat == "utility":
                continue  # Skip utility nodes

            groups.append({
                "title": cat.replace("_", " ").title(),
                "bounding": [
                    bounds["min_x"] - padding,
                    bounds["min_y"] - padding,
                    bounds["max_x"] - bounds["min_x"] + padding * 2,
                    bounds["max_y"] - bounds["min_y"] + padding * 2,
                ],
                "color": cls.CATEGORY_COLORS.get(cat, "#708090"),
            })

        workflow["groups"] = groups
        return workflow


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW DIFF - Show what changed
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowDiff:
    """Compare two workflows and show differences."""

    @staticmethod
    def diff(old_workflow: Dict, new_workflow: Dict) -> Dict:
        """Generate a diff between two workflows."""
        old_nodes = {n.get("id"): n for n in old_workflow.get("nodes", [])}
        new_nodes = {n.get("id"): n for n in new_workflow.get("nodes", [])}

        diff = {
            "added_nodes": [],
            "removed_nodes": [],
            "modified_nodes": [],
            "unchanged_nodes": 0,
        }

        # Find added and modified nodes
        for node_id, node in new_nodes.items():
            if node_id not in old_nodes:
                diff["added_nodes"].append({
                    "id": node_id,
                    "type": node.get("type"),
                    "title": node.get("title", node.get("type")),
                })
            else:
                old_node = old_nodes[node_id]
                changes = WorkflowDiff._compare_nodes(old_node, node)
                if changes:
                    diff["modified_nodes"].append({
                        "id": node_id,
                        "type": node.get("type"),
                        "changes": changes,
                    })
                else:
                    diff["unchanged_nodes"] += 1

        # Find removed nodes
        for node_id, node in old_nodes.items():
            if node_id not in new_nodes:
                diff["removed_nodes"].append({
                    "id": node_id,
                    "type": node.get("type"),
                    "title": node.get("title", node.get("type")),
                })

        return diff

    @staticmethod
    def _compare_nodes(old_node: Dict, new_node: Dict) -> List[Dict]:
        """Compare two nodes and return list of changes."""
        changes = []

        # Compare widgets
        old_widgets = old_node.get("widgets_values", [])
        new_widgets = new_node.get("widgets_values", [])

        for i in range(max(len(old_widgets), len(new_widgets))):
            old_val = old_widgets[i] if i < len(old_widgets) else None
            new_val = new_widgets[i] if i < len(new_widgets) else None

            if old_val != new_val:
                changes.append({
                    "widget_index": i,
                    "old_value": old_val,
                    "new_value": new_val,
                })

        # Compare mode (bypass/mute)
        if old_node.get("mode") != new_node.get("mode"):
            changes.append({
                "property": "mode",
                "old_value": old_node.get("mode"),
                "new_value": new_node.get("mode"),
            })

        return changes

    @staticmethod
    def summarize(diff: Dict) -> str:
        """Create a human-readable summary of the diff."""
        lines = []

        if diff["added_nodes"]:
            lines.append(f"+ {len(diff['added_nodes'])} nodes added")
            for n in diff["added_nodes"][:3]:
                lines.append(f"  + {n['type']}")

        if diff["removed_nodes"]:
            lines.append(f"- {len(diff['removed_nodes'])} nodes removed")
            for n in diff["removed_nodes"][:3]:
                lines.append(f"  - {n['type']}")

        if diff["modified_nodes"]:
            lines.append(f"~ {len(diff['modified_nodes'])} nodes modified")
            for n in diff["modified_nodes"][:5]:
                changes_str = ", ".join([
                    f"{c.get('property', f'widget[{c.get(\"widget_index\")}]')}"
                    for c in n["changes"][:3]
                ])
                lines.append(f"  ~ {n['type']}: {changes_str}")

        if diff["unchanged_nodes"]:
            lines.append(f"= {diff['unchanged_nodes']} nodes unchanged")

        return "\n".join(lines) if lines else "No changes detected"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def is_safe_to_overwrite(original: Dict, modified: Dict) -> Tuple[bool, List[str]]:
    """
    Check if it's safe to overwrite the original with the modified workflow.
    Returns (is_safe, list_of_warnings).
    """
    original_fp = WorkflowFingerprinter.fingerprint(original)
    modified_fp = WorkflowFingerprinter.fingerprint(modified)

    return modified_fp.is_compatible_with(original_fp)


def suggest_filename(original_path: str, modified: Dict) -> str:
    """Suggest a filename for an incompatible workflow."""
    fp = WorkflowFingerprinter.fingerprint(modified)

    # Add model family to filename
    base = original_path.rsplit(".", 1)[0]
    ext = original_path.rsplit(".", 1)[1] if "." in original_path else "json"

    return f"{base}_{fp.model_family.value}.{ext}"
