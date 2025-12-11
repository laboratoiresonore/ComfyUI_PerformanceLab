#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               🎛️ MODEL TUNER - Smart Optimization for Any Model             ║
║        Auto-detect SD1.5, SDXL, Flux, and optimize with one click!          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
- Auto-detect model type from workflow
- Model-specific optimization presets
- LoRA strength tuning wizard
- Sampler/scheduler recommendations
- One-click optimization profiles
"""

import os
import json
import copy
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class ModelType(Enum):
    SD15 = "sd15"           # Stable Diffusion 1.5
    SD21 = "sd21"           # Stable Diffusion 2.1
    SDXL = "sdxl"           # Stable Diffusion XL
    SDXL_TURBO = "sdxl_turbo"
    SD3 = "sd3"             # Stable Diffusion 3
    FLUX_DEV = "flux_dev"   # Flux Dev
    FLUX_SCHNELL = "flux_schnell"  # Flux Schnell
    CASCADE = "cascade"     # Stable Cascade
    PIXART = "pixart"       # PixArt
    HUNYUAN = "hunyuan"     # HunyuanDiT
    KOLORS = "kolors"       # Kolors
    AURAFLOW = "auraflow"   # AuraFlow
    UNKNOWN = "unknown"


@dataclass
class ModelProfile:
    """Optimal settings for a specific model type."""
    name: str
    type: ModelType
    base_resolution: Tuple[int, int]
    optimal_steps: int
    min_steps: int
    max_steps: int
    default_cfg: float
    cfg_range: Tuple[float, float]
    best_samplers: List[str]
    best_schedulers: List[str]
    vram_estimate_gb: float
    supports_refiner: bool
    clip_skip: int
    lora_strength_default: float
    lora_strength_range: Tuple[float, float]
    tips: List[str]


# Model profiles database
MODEL_PROFILES: Dict[ModelType, ModelProfile] = {
    ModelType.SD15: ModelProfile(
        name="Stable Diffusion 1.5",
        type=ModelType.SD15,
        base_resolution=(512, 512),
        optimal_steps=25,
        min_steps=15,
        max_steps=50,
        default_cfg=7.5,
        cfg_range=(5.0, 12.0),
        best_samplers=["dpmpp_2m", "euler_ancestral", "dpmpp_sde"],
        best_schedulers=["karras", "normal", "exponential"],
        vram_estimate_gb=4.0,
        supports_refiner=False,
        clip_skip=1,
        lora_strength_default=0.8,
        lora_strength_range=(0.4, 1.2),
        tips=[
            "Use 512x512 or 768x768 for best results",
            "CFG 7-8 works well for most prompts",
            "Karras scheduler often gives cleaner results",
            "Clip skip 2 for anime models",
        ]
    ),
    ModelType.SD21: ModelProfile(
        name="Stable Diffusion 2.1",
        type=ModelType.SD21,
        base_resolution=(768, 768),
        optimal_steps=30,
        min_steps=20,
        max_steps=50,
        default_cfg=7.0,
        cfg_range=(5.0, 10.0),
        best_samplers=["dpmpp_2m", "euler", "heun"],
        best_schedulers=["karras", "normal"],
        vram_estimate_gb=5.0,
        supports_refiner=False,
        clip_skip=1,
        lora_strength_default=0.75,
        lora_strength_range=(0.3, 1.0),
        tips=[
            "Best at 768x768 native resolution",
            "Requires different prompting style than SD1.5",
            "Lower CFG values often work better",
        ]
    ),
    ModelType.SDXL: ModelProfile(
        name="Stable Diffusion XL",
        type=ModelType.SDXL,
        base_resolution=(1024, 1024),
        optimal_steps=30,
        min_steps=20,
        max_steps=50,
        default_cfg=7.0,
        cfg_range=(4.0, 10.0),
        best_samplers=["dpmpp_2m_sde", "euler_ancestral", "dpmpp_2m"],
        best_schedulers=["karras", "sgm_uniform", "normal"],
        vram_estimate_gb=6.5,
        supports_refiner=True,
        clip_skip=2,
        lora_strength_default=0.8,
        lora_strength_range=(0.4, 1.2),
        tips=[
            "1024x1024 is the native resolution",
            "Use refiner at ~0.8 denoise for best quality",
            "Lower CFG (5-7) often produces better results",
            "Add style tags like 'cinematic, detailed' for better output",
        ]
    ),
    ModelType.SDXL_TURBO: ModelProfile(
        name="SDXL Turbo",
        type=ModelType.SDXL_TURBO,
        base_resolution=(512, 512),
        optimal_steps=4,
        min_steps=1,
        max_steps=8,
        default_cfg=1.0,
        cfg_range=(0.0, 2.0),
        best_samplers=["euler_ancestral", "euler", "dpmpp_sde"],
        best_schedulers=["normal", "sgm_uniform"],
        vram_estimate_gb=5.0,
        supports_refiner=False,
        clip_skip=2,
        lora_strength_default=0.6,
        lora_strength_range=(0.2, 0.8),
        tips=[
            "Use only 1-4 steps",
            "CFG must be 0-1, higher breaks output",
            "512x512 works best, can go to 768",
            "No negative prompts needed",
        ]
    ),
    ModelType.SD3: ModelProfile(
        name="Stable Diffusion 3",
        type=ModelType.SD3,
        base_resolution=(1024, 1024),
        optimal_steps=28,
        min_steps=20,
        max_steps=50,
        default_cfg=4.5,
        cfg_range=(3.0, 7.0),
        best_samplers=["euler", "dpmpp_2m"],
        best_schedulers=["sgm_uniform", "normal"],
        vram_estimate_gb=8.0,
        supports_refiner=False,
        clip_skip=1,
        lora_strength_default=0.7,
        lora_strength_range=(0.3, 1.0),
        tips=[
            "Uses three text encoders (CLIP L, CLIP G, T5)",
            "Lower CFG (3.5-5) works best",
            "28 steps is optimal for most cases",
            "Great at text rendering",
        ]
    ),
    ModelType.FLUX_DEV: ModelProfile(
        name="Flux Dev",
        type=ModelType.FLUX_DEV,
        base_resolution=(1024, 1024),
        optimal_steps=28,
        min_steps=20,
        max_steps=50,
        default_cfg=3.5,
        cfg_range=(1.0, 5.0),
        best_samplers=["euler", "ipndm", "deis"],
        best_schedulers=["normal", "beta", "linear"],
        vram_estimate_gb=12.0,
        supports_refiner=False,
        clip_skip=1,
        lora_strength_default=0.8,
        lora_strength_range=(0.4, 1.2),
        tips=[
            "Use guidance scale 3-4 (called 'distilled CFG')",
            "20-30 steps is optimal",
            "T5 encoder is critical - use full prompts",
            "Supports any aspect ratio well",
            "Great at text rendering and hands",
        ]
    ),
    ModelType.FLUX_SCHNELL: ModelProfile(
        name="Flux Schnell",
        type=ModelType.FLUX_SCHNELL,
        base_resolution=(1024, 1024),
        optimal_steps=4,
        min_steps=1,
        max_steps=8,
        default_cfg=1.0,
        cfg_range=(0.0, 2.0),
        best_samplers=["euler", "ipndm"],
        best_schedulers=["normal", "beta"],
        vram_estimate_gb=10.0,
        supports_refiner=False,
        clip_skip=1,
        lora_strength_default=0.5,
        lora_strength_range=(0.2, 0.8),
        tips=[
            "Only needs 1-4 steps!",
            "No CFG needed (set to 1)",
            "Fastest high-quality model available",
            "Great for iteration and testing",
        ]
    ),
    ModelType.CASCADE: ModelProfile(
        name="Stable Cascade",
        type=ModelType.CASCADE,
        base_resolution=(1024, 1024),
        optimal_steps=20,
        min_steps=10,
        max_steps=30,
        default_cfg=4.0,
        cfg_range=(2.0, 7.0),
        best_samplers=["euler", "dpmpp_2m"],
        best_schedulers=["normal"],
        vram_estimate_gb=8.0,
        supports_refiner=True,
        clip_skip=1,
        lora_strength_default=0.7,
        lora_strength_range=(0.3, 1.0),
        tips=[
            "Two-stage model: Stage C (latent) + Stage B (decode)",
            "20 steps for Stage C, 10 for Stage B",
            "Lower VRAM than SDXL for same quality",
        ]
    ),
    ModelType.PIXART: ModelProfile(
        name="PixArt",
        type=ModelType.PIXART,
        base_resolution=(1024, 1024),
        optimal_steps=20,
        min_steps=15,
        max_steps=35,
        default_cfg=4.5,
        cfg_range=(3.0, 7.0),
        best_samplers=["dpmpp_2m", "euler"],
        best_schedulers=["karras", "normal"],
        vram_estimate_gb=8.0,
        supports_refiner=False,
        clip_skip=1,
        lora_strength_default=0.7,
        lora_strength_range=(0.3, 1.0),
        tips=[
            "T5 text encoder for prompts",
            "Good at artistic styles",
            "Lower steps than SD models",
        ]
    ),
    ModelType.HUNYUAN: ModelProfile(
        name="HunyuanDiT",
        type=ModelType.HUNYUAN,
        base_resolution=(1024, 1024),
        optimal_steps=30,
        min_steps=20,
        max_steps=50,
        default_cfg=6.0,
        cfg_range=(4.0, 9.0),
        best_samplers=["euler", "dpmpp_2m"],
        best_schedulers=["normal", "karras"],
        vram_estimate_gb=10.0,
        supports_refiner=False,
        clip_skip=1,
        lora_strength_default=0.8,
        lora_strength_range=(0.4, 1.0),
        tips=[
            "Supports both Chinese and English prompts",
            "Good at Asian aesthetics",
            "DiT architecture like SD3",
        ]
    ),
    ModelType.KOLORS: ModelProfile(
        name="Kolors",
        type=ModelType.KOLORS,
        base_resolution=(1024, 1024),
        optimal_steps=25,
        min_steps=15,
        max_steps=40,
        default_cfg=5.0,
        cfg_range=(3.0, 8.0),
        best_samplers=["euler", "dpmpp_2m_sde"],
        best_schedulers=["normal", "karras"],
        vram_estimate_gb=8.0,
        supports_refiner=False,
        clip_skip=1,
        lora_strength_default=0.75,
        lora_strength_range=(0.4, 1.0),
        tips=[
            "Great at Chinese text and aesthetics",
            "Similar to SDXL quality",
        ]
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class ModelDetector:
    """Automatically detect model type from workflow."""

    # Patterns to identify model types from node types and model names
    DETECTION_PATTERNS = {
        ModelType.FLUX_SCHNELL: [
            r"flux.*schnell", r"schnell", r"flux_schnell",
        ],
        ModelType.FLUX_DEV: [
            r"flux.*dev", r"flux(?!.*schnell)", r"flux_dev", r"FluxGuidance",
        ],
        ModelType.SD3: [
            r"sd3", r"stable.*diffusion.*3", r"sd_3",
        ],
        ModelType.SDXL_TURBO: [
            r"turbo", r"sdxl.*turbo", r"lightning",
        ],
        ModelType.SDXL: [
            r"sdxl", r"sd_xl", r"stable.*xl", r"1024.*1024",
        ],
        ModelType.CASCADE: [
            r"cascade", r"stage.*[bc]", r"stable.*cascade",
        ],
        ModelType.PIXART: [
            r"pixart", r"pix.*art",
        ],
        ModelType.HUNYUAN: [
            r"hunyuan", r"hunyuandit",
        ],
        ModelType.KOLORS: [
            r"kolors",
        ],
        ModelType.SD21: [
            r"sd.*2\.1", r"sd21", r"768.*v",
        ],
        ModelType.SD15: [
            r"sd.*1\.5", r"sd15", r"v1-5", r"512.*512",
        ],
    }

    # Node type patterns that indicate specific models
    NODE_PATTERNS = {
        "FluxGuidance": ModelType.FLUX_DEV,
        "FluxLoader": ModelType.FLUX_DEV,
        "UNETLoaderFlux": ModelType.FLUX_DEV,
        "ModelSamplingFlux": ModelType.FLUX_DEV,
        "SD3": ModelType.SD3,
        "StableCascade": ModelType.CASCADE,
        "CascadeLoader": ModelType.CASCADE,
        "PixArt": ModelType.PIXART,
        "Hunyuan": ModelType.HUNYUAN,
        "Kolors": ModelType.KOLORS,
    }

    @classmethod
    def detect_from_workflow(cls, workflow: Dict) -> Tuple[ModelType, float, List[str]]:
        """
        Detect model type from workflow.
        Returns (model_type, confidence, detection_hints).
        """
        hints = []
        scores: Dict[ModelType, float] = {t: 0.0 for t in ModelType}

        nodes = workflow.get("nodes", [])

        for node in nodes:
            node_type = node.get("type", "")
            node_title = node.get("title", "")
            widgets = node.get("widgets_values", [])

            # Check node type patterns
            for pattern, model_type in cls.NODE_PATTERNS.items():
                if pattern.lower() in node_type.lower():
                    scores[model_type] += 3.0
                    hints.append(f"Node type '{node_type}' suggests {model_type.value}")

            # Check widgets for model names
            for widget in widgets:
                if isinstance(widget, str):
                    widget_lower = widget.lower()
                    for model_type, patterns in cls.DETECTION_PATTERNS.items():
                        for pattern in patterns:
                            if re.search(pattern, widget_lower, re.IGNORECASE):
                                scores[model_type] += 2.0
                                hints.append(f"Widget value '{widget[:50]}...' suggests {model_type.value}")
                                break

            # Check node title
            title_lower = node_title.lower()
            for model_type, patterns in cls.DETECTION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, title_lower, re.IGNORECASE):
                        scores[model_type] += 1.5
                        hints.append(f"Node title '{node_title}' suggests {model_type.value}")
                        break

        # Resolution hints
        resolutions = cls._extract_resolutions(workflow)
        if resolutions:
            max_res = max(resolutions)
            if max_res <= 576:
                scores[ModelType.SD15] += 1.0
                scores[ModelType.SDXL_TURBO] += 1.0
                scores[ModelType.FLUX_SCHNELL] += 0.5
            elif max_res <= 768:
                scores[ModelType.SD15] += 0.5
                scores[ModelType.SD21] += 1.0
            elif max_res >= 1024:
                scores[ModelType.SDXL] += 0.5
                scores[ModelType.FLUX_DEV] += 0.5
                scores[ModelType.SD3] += 0.5

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Calculate confidence
        total_score = sum(scores.values())
        confidence = (best_score / total_score) if total_score > 0 else 0.0

        if best_score < 1.0:
            return ModelType.UNKNOWN, 0.0, ["Could not detect model type"]

        return best_type, min(confidence, 1.0), hints[:5]

    @staticmethod
    def _extract_resolutions(workflow: Dict) -> List[int]:
        """Extract resolution values from workflow."""
        resolutions = []
        for node in workflow.get("nodes", []):
            for widget in node.get("widgets_values", []):
                if isinstance(widget, int) and 256 <= widget <= 4096 and widget % 8 == 0:
                    resolutions.append(widget)
        return resolutions


# ═══════════════════════════════════════════════════════════════════════════════
# LORA TUNER
# ═══════════════════════════════════════════════════════════════════════════════

class LoRATuner:
    """Smart LoRA strength tuning."""

    @staticmethod
    def find_lora_nodes(workflow: Dict) -> List[Dict]:
        """Find all LoRA-related nodes in workflow."""
        lora_nodes = []
        for node in workflow.get("nodes", []):
            node_type = node.get("type", "").lower()
            if "lora" in node_type:
                lora_nodes.append({
                    "id": node.get("id"),
                    "type": node.get("type"),
                    "title": node.get("title", node.get("type")),
                    "widgets": node.get("widgets_values", []),
                    "node": node,
                })
        return lora_nodes

    @staticmethod
    def extract_lora_info(lora_nodes: List[Dict]) -> List[Dict]:
        """Extract LoRA names and strengths from nodes."""
        loras = []
        for ln in lora_nodes:
            widgets = ln["widgets"]
            # Common patterns: [lora_name, strength_model, strength_clip] or [lora_name, strength]
            for i, w in enumerate(widgets):
                if isinstance(w, str) and (".safetensors" in w or "lora" in w.lower()):
                    lora_info = {
                        "name": w,
                        "node_id": ln["id"],
                        "node_type": ln["type"],
                        "strength_model": None,
                        "strength_clip": None,
                    }
                    # Look for strength values after the name
                    if i + 1 < len(widgets) and isinstance(widgets[i + 1], (int, float)):
                        lora_info["strength_model"] = float(widgets[i + 1])
                    if i + 2 < len(widgets) and isinstance(widgets[i + 2], (int, float)):
                        lora_info["strength_clip"] = float(widgets[i + 2])
                    loras.append(lora_info)
        return loras

    @staticmethod
    def generate_strength_variants(
        base_strength: float,
        num_variants: int = 5,
        range_pct: float = 0.3
    ) -> List[float]:
        """Generate strength variants for A/B testing."""
        min_str = max(0.0, base_strength * (1 - range_pct))
        max_str = min(2.0, base_strength * (1 + range_pct))
        step = (max_str - min_str) / (num_variants - 1) if num_variants > 1 else 0

        return [round(min_str + i * step, 2) for i in range(num_variants)]

    @staticmethod
    def apply_lora_strength(
        workflow: Dict,
        node_id: int,
        strength_model: float,
        strength_clip: Optional[float] = None
    ) -> Dict:
        """Apply LoRA strength to a specific node."""
        workflow = copy.deepcopy(workflow)

        for node in workflow.get("nodes", []):
            if node.get("id") == node_id:
                widgets = node.get("widgets_values", [])
                # Find and update strength values
                for i, w in enumerate(widgets):
                    if isinstance(w, str) and (".safetensors" in w or "lora" in w.lower()):
                        if i + 1 < len(widgets):
                            widgets[i + 1] = strength_model
                        if strength_clip is not None and i + 2 < len(widgets):
                            widgets[i + 2] = strength_clip
                        break
                break

        return workflow

    @staticmethod
    def get_recommended_strength(model_type: ModelType, lora_type: str = "general") -> float:
        """Get recommended LoRA strength for a model type."""
        profile = MODEL_PROFILES.get(model_type)
        if profile:
            return profile.lora_strength_default

        # Defaults by lora type
        defaults = {
            "general": 0.8,
            "style": 0.7,
            "character": 0.9,
            "concept": 0.6,
            "detail": 0.5,
        }
        return defaults.get(lora_type, 0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLER/SCHEDULER RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════════

class SamplerRecommender:
    """Recommend samplers and schedulers based on model and use case."""

    USE_CASES = {
        "speed": {
            "description": "Fastest generation",
            "samplers": ["euler", "euler_ancestral", "lcm"],
            "schedulers": ["normal", "sgm_uniform"],
            "steps_multiplier": 0.6,
        },
        "quality": {
            "description": "Best quality output",
            "samplers": ["dpmpp_2m_sde", "dpmpp_3m_sde", "heun"],
            "schedulers": ["karras", "exponential"],
            "steps_multiplier": 1.2,
        },
        "balanced": {
            "description": "Good balance of speed and quality",
            "samplers": ["dpmpp_2m", "euler_ancestral"],
            "schedulers": ["karras", "normal"],
            "steps_multiplier": 1.0,
        },
        "creative": {
            "description": "More variation and creativity",
            "samplers": ["euler_ancestral", "dpmpp_2m_sde"],
            "schedulers": ["normal", "beta"],
            "steps_multiplier": 1.0,
        },
        "consistent": {
            "description": "Reproducible results",
            "samplers": ["euler", "dpmpp_2m"],
            "schedulers": ["normal", "simple"],
            "steps_multiplier": 1.0,
        },
    }

    @classmethod
    def recommend(
        cls,
        model_type: ModelType,
        use_case: str = "balanced"
    ) -> Dict[str, Any]:
        """Get sampler/scheduler recommendations."""
        profile = MODEL_PROFILES.get(model_type, MODEL_PROFILES[ModelType.SD15])
        use_case_config = cls.USE_CASES.get(use_case, cls.USE_CASES["balanced"])

        # Intersect model's best samplers with use case preferences
        recommended_samplers = [
            s for s in profile.best_samplers
            if s in use_case_config["samplers"]
        ] or profile.best_samplers[:2]

        recommended_schedulers = [
            s for s in profile.best_schedulers
            if s in use_case_config["schedulers"]
        ] or profile.best_schedulers[:2]

        steps = int(profile.optimal_steps * use_case_config["steps_multiplier"])
        steps = max(profile.min_steps, min(profile.max_steps, steps))

        return {
            "model": profile.name,
            "use_case": use_case,
            "description": use_case_config["description"],
            "recommended_sampler": recommended_samplers[0] if recommended_samplers else "euler",
            "alternative_samplers": recommended_samplers[1:],
            "recommended_scheduler": recommended_schedulers[0] if recommended_schedulers else "normal",
            "alternative_schedulers": recommended_schedulers[1:],
            "recommended_steps": steps,
            "recommended_cfg": profile.default_cfg,
            "cfg_range": profile.cfg_range,
        }

    @classmethod
    def get_all_recommendations(cls, model_type: ModelType) -> Dict[str, Dict]:
        """Get recommendations for all use cases."""
        return {
            use_case: cls.recommend(model_type, use_case)
            for use_case in cls.USE_CASES.keys()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class ModelOptimizer:
    """Apply model-specific optimizations to workflows."""

    @staticmethod
    def optimize_for_model(
        workflow: Dict,
        model_type: ModelType,
        profile_name: str = "balanced"
    ) -> Dict:
        """Apply model-specific optimizations."""
        workflow = copy.deepcopy(workflow)
        profile = MODEL_PROFILES.get(model_type, MODEL_PROFILES[ModelType.SD15])
        rec = SamplerRecommender.recommend(model_type, profile_name)

        for node in workflow.get("nodes", []):
            node_type = node.get("type", "").lower()
            widgets = node.get("widgets_values", [])

            # Update sampler nodes
            if "sampler" in node_type or "ksampler" in node_type:
                for i, w in enumerate(widgets):
                    # Update steps
                    if isinstance(w, int) and 1 <= w <= 150:
                        widgets[i] = rec["recommended_steps"]
                    # Update CFG
                    if isinstance(w, float) and 1.0 <= w <= 30.0:
                        widgets[i] = rec["recommended_cfg"]
                    # Update sampler name
                    if isinstance(w, str):
                        if any(s in w.lower() for s in ["euler", "dpm", "heun", "lms", "uni"]):
                            widgets[i] = rec["recommended_sampler"]
                        if any(s in w.lower() for s in ["karras", "normal", "sgm", "exponential"]):
                            widgets[i] = rec["recommended_scheduler"]

            # Update resolution for empty latent nodes
            if "empty" in node_type and "latent" in node_type:
                base_w, base_h = profile.base_resolution
                for i, w in enumerate(widgets):
                    if isinstance(w, int) and 256 <= w <= 4096 and w % 8 == 0:
                        # Maintain aspect ratio if possible
                        if i == 0:  # Width
                            widgets[i] = base_w
                        elif i == 1:  # Height
                            widgets[i] = base_h

        return workflow

    @staticmethod
    def create_speed_variant(workflow: Dict, model_type: ModelType) -> Dict:
        """Create a speed-optimized variant."""
        return ModelOptimizer.optimize_for_model(workflow, model_type, "speed")

    @staticmethod
    def create_quality_variant(workflow: Dict, model_type: ModelType) -> Dict:
        """Create a quality-optimized variant."""
        return ModelOptimizer.optimize_for_model(workflow, model_type, "quality")

    @staticmethod
    def apply_turbo_settings(workflow: Dict) -> Dict:
        """Apply turbo/lightning model settings (1-4 steps, low CFG)."""
        workflow = copy.deepcopy(workflow)

        for node in workflow.get("nodes", []):
            node_type = node.get("type", "").lower()
            widgets = node.get("widgets_values", [])

            if "sampler" in node_type or "ksampler" in node_type:
                for i, w in enumerate(widgets):
                    if isinstance(w, int) and 1 <= w <= 150:  # Steps
                        widgets[i] = 4
                    if isinstance(w, float) and 1.0 <= w <= 30.0:  # CFG
                        widgets[i] = 1.0

        return workflow


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE TUNER (CLI)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelTunerCLI:
    """Interactive CLI for model tuning."""

    # Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    def __init__(self, workflow_path: str):
        self.workflow_path = workflow_path
        self.workflow = self._load_workflow()
        self.model_type, self.confidence, self.hints = ModelDetector.detect_from_workflow(self.workflow)
        self.profile = MODEL_PROFILES.get(self.model_type)

    def _load_workflow(self) -> Dict:
        with open(self.workflow_path, 'r') as f:
            return json.load(f)

    def _save_workflow(self, path: str, workflow: Dict):
        with open(path, 'w') as f:
            json.dump(workflow, f, indent=2)

    def styled(self, text: str, *styles) -> str:
        return f"{''.join(styles)}{text}{self.RESET}"

    def print_header(self):
        print(f"""
{self.CYAN}{self.BOLD}╔══════════════════════════════════════════════════════════════════╗
║                    🎛️  MODEL TUNER                               ║
║              Smart Optimization for Any Model                     ║
╚══════════════════════════════════════════════════════════════════╝{self.RESET}
""")

    def show_detection(self):
        """Show model detection results."""
        print(f"\n  {self.styled('📊 Model Detection', self.BOLD)}")
        print(f"  {'─' * 50}")

        if self.model_type == ModelType.UNKNOWN:
            print(f"  {self.styled('⚠ Could not detect model type', self.YELLOW)}")
            print(f"  {self.DIM}Please select manually{self.RESET}")
        else:
            confidence_pct = int(self.confidence * 100)
            conf_color = self.GREEN if confidence_pct > 70 else self.YELLOW
            print(f"  Detected: {self.styled(self.profile.name if self.profile else 'Unknown', self.BOLD, self.CYAN)}")
            print(f"  Confidence: {self.styled(f'{confidence_pct}%', conf_color)}")

            if self.hints:
                print(f"\n  {self.DIM}Detection hints:{self.RESET}")
                for hint in self.hints[:3]:
                    print(f"    • {self.DIM}{hint}{self.RESET}")

    def show_profile(self):
        """Show current model profile."""
        if not self.profile:
            return

        print(f"\n  {self.styled('⚙️ Optimal Settings for ' + self.profile.name, self.BOLD)}")
        print(f"  {'─' * 50}")
        print(f"  Resolution:  {self.styled(f'{self.profile.base_resolution[0]}x{self.profile.base_resolution[1]}', self.CYAN)}")
        print(f"  Steps:       {self.styled(str(self.profile.optimal_steps), self.CYAN)} (range: {self.profile.min_steps}-{self.profile.max_steps})")
        print(f"  CFG:         {self.styled(str(self.profile.default_cfg), self.CYAN)} (range: {self.profile.cfg_range[0]}-{self.profile.cfg_range[1]})")
        print(f"  VRAM Est:    {self.styled(f'~{self.profile.vram_estimate_gb} GB', self.CYAN)}")

        print(f"\n  {self.styled('Best Samplers:', self.DIM)} {', '.join(self.profile.best_samplers[:3])}")
        print(f"  {self.styled('Best Schedulers:', self.DIM)} {', '.join(self.profile.best_schedulers[:3])}")

        if self.profile.tips:
            print(f"\n  {self.styled('💡 Tips:', self.YELLOW)}")
            for tip in self.profile.tips[:3]:
                print(f"    • {tip}")

    def show_lora_info(self):
        """Show LoRA information."""
        lora_nodes = LoRATuner.find_lora_nodes(self.workflow)
        loras = LoRATuner.extract_lora_info(lora_nodes)

        if not loras:
            print(f"\n  {self.DIM}No LoRAs detected in workflow{self.RESET}")
            return

        print(f"\n  {self.styled('🎨 LoRAs Detected', self.BOLD)}")
        print(f"  {'─' * 50}")

        for lora in loras:
            name = os.path.basename(lora["name"])[:40]
            strength = lora.get("strength_model", "N/A")
            recommended = LoRATuner.get_recommended_strength(self.model_type)

            print(f"  • {self.styled(name, self.CYAN)}")
            print(f"    Current: {strength} | Recommended: {recommended}")

    def run(self):
        """Run the interactive tuner."""
        self.print_header()
        self.show_detection()
        self.show_profile()
        self.show_lora_info()

        print(f"\n  {self.styled('Actions:', self.BOLD)}")
        print(f"    {self.styled('1', self.CYAN)} Apply optimal settings")
        print(f"    {self.styled('2', self.CYAN)} Create speed variant")
        print(f"    {self.styled('3', self.CYAN)} Create quality variant")
        print(f"    {self.styled('4', self.CYAN)} LoRA strength wizard")
        print(f"    {self.styled('5', self.CYAN)} Show all recommendations")
        print(f"    {self.styled('Q', self.CYAN)} Quit")

        while True:
            choice = input(f"\n  {self.styled('▶', self.CYAN)} Choice: ").strip().lower()

            if choice == 'q':
                break
            elif choice == '1':
                self._apply_optimal()
            elif choice == '2':
                self._create_variant("speed")
            elif choice == '3':
                self._create_variant("quality")
            elif choice == '4':
                self._lora_wizard()
            elif choice == '5':
                self._show_all_recommendations()

    def _apply_optimal(self):
        """Apply optimal settings for detected model."""
        optimized = ModelOptimizer.optimize_for_model(self.workflow, self.model_type, "balanced")
        out_path = self.workflow_path.replace(".json", "_optimized.json")
        self._save_workflow(out_path, optimized)
        print(f"  {self.styled('✓', self.GREEN)} Saved: {out_path}")

    def _create_variant(self, variant_type: str):
        """Create a speed or quality variant."""
        if variant_type == "speed":
            optimized = ModelOptimizer.create_speed_variant(self.workflow, self.model_type)
        else:
            optimized = ModelOptimizer.create_quality_variant(self.workflow, self.model_type)

        out_path = self.workflow_path.replace(".json", f"_{variant_type}.json")
        self._save_workflow(out_path, optimized)
        print(f"  {self.styled('✓', self.GREEN)} Saved: {out_path}")

    def _lora_wizard(self):
        """Interactive LoRA tuning wizard."""
        lora_nodes = LoRATuner.find_lora_nodes(self.workflow)
        loras = LoRATuner.extract_lora_info(lora_nodes)

        if not loras:
            print(f"  {self.styled('No LoRAs found', self.YELLOW)}")
            return

        print(f"\n  {self.styled('🎨 LoRA Strength Wizard', self.BOLD)}")

        for i, lora in enumerate(loras):
            name = os.path.basename(lora["name"])[:30]
            current = lora.get("strength_model", 0.8)

            print(f"\n  LoRA: {self.styled(name, self.CYAN)}")
            print(f"  Current strength: {current}")

            variants = LoRATuner.generate_strength_variants(current)
            print(f"  Test variants: {variants}")

            new_strength = input(f"  New strength (Enter to keep): ").strip()
            if new_strength:
                try:
                    strength = float(new_strength)
                    self.workflow = LoRATuner.apply_lora_strength(
                        self.workflow, lora["node_id"], strength, strength
                    )
                    print(f"  {self.styled('✓', self.GREEN)} Updated to {strength}")
                except ValueError:
                    print(f"  {self.styled('Invalid value', self.RED)}")

    def _show_all_recommendations(self):
        """Show recommendations for all use cases."""
        print(f"\n  {self.styled('📋 All Recommendations for ' + (self.profile.name if self.profile else 'Unknown'), self.BOLD)}")
        print(f"  {'─' * 50}")

        all_recs = SamplerRecommender.get_all_recommendations(self.model_type)

        for use_case, rec in all_recs.items():
            print(f"\n  {self.styled(use_case.upper(), self.CYAN)} - {rec['description']}")
            print(f"    Sampler:   {rec['recommended_sampler']}")
            print(f"    Scheduler: {rec['recommended_scheduler']}")
            print(f"    Steps:     {rec['recommended_steps']}")
            print(f"    CFG:       {rec['recommended_cfg']}")


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION FUNCTIONS (for performance_lab.py)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_model(workflow: Dict) -> Dict:
    """Detect model type and return info dict."""
    model_type, confidence, hints = ModelDetector.detect_from_workflow(workflow)
    profile = MODEL_PROFILES.get(model_type)

    return {
        "type": model_type.value,
        "name": profile.name if profile else "Unknown",
        "confidence": confidence,
        "hints": hints,
        "profile": profile,
    }


def get_model_presets(model_type_str: str) -> Dict[str, Dict]:
    """Get all presets for a model type string."""
    try:
        model_type = ModelType(model_type_str)
    except ValueError:
        model_type = ModelType.UNKNOWN

    return SamplerRecommender.get_all_recommendations(model_type)


def optimize_workflow(workflow: Dict, preset: str = "balanced") -> Dict:
    """Optimize workflow based on detected model and preset."""
    model_type, _, _ = ModelDetector.detect_from_workflow(workflow)
    return ModelOptimizer.optimize_for_model(workflow, model_type, preset)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_tuner.py <workflow.json>")
        print("\nOr import as module:")
        print("  from model_tuner import detect_model, optimize_workflow")
        sys.exit(1)

    workflow_path = sys.argv[1]
    if not os.path.exists(workflow_path):
        print(f"File not found: {workflow_path}")
        sys.exit(1)

    tuner = ModelTunerCLI(workflow_path)
    tuner.run()


if __name__ == "__main__":
    main()
