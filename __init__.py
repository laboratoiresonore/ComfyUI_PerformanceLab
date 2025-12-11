"""
ComfyUI Performance Lab - Iterative workflow optimization with AI assistance.

This node pack provides comprehensive optimization tools:
- Performance monitoring (timing, VRAM tracking)
- Quick optimizations (resolution cap, bypass upscalers, etc.)
- Workflow analysis and diagnostics
- LLM prompt generation for AI-assisted optimization
- Before/after comparison

Install: Place in ComfyUI/custom_nodes/ComfyUI_PerformanceLab/
"""

import os
import sys
import json
import time
import copy
import re
from typing import Dict, Any, List, Tuple, Optional

# Version and metadata
__version__ = "0.8.0"
__author__ = "Laboratoire Sonore"
__description__ = "ComfyUI Performance Lab - Iterative workflow optimization with AI assistance"

# ComfyUI Manager integration - explicit exports
# WEB_DIRECTORY is None because we don't have custom web assets
WEB_DIRECTORY = None

# __all__ defines what's exported when using "from package import *"
__all__ = [
    "__version__",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

# Ensure the module directory is in path for imports
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MONITORING NODES
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_Timer:
    """Start Timer - Place at the START of your workflow."""

    DESCRIPTION = """⏱️ START TIMER

HOW TO USE:
1. Add this node at the BEGINNING of your workflow
2. Connect the 'timer' output to a Performance Report node at the END
3. Run your workflow - the report will show how long it took

TIPS:
• No inputs needed - just add it and connect
• Works with any workflow
• Pair with 📊 Performance Report to see results"""

    CATEGORY = "⚡ Performance Lab/Monitoring"
    FUNCTION = "start"
    RETURN_TYPES = ("PERF_TIMER",)
    RETURN_NAMES = ("timer",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def start(self):
        return ({"start_time": time.time(), "vram_readings": []},)


class PerfLab_Report:
    """Performance Report - Shows timing and VRAM results."""

    DESCRIPTION = """📊 PERFORMANCE REPORT

HOW TO USE:
1. Add this node at the END of your workflow
2. Connect ANY output from your last node to 'trigger'
3. (Optional) Connect a Timer node to 'timer' for accurate timing
4. Run workflow - results appear in console and outputs

OUTPUTS:
• report: Text summary of performance
• duration_sec: Time in seconds (for Compare node)
• peak_vram_gb: Peak VRAM used (for Compare node)

TIPS:
• Connect final image/latent to 'trigger' input
• Use with ⏱️ Start Timer for accurate timing
• Feed outputs to 📊 Compare Results for before/after"""

    CATEGORY = "⚡ Performance Lab/Monitoring"
    FUNCTION = "report"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "duration_sec", "peak_vram_gb")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {"tooltip": "Connect ANY output here to trigger the report"}),
            },
            "optional": {
                "timer": ("PERF_TIMER", {"tooltip": "Connect from ⏱️ Start Timer for accurate timing"}),
                "label": ("STRING", {"default": "Generation", "tooltip": "Name for this measurement"}),
            }
        }

    def report(self, trigger, timer=None, label="Generation"):
        end_time = time.time()

        # Calculate duration
        if timer and "start_time" in timer:
            duration = end_time - timer["start_time"]
        else:
            duration = 0.0

        # Get VRAM info
        peak_vram = 0.0
        current_vram = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
                current_vram = torch.cuda.memory_allocated() / (1024**3)
                torch.cuda.reset_peak_memory_stats()
        except:
            pass

        # Build report
        report_lines = [
            f"═══ {label} Complete ═══",
            f"⏱️  Duration: {duration:.2f} seconds",
            f"📈 Peak VRAM: {peak_vram:.2f} GB",
            f"📊 Current VRAM: {current_vram:.2f} GB",
        ]

        report = "\n".join(report_lines)
        print(f"\n[Performance Lab]\n{report}\n")

        return (report, duration, peak_vram)


class PerfLab_VRAMMonitor:
    """VRAM Monitor - Check GPU memory at any point."""

    DESCRIPTION = """💾 VRAM MONITOR

HOW TO USE:
1. Place this node ANYWHERE in your workflow
2. Connect any data to 'passthrough' (it passes through unchanged)
3. Run workflow - VRAM info prints to console

OUTPUTS:
• vram_info: Text showing used/free/total VRAM
• used_gb: VRAM currently in use
• free_gb: Available VRAM
• passthrough: Your input, unchanged

TIPS:
• Place between nodes to find which uses most VRAM
• Add checkpoint_name to label the measurement
• Passthrough lets you insert without breaking connections"""

    CATEGORY = "⚡ Performance Lab/Monitoring"
    FUNCTION = "check"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "*")
    RETURN_NAMES = ("vram_info", "used_gb", "free_gb", "passthrough")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "passthrough": ("*", {"tooltip": "Connect any data - it passes through unchanged"}),
                "checkpoint_name": ("STRING", {"default": "", "tooltip": "Label for this measurement point"}),
            }
        }

    def check(self, passthrough=None, checkpoint_name=""):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                free = total - reserved

                label = f" [{checkpoint_name}]" if checkpoint_name else ""
                info = f"💾 VRAM{label}: {used:.1f}GB used / {free:.1f}GB free / {total:.1f}GB total ({gpu_name})"
                print(f"[Performance Lab] {info}")
                return (info, used, free, passthrough)
            else:
                return ("No CUDA GPU", 0.0, 0.0, passthrough)
        except Exception as e:
            return (f"Error: {e}", 0.0, 0.0, passthrough)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK OPTIMIZATION NODES
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_CapResolution:
    """Cap Resolution - Limit dimensions for faster testing."""

    DESCRIPTION = """📐 CAP RESOLUTION

HOW TO USE:
1. Connect your width/height values to this node
2. Set 'max_size' to your target (e.g., 768)
3. Connect outputs to Empty Latent or other nodes
4. Toggle 'enabled' to quickly compare original vs capped

INPUTS:
• width/height: Your original dimensions
• max_size: Maximum allowed dimension
• enabled: Turn optimization on/off

WHY USE THIS:
• Resolution affects VRAM quadratically (2x res = 4x VRAM)
• 768px is great for testing, use 1024+ for finals
• Keeps aspect ratio intact"""

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "cap"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192,
                         "tooltip": "Original width - connect from your workflow or type a value"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192,
                          "tooltip": "Original height - connect from your workflow or type a value"}),
                "max_size": ("INT", {"default": 768, "min": 256, "max": 2048,
                            "tooltip": "Maximum dimension - 768 for testing, 1024+ for finals"}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True,
                           "tooltip": "ON = cap resolution, OFF = pass through original"}),
            }
        }

    def cap(self, width: int, height: int, max_size: int, enabled: bool = True):
        if not enabled:
            return (width, height)

        # Scale down proportionally if needed
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale) // 8 * 8  # Keep divisible by 8
            new_height = int(height * scale) // 8 * 8
            print(f"[Performance Lab] Resolution capped: {width}x{height} → {new_width}x{new_height}")
            return (new_width, new_height)

        return (width, height)


class PerfLab_ReduceSteps:
    """Reduce Steps - Lower sampling steps for faster iteration."""

    DESCRIPTION = """🔢 REDUCE STEPS

HOW TO USE:
1. Connect your steps value to this node
2. Set 'max_steps' to your testing limit (e.g., 20)
3. Connect output to your KSampler
4. Toggle 'enabled' to compare fast vs quality

INPUTS:
• steps: Your original step count
• max_steps: Maximum allowed steps
• enabled: Turn optimization on/off

WHY USE THIS:
• Steps scale linearly with time (2x steps = 2x time)
• 15-20 steps is enough to check composition
• Use full steps only for final renders"""

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "reduce"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("steps",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 200,
                         "tooltip": "Your original step count - connect from workflow or enter value"}),
                "max_steps": ("INT", {"default": 20, "min": 1, "max": 100,
                             "tooltip": "Maximum steps during testing - 15-20 is usually enough"}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True,
                           "tooltip": "ON = limit steps, OFF = use original value"}),
            }
        }

    def reduce(self, steps: int, max_steps: int, enabled: bool = True):
        if not enabled:
            return (steps,)

        if steps > max_steps:
            print(f"[Performance Lab] Steps reduced: {steps} → {max_steps}")
            return (max_steps,)
        return (steps,)


class PerfLab_ReduceBatch:
    """Reduce Batch - Force batch size to 1 for VRAM savings."""

    DESCRIPTION = """📦 REDUCE BATCH

HOW TO USE:
1. Connect your batch_size value to this node
2. Enable 'force_single' to always use batch=1
3. Connect output to Empty Latent Image

INPUTS:
• batch_size: Your original batch size
• force_single: Force to 1 when enabled

WHY USE THIS:
• Batch size multiplies VRAM usage directly
• Batch=4 uses 4x the VRAM of batch=1
• Essential for low VRAM GPUs
• Disable for final batch renders"""

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "reduce"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("batch_size",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64,
                              "tooltip": "Your original batch size - each image uses more VRAM"}),
            },
            "optional": {
                "force_single": ("BOOLEAN", {"default": True,
                                "tooltip": "ON = force batch=1 (saves VRAM), OFF = keep original"}),
            }
        }

    def reduce(self, batch_size: int, force_single: bool = True):
        if force_single and batch_size > 1:
            print(f"[Performance Lab] Batch size reduced: {batch_size} → 1")
            return (1,)
        return (batch_size,)


class PerfLab_OptimizeCFG:
    """Optimize CFG - Auto-adjust CFG for your model type."""

    DESCRIPTION = """🎯 OPTIMIZE CFG

HOW TO USE:
1. Select your model_type from the dropdown
2. Connect cfg output to your KSampler
3. Enable 'auto_adjust' to use optimal values

OPTIMAL CFG BY MODEL:
• SD 1.5: 7.5
• SDXL: 7.0
• SD3: 4.5
• Flux Dev: 3.5
• Flux Schnell: 1.0

WHY USE THIS:
• Wrong CFG causes black/burned images
• Flux needs LOW CFG (1-3.5)
• SD models need MEDIUM CFG (5-8)
• Select 'Custom' to use your own value"""

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "optimize"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("cfg",)

    MODEL_CFG = {
        "SD 1.5": 7.5,
        "SDXL": 7.0,
        "SD3": 4.5,
        "Flux Dev": 3.5,
        "Flux Schnell": 1.0,
        "Custom": 7.0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5,
                       "tooltip": "Your CFG value - will be adjusted based on model type"}),
                "model_type": (list(cls.MODEL_CFG.keys()),
                              {"tooltip": "Select your model - Flux needs low CFG, SD needs higher"}),
            },
            "optional": {
                "auto_adjust": ("BOOLEAN", {"default": True,
                               "tooltip": "ON = use optimal CFG for model, OFF = use your value"}),
            }
        }

    def optimize(self, cfg: float, model_type: str, auto_adjust: bool = True):
        if auto_adjust and model_type != "Custom":
            optimal = self.MODEL_CFG.get(model_type, cfg)
            if cfg != optimal:
                print(f"[Performance Lab] CFG adjusted for {model_type}: {cfg} → {optimal}")
            return (optimal,)
        return (cfg,)


class PerfLab_SpeedPreset:
    """Speed Preset - All optimizations in one node."""

    DESCRIPTION = """🚀 SPEED TEST PRESET

HOW TO USE:
1. Connect your width, height, steps, cfg
2. Connect outputs to your workflow nodes
3. Toggle 'enabled' to switch between test/production

WHAT IT DOES:
• Caps resolution to target (default 512px)
• Limits steps to target (default 15)
• Keeps CFG unchanged

OUTPUTS:
• width/height: Capped dimensions
• steps: Limited step count
• cfg: Passed through

USE WHEN:
• Testing workflow changes quickly
• Iterating on prompts/composition
• Debugging errors (fast feedback)"""

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "apply"
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "steps", "cfg")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192,
                         "tooltip": "Original width from your workflow"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192,
                          "tooltip": "Original height from your workflow"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200,
                         "tooltip": "Original steps from your workflow"}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0,
                       "tooltip": "CFG passes through unchanged"}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True,
                           "tooltip": "ON = fast testing mode, OFF = original values"}),
                "target_resolution": ("INT", {"default": 512, "min": 256, "max": 1024,
                                     "tooltip": "Max resolution during speed test"}),
                "target_steps": ("INT", {"default": 15, "min": 4, "max": 50,
                                "tooltip": "Max steps during speed test"}),
            }
        }

    def apply(self, width, height, steps, cfg, enabled=True, target_resolution=512, target_steps=15):
        if not enabled:
            return (width, height, steps, cfg)

        # Cap resolution
        if width > target_resolution or height > target_resolution:
            scale = target_resolution / max(width, height)
            width = int(width * scale) // 8 * 8
            height = int(height * scale) // 8 * 8

        # Cap steps
        steps = min(steps, target_steps)

        print(f"[Performance Lab] Speed preset applied: {width}x{height}, {steps} steps")
        return (width, height, steps, cfg)


class PerfLab_LowVRAMPreset:
    """Low VRAM Preset - Optimized for 6GB/8GB/12GB GPUs."""

    DESCRIPTION = """💾 LOW VRAM PRESET

HOW TO USE:
1. Connect width, height, steps, batch_size
2. Select your GPU's VRAM from dropdown
3. Connect outputs to your workflow

VRAM PRESETS:
• 6GB: 512px max, 20 steps, batch=1
• 8GB: 768px max, 25 steps, batch=1
• 12GB: 1024px max, 30 steps, batch=1

OUTPUTS:
• width/height: VRAM-safe dimensions
• steps: Reasonable step count
• batch_size: Always 1 for safety

USE WHEN:
• Getting CUDA out of memory errors
• Running SDXL on 8GB GPU
• Want safe defaults for your card"""

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "apply"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "steps", "batch_size")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192,
                         "tooltip": "Original width from your workflow"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192,
                          "tooltip": "Original height from your workflow"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200,
                         "tooltip": "Original steps from your workflow"}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64,
                              "tooltip": "Original batch size from your workflow"}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True,
                           "tooltip": "ON = apply VRAM limits, OFF = use original values"}),
                "vram_target": (["6GB", "8GB", "12GB"],
                               {"tooltip": "Select your GPU's VRAM capacity"}),
            }
        }

    VRAM_SETTINGS = {
        "6GB": {"max_res": 512, "max_steps": 20},
        "8GB": {"max_res": 768, "max_steps": 25},
        "12GB": {"max_res": 1024, "max_steps": 30},
    }

    def apply(self, width, height, steps, batch_size, enabled=True, vram_target="8GB"):
        if not enabled:
            return (width, height, steps, batch_size)

        settings = self.VRAM_SETTINGS.get(vram_target, self.VRAM_SETTINGS["8GB"])
        max_res = settings["max_res"]
        max_steps = settings["max_steps"]

        # Cap resolution
        if width > max_res or height > max_res:
            scale = max_res / max(width, height)
            width = int(width * scale) // 8 * 8
            height = int(height * scale) // 8 * 8

        # Cap steps and force batch=1
        steps = min(steps, max_steps)
        batch_size = 1

        print(f"[Performance Lab] {vram_target} preset: {width}x{height}, {steps} steps, batch=1")
        return (width, height, steps, batch_size)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS & DIAGNOSTIC NODES
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_Analyzer:
    """Workflow Analyzer - Analyze workflow and get suggestions."""

    DESCRIPTION = """🔍 WORKFLOW ANALYZER

HOW TO USE:
1. Export your workflow as JSON (Ctrl+S in ComfyUI)
2. Paste the JSON into this node
3. Read the analysis and suggestions

OUTPUTS:
• analysis: Node counts, features detected
• suggestions: Specific optimization tips

DETECTS:
• Upscaling (ESRGAN, etc.)
• ControlNet usage
• Model type (SDXL, Flux, etc.)
• Video generation nodes

USE WHEN:
• Starting optimization on a new workflow
• Want to understand workflow complexity
• Need suggestions for what to optimize"""

    CATEGORY = "⚡ Performance Lab/Analysis"
    FUNCTION = "analyze"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("analysis", "suggestions")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json": ("STRING", {
                    "multiline": True,
                    "default": '{"nodes": [], "links": []}'
                }),
            }
        }

    def analyze(self, workflow_json: str):
        try:
            workflow = json.loads(workflow_json)
            nodes = workflow.get("nodes", [])
            links = workflow.get("links", [])

            # Count node types
            node_types = {}
            for node in nodes:
                ntype = node.get("type", "Unknown")
                node_types[ntype] = node_types.get(ntype, 0) + 1

            # Detect features and issues
            features = []
            suggestions = []
            node_str = " ".join(node_types.keys()).lower()

            # Feature detection
            if "upscale" in node_str or "esrgan" in node_str:
                features.append("Upscaling")
                suggestions.append("💡 Bypass upscalers during testing to save 2-4GB VRAM")

            if "controlnet" in node_str:
                features.append("ControlNet")
                suggestions.append("💡 ControlNet adds VRAM overhead - disable during initial tests")

            if "sdxl" in node_str or "xl" in node_str:
                features.append("SDXL")
                suggestions.append("💡 SDXL uses ~2x VRAM vs SD1.5 - use 1024x1024 max")

            if "flux" in node_str:
                features.append("Flux")
                suggestions.append("💡 Flux works best with CFG 1-3.5")

            if "video" in node_str or "animatediff" in node_str:
                features.append("Video")
                suggestions.append("💡 Video generation is very VRAM intensive - reduce frame count")

            # Build analysis
            analysis_lines = [
                "═══ Workflow Analysis ═══",
                f"📊 Total Nodes: {len(nodes)}",
                f"🔗 Total Links: {len(links)}",
                f"✨ Features: {', '.join(features) if features else 'Basic SD'}",
                "",
                "📋 Node Types:",
            ]

            for ntype, count in sorted(node_types.items(), key=lambda x: -x[1])[:10]:
                analysis_lines.append(f"   • {ntype}: {count}")

            analysis = "\n".join(analysis_lines)
            suggestions_text = "\n".join(suggestions) if suggestions else "✅ No obvious issues found"

            print(f"\n[Performance Lab]\n{analysis}\n\n{suggestions_text}\n")
            return (analysis, suggestions_text)

        except json.JSONDecodeError:
            return ("❌ Invalid JSON", "Paste valid workflow JSON")
        except Exception as e:
            return (f"❌ Error: {e}", "")


class PerfLab_BlackImageFix:
    """Black Image Fix - Diagnose why you're getting dark images."""

    DESCRIPTION = """🔧 BLACK IMAGE FIX

HOW TO USE:
1. Select your model type from dropdown
2. Enter your current CFG and steps
3. Read diagnosis and use suggested values

OUTPUTS:
• diagnosis: What might be wrong
• suggested_cfg: Optimal CFG for your model
• suggested_steps: Minimum recommended steps

COMMON CAUSES OF BLACK IMAGES:
• Flux with CFG > 4 (use 1-3.5)
• SD with CFG < 3 (use 5-8)
• Too few steps (< 15)
• Missing/wrong VAE
• Empty prompt

FIX IT:
Connect suggested_cfg and suggested_steps
to your KSampler to try the fix!"""

    CATEGORY = "⚡ Performance Lab/Analysis"
    FUNCTION = "diagnose"
    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("diagnosis", "suggested_cfg", "suggested_steps")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["SD 1.5", "SDXL", "SD3", "Flux Dev", "Flux Schnell", "Unknown"],),
                "current_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
                "current_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
            },
        }

    # Optimal settings per model
    MODEL_SETTINGS = {
        "SD 1.5": {"cfg": 7.5, "min_steps": 20, "resolution": 512},
        "SDXL": {"cfg": 7.0, "min_steps": 25, "resolution": 1024},
        "SD3": {"cfg": 4.5, "min_steps": 25, "resolution": 1024},
        "Flux Dev": {"cfg": 3.5, "min_steps": 25, "resolution": 1024},
        "Flux Schnell": {"cfg": 1.0, "min_steps": 4, "resolution": 1024},
        "Unknown": {"cfg": 7.0, "min_steps": 20, "resolution": 768},
    }

    def diagnose(self, model_type: str, current_cfg: float, current_steps: int):
        settings = self.MODEL_SETTINGS.get(model_type, self.MODEL_SETTINGS["Unknown"])
        issues = []

        # Check CFG
        optimal_cfg = settings["cfg"]
        if model_type in ["Flux Dev", "Flux Schnell"] and current_cfg > 4:
            issues.append(f"⚠️ CFG too high for Flux! ({current_cfg} → use {optimal_cfg})")
        elif model_type not in ["Flux Dev", "Flux Schnell"] and current_cfg < 3:
            issues.append(f"⚠️ CFG too low for {model_type}! ({current_cfg} → use {optimal_cfg})")

        # Check steps
        min_steps = settings["min_steps"]
        if current_steps < min_steps:
            issues.append(f"⚠️ Steps too low! ({current_steps} → use at least {min_steps})")

        # General tips
        tips = [
            "",
            "📋 Other common fixes:",
            "• Add a VAE Loader if using checkpoint's built-in VAE",
            "• Check all node connections are intact",
            "• Verify prompt is not empty",
            "• Try a different sampler (euler, dpmpp_2m)",
        ]

        if issues:
            diagnosis = "\n".join(["🔧 Black Image Diagnostic", ""] + issues + tips)
        else:
            diagnosis = f"✅ Settings look OK for {model_type}\n" + "\n".join(tips)

        print(f"\n[Performance Lab]\n{diagnosis}\n")
        return (diagnosis, optimal_cfg, max(current_steps, min_steps))


class PerfLab_Compare:
    """Compare Results - See before/after improvement."""

    DESCRIPTION = """📊 COMPARE RESULTS

HOW TO USE:
1. Run workflow BEFORE optimization, note duration/VRAM
2. Run workflow AFTER optimization
3. Enter both values to see % improvement

INPUTS:
• before_duration: Time before optimization
• after_duration: Time after optimization
• before_vram: Peak VRAM before (optional)
• after_vram: Peak VRAM after (optional)

OUTPUT:
• comparison: Table showing changes and % difference

TIPS:
• Get duration/VRAM from Performance Report node
• Green = improvement, Red = regression
• Use to verify optimizations actually helped"""

    CATEGORY = "⚡ Performance Lab/Analysis"
    FUNCTION = "compare"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before_duration": ("FLOAT", {"default": 0.0}),
                "after_duration": ("FLOAT", {"default": 0.0}),
            },
            "optional": {
                "before_vram": ("FLOAT", {"default": 0.0}),
                "after_vram": ("FLOAT", {"default": 0.0}),
                "before_label": ("STRING", {"default": "Before"}),
                "after_label": ("STRING", {"default": "After"}),
            }
        }

    def compare(self, before_duration, after_duration,
                before_vram=0.0, after_vram=0.0,
                before_label="Before", after_label="After"):

        # Calculate changes
        if before_duration > 0:
            time_change = ((after_duration - before_duration) / before_duration) * 100
            time_str = f"{time_change:+.1f}%"
            time_verdict = "🟢 Faster!" if time_change < 0 else "🔴 Slower"
        else:
            time_str = "N/A"
            time_verdict = ""

        if before_vram > 0:
            vram_change = ((after_vram - before_vram) / before_vram) * 100
            vram_str = f"{vram_change:+.1f}%"
            vram_verdict = "🟢 Less VRAM" if vram_change < 0 else "🔴 More VRAM"
        else:
            vram_str = "N/A"
            vram_verdict = ""

        comparison = f"""═══ Comparison ═══

┌─────────────┬─────────────┬─────────────┬──────────┐
│   Metric    │  {before_label:^9}  │  {after_label:^9}  │  Change  │
├─────────────┼─────────────┼─────────────┼──────────┤
│ Duration    │  {before_duration:>7.2f}s   │  {after_duration:>7.2f}s   │ {time_str:>8} │
│ Peak VRAM   │  {before_vram:>6.2f} GB  │  {after_vram:>6.2f} GB  │ {vram_str:>8} │
└─────────────┴─────────────┴─────────────┴──────────┘

{time_verdict} {vram_verdict}
"""
        print(f"\n[Performance Lab]\n{comparison}")
        return (comparison,)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM INTEGRATION NODES
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_GeneratePrompt:
    """Generate LLM Prompt - Get AI help from Claude/GPT/Gemini."""

    DESCRIPTION = """🤖 GENERATE LLM PROMPT

HOW TO USE:
1. Enter your optimization goal
2. Add current performance metrics (optional)
3. Copy the output prompt
4. Paste into Claude, ChatGPT, or Gemini
5. Follow the AI's suggestions!

INPUTS:
• goal: What you want to achieve
• current_duration: From Performance Report
• current_vram: From Performance Report
• model_type: SD 1.5, SDXL, Flux, etc.
• workflow_json: Paste workflow for analysis

OUTPUT:
• llm_prompt: Ready to paste into any AI

EXAMPLE GOALS:
• "Make this run under 10 seconds"
• "Fit on 8GB VRAM"
• "Fix why I get black images"
• "Improve quality without slowing down\""""

    CATEGORY = "⚡ Performance Lab/LLM"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("llm_prompt",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "goal": ("STRING", {
                    "multiline": True,
                    "default": "Make this workflow faster and use less VRAM"
                }),
            },
            "optional": {
                "current_duration": ("FLOAT", {"default": 0.0}),
                "current_vram": ("FLOAT", {"default": 0.0}),
                "model_type": (["SD 1.5", "SDXL", "SD3", "Flux", "Unknown"],),
                "workflow_json": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def generate(self, goal, current_duration=0.0, current_vram=0.0,
                 model_type="Unknown", workflow_json=""):

        prompt_parts = [
            "I need help optimizing my ComfyUI workflow.",
            "",
            f"**My Goal:** {goal}",
            "",
        ]

        if current_duration > 0:
            prompt_parts.append(f"**Current Duration:** {current_duration:.2f} seconds")
        if current_vram > 0:
            prompt_parts.append(f"**Current VRAM Usage:** {current_vram:.2f} GB")
        if model_type != "Unknown":
            prompt_parts.append(f"**Model Type:** {model_type}")

        prompt_parts.extend([
            "",
            "Please suggest specific changes I can make to improve performance.",
            "Focus on:",
            "- Resolution adjustments",
            "- Step count optimization",
            "- Sampler selection",
            "- CFG scale",
            "- Any nodes I could bypass or optimize",
            "",
            "Provide specific values I should use.",
        ])

        if workflow_json and workflow_json.strip().startswith("{"):
            prompt_parts.extend([
                "",
                "**Workflow JSON:**",
                "```json",
                workflow_json[:2000] + ("..." if len(workflow_json) > 2000 else ""),
                "```"
            ])

        prompt = "\n".join(prompt_parts)
        print(f"\n[Performance Lab] LLM Prompt Generated ({len(prompt)} chars)\n")
        print("Copy this prompt and paste it into Claude, ChatGPT, or Gemini:\n")
        print("-" * 60)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 60)

        return (prompt,)


class PerfLab_LLMClient:
    """LLM Client - Call Ollama, OpenAI, or compatible APIs directly."""

    DESCRIPTION = """🤖 LLM CLIENT - CALL AI DIRECTLY

HOW TO USE:
1. Select your LLM provider (Ollama is free & local!)
2. Enter your prompt or connect from Generate Prompt
3. Get AI response directly in ComfyUI

PROVIDERS:
• Ollama (FREE, LOCAL): Install from ollama.com, run locally
• OpenAI: Requires API key
• Anthropic: Requires API key
• Custom: Any OpenAI-compatible API

DEFAULT (Ollama):
• URL: http://127.0.0.1:11434
• Model: llama3.2 (or mistral, codellama, etc.)

OUTPUTS:
• response: The LLM's response text
• success: Whether the call succeeded"""

    CATEGORY = "⚡ Performance Lab/LLM"
    FUNCTION = "call_llm"
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("response", "success", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Analyze this ComfyUI workflow and suggest optimizations.",
                    "tooltip": "The prompt to send to the LLM"
                }),
                "provider": (["Ollama (Local)", "OpenAI", "Anthropic", "Custom"],
                            {"default": "Ollama (Local)",
                             "tooltip": "Ollama is FREE and runs locally!"}),
            },
            "optional": {
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "API URL (Ollama: 11434, OpenAI: api.openai.com)"
                }),
                "model": ("STRING", {
                    "default": "llama3.2",
                    "tooltip": "Model name (Ollama: llama3.2, mistral, codellama)"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key (not needed for Ollama)"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a ComfyUI workflow optimization expert. Provide specific, actionable suggestions.",
                    "tooltip": "System prompt to set AI behavior"
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 100,
                    "max": 8192,
                    "tooltip": "Maximum response length"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Creativity (0=focused, 1+=creative)"
                }),
            }
        }

    def call_llm(self, prompt, provider, api_url="http://127.0.0.1:11434",
                 model="llama3.2", api_key="", system_prompt="",
                 max_tokens=2048, temperature=0.7):

        import urllib.request
        import urllib.error

        try:
            if provider == "Ollama (Local)":
                return self._call_ollama(api_url, model, prompt, system_prompt, max_tokens, temperature)
            elif provider == "OpenAI":
                return self._call_openai(api_key, model, prompt, system_prompt, max_tokens, temperature)
            elif provider == "Anthropic":
                return self._call_anthropic(api_key, model, prompt, system_prompt, max_tokens, temperature)
            else:  # Custom - use OpenAI format
                return self._call_openai_format(api_url, api_key, model, prompt, system_prompt, max_tokens, temperature)
        except Exception as e:
            error_msg = f"❌ LLM Error: {str(e)}"
            print(f"[Performance Lab] {error_msg}")
            return ("", False, error_msg)

    def _call_ollama(self, api_url, model, prompt, system_prompt, max_tokens, temperature):
        """Call Ollama API (local, free)."""
        import urllib.request

        url = f"{api_url.rstrip('/')}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result.get("response", "")
                print(f"[Performance Lab] Ollama response received ({len(text)} chars)")
                return (text, True, f"✅ Ollama ({model}): {len(text)} chars")
        except urllib.error.URLError as e:
            return ("", False, f"❌ Cannot connect to Ollama at {api_url}. Is it running?")

    def _call_openai(self, api_key, model, prompt, system_prompt, max_tokens, temperature):
        """Call OpenAI API."""
        return self._call_openai_format(
            "https://api.openai.com/v1", api_key,
            model or "gpt-4o-mini", prompt, system_prompt, max_tokens, temperature
        )

    def _call_anthropic(self, api_key, model, prompt, system_prompt, max_tokens, temperature):
        """Call Anthropic API."""
        import urllib.request

        url = "https://api.anthropic.com/v1/messages"

        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": model or "claude-3-haiku-20240307",
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers)

        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get("content", [{}])[0].get("text", "")
            return (text, True, f"✅ Anthropic ({model}): {len(text)} chars")

    def _call_openai_format(self, api_url, api_key, model, prompt, system_prompt, max_tokens, temperature):
        """Call OpenAI-compatible API."""
        import urllib.request

        url = f"{api_url.rstrip('/')}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers)

        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return (text, True, f"✅ {model}: {len(text)} chars")


class PerfLab_AutoOptimize:
    """Auto Optimize - Send performance data to LLM and get optimization suggestions."""

    DESCRIPTION = """🚀 AUTO OPTIMIZE - AI-POWERED SUGGESTIONS

THE AUTOMATED OPTIMIZATION LOOP!

HOW TO USE:
1. Connect your performance report
2. Select optimization goal (speed/vram/quality)
3. Run to get AI suggestions
4. Apply suggestions and test
5. Keep or reject based on results

INPUTS:
• performance_report: From Performance Report node
• goal: What to optimize for
• current values: Your current settings

OUTPUTS:
• suggested_steps: Recommended step count
• suggested_resolution: Recommended resolution
• suggested_cfg: Recommended CFG
• explanation: Why these changes help
• raw_response: Full LLM response

WORKFLOW:
Run → Get Suggestions → Apply → Test → Keep/Reject → Repeat!"""

    CATEGORY = "⚡ Performance Lab/LLM"
    FUNCTION = "optimize"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("suggested_steps", "suggested_resolution", "suggested_cfg", "explanation", "raw_response")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "goal": (["speed", "vram", "quality", "balanced"],
                        {"default": "speed",
                         "tooltip": "What to optimize for"}),
            },
            "optional": {
                "performance_report": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect from Performance Report node"
                }),
                "current_steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 150,
                    "tooltip": "Your current step count"
                }),
                "current_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "tooltip": "Your current resolution"
                }),
                "current_cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 30.0,
                    "tooltip": "Your current CFG"
                }),
                "model_type": (["SD 1.5", "SDXL", "SD3", "Flux Dev", "Flux Schnell", "Unknown"],
                              {"default": "Unknown",
                               "tooltip": "Your model type (affects recommendations)"}),
                "ollama_url": ("STRING", {
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "Ollama API URL"
                }),
                "ollama_model": ("STRING", {
                    "default": "llama3.2",
                    "tooltip": "Ollama model to use"
                }),
            }
        }

    def optimize(self, goal, performance_report="", current_steps=30,
                 current_resolution=1024, current_cfg=7.0, model_type="Unknown",
                 ollama_url="http://127.0.0.1:11434", ollama_model="llama3.2"):

        # Build the optimization prompt
        prompt = f"""Analyze this ComfyUI workflow performance and suggest optimizations.

GOAL: {goal.upper()}

CURRENT SETTINGS:
- Steps: {current_steps}
- Resolution: {current_resolution}
- CFG: {current_cfg}
- Model Type: {model_type}

PERFORMANCE DATA:
{performance_report if performance_report else "No performance data yet"}

Based on the goal of "{goal}", suggest SPECIFIC numeric values for:
1. Steps (integer between 4-100)
2. Resolution (integer, multiple of 8, between 256-2048)
3. CFG (float between 0.5-20)

IMPORTANT: Respond in this EXACT format:
STEPS: [number]
RESOLUTION: [number]
CFG: [number]
EXPLANATION: [brief explanation why]

Example:
STEPS: 20
RESOLUTION: 768
CFG: 7.0
EXPLANATION: Reducing steps from 30 to 20 saves ~33% time with minimal quality loss. 768px is optimal for 8GB VRAM.
"""

        system_prompt = """You are a ComfyUI optimization expert. Given performance metrics and a goal, suggest specific numeric parameter changes.

Key optimization rules:
- For SPEED: Reduce steps (15-25 usually enough), lower resolution
- For VRAM: Reduce resolution first (has quadratic effect), then batch size, use tiled VAE
- For QUALITY: More steps (30-50), native resolution, appropriate CFG for model
- Flux models need CFG 1-4, SD models need CFG 5-8
- SD1.5 native: 512px, SDXL/Flux native: 1024px

Always respond with EXACT numbers in the format requested."""

        # Call Ollama
        try:
            import urllib.request

            url = f"{ollama_url.rstrip('/')}/api/generate"
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 500}
            }

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                llm_response = result.get("response", "")

            # Parse the response
            suggested_steps = current_steps
            suggested_resolution = current_resolution
            suggested_cfg = current_cfg
            explanation = "Could not parse LLM response"

            lines = llm_response.upper().split('\n')
            for line in lines:
                if line.startswith('STEPS:'):
                    try:
                        suggested_steps = int(''.join(filter(str.isdigit, line.split(':')[1])))
                    except:
                        pass
                elif line.startswith('RESOLUTION:'):
                    try:
                        suggested_resolution = int(''.join(filter(str.isdigit, line.split(':')[1])))
                    except:
                        pass
                elif line.startswith('CFG:'):
                    try:
                        cfg_str = line.split(':')[1].strip()
                        suggested_cfg = float(''.join(c for c in cfg_str if c.isdigit() or c == '.'))
                    except:
                        pass

            # Extract explanation
            if 'EXPLANATION:' in llm_response.upper():
                explanation = llm_response.split('EXPLANATION:')[-1].split('xplanation:')[-1].strip()

            # Clamp values to safe ranges
            suggested_steps = max(4, min(100, suggested_steps))
            suggested_resolution = max(256, min(2048, (suggested_resolution // 8) * 8))
            suggested_cfg = max(0.5, min(20.0, suggested_cfg))

            print(f"[Performance Lab] AutoOptimize suggestions:")
            print(f"  Steps: {current_steps} → {suggested_steps}")
            print(f"  Resolution: {current_resolution} → {suggested_resolution}")
            print(f"  CFG: {current_cfg} → {suggested_cfg}")

            return (suggested_steps, suggested_resolution, suggested_cfg, explanation, llm_response)

        except urllib.error.URLError:
            explanation = "❌ Cannot connect to Ollama. Make sure it's running (ollama serve)"
            return (current_steps, current_resolution, current_cfg, explanation, "Connection failed")
        except Exception as e:
            explanation = f"❌ Error: {str(e)}"
            return (current_steps, current_resolution, current_cfg, explanation, str(e))


class PerfLab_ApplyOrRevert:
    """Apply or Revert - Compare results and decide to keep or reject changes."""

    DESCRIPTION = """✅❌ APPLY OR REVERT

COMPARE BEFORE/AFTER AND DECIDE!

HOW TO USE:
1. Connect your BEFORE results (original)
2. Connect your AFTER results (with optimization)
3. Set keep_changes based on results
4. Output goes to your workflow

This is the DECISION node in the optimization loop:
• If AFTER is better → keep_changes = TRUE → use new values
• If AFTER is worse → keep_changes = FALSE → use original values

OUTPUTS:
• The values based on your decision
• comparison: Shows the improvement/regression"""

    CATEGORY = "⚡ Performance Lab/LLM"
    FUNCTION = "decide"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("steps", "resolution", "cfg", "comparison")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keep_changes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "TRUE = use new values, FALSE = use original"
                }),
                "original_steps": ("INT", {"default": 30, "min": 1, "max": 150}),
                "original_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "original_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
                "new_steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "new_resolution": ("INT", {"default": 768, "min": 256, "max": 4096}),
                "new_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
            },
            "optional": {
                "before_duration": ("FLOAT", {"default": 0.0, "tooltip": "Duration before optimization"}),
                "after_duration": ("FLOAT", {"default": 0.0, "tooltip": "Duration after optimization"}),
                "before_vram": ("FLOAT", {"default": 0.0, "tooltip": "VRAM before"}),
                "after_vram": ("FLOAT", {"default": 0.0, "tooltip": "VRAM after"}),
            }
        }

    def decide(self, keep_changes, original_steps, original_resolution, original_cfg,
               new_steps, new_resolution, new_cfg,
               before_duration=0.0, after_duration=0.0, before_vram=0.0, after_vram=0.0):

        # Calculate changes
        if before_duration > 0 and after_duration > 0:
            speed_change = ((after_duration - before_duration) / before_duration) * 100
            speed_str = f"{speed_change:+.1f}%"
            speed_verdict = "🟢 Faster!" if speed_change < 0 else "🔴 Slower"
        else:
            speed_str = "N/A"
            speed_verdict = ""

        if before_vram > 0 and after_vram > 0:
            vram_change = ((after_vram - before_vram) / before_vram) * 100
            vram_str = f"{vram_change:+.1f}%"
            vram_verdict = "🟢 Less VRAM" if vram_change < 0 else "🔴 More VRAM"
        else:
            vram_str = "N/A"
            vram_verdict = ""

        decision = "✅ KEEPING NEW VALUES" if keep_changes else "❌ REVERTING TO ORIGINAL"

        comparison = f"""═══ Optimization Decision ═══

{decision}

PERFORMANCE COMPARISON:
• Duration: {before_duration:.2f}s → {after_duration:.2f}s ({speed_str}) {speed_verdict}
• VRAM: {before_vram:.2f}GB → {after_vram:.2f}GB ({vram_str}) {vram_verdict}

SETTINGS:
• Steps: {original_steps} → {new_steps}
• Resolution: {original_resolution} → {new_resolution}
• CFG: {original_cfg} → {new_cfg}

OUTPUT: {'New values' if keep_changes else 'Original values'}"""

        print(f"[Performance Lab] {decision}")

        if keep_changes:
            return (new_steps, new_resolution, new_cfg, comparison)
        else:
            return (original_steps, original_resolution, original_cfg, comparison)


class PerfLab_ShowText:
    """Show Text - Display any text in the node preview."""

    DESCRIPTION = """📝 SHOW TEXT

HOW TO USE:
1. Connect any STRING output to this node
2. The text appears in both the node and console

USE FOR:
• Viewing analysis results
• Checking LLM prompts before copying
• Debugging string outputs
• Displaying reports

TIPS:
• Great for inspecting what's in a string
• Connect to Performance Report output
• Connect to Analyzer suggestions output"""

    CATEGORY = "⚡ Performance Lab/Utility"
    FUNCTION = "show"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    def show(self, text):
        print(f"\n[Performance Lab]\n{text}\n")
        return (text,)


class PerfLab_Switch:
    """A/B Switch - Toggle between two any-type inputs."""

    DESCRIPTION = """🔀 A/B SWITCH

HOW TO USE:
1. Connect your TEST value to 'a_test'
2. Connect your PRODUCTION value to 'b_production'
3. Toggle 'use_b' to switch between them

INPUTS:
• a_test: Your optimized/test configuration
• b_production: Your original/production configuration
• use_b: Toggle (OFF = test, ON = production)

USE FOR:
• Comparing before/after settings
• Quick test mode vs production mode toggle
• Switching between different node chains
• A/B testing different approaches

TIPS:
• Works with ANY type (images, latents, models, etc.)
• Use with Int/Float Switch for typed values"""

    CATEGORY = "⚡ Performance Lab/Utility"
    FUNCTION = "switch"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_b": ("BOOLEAN", {"default": False,
                         "tooltip": "OFF = use A (test), ON = use B (production)"}),
            },
            "optional": {
                "a_test": ("*",),
                "b_production": ("*",),
            }
        }

    def switch(self, use_b, a_test=None, b_production=None):
        if use_b:
            return (b_production,)
        return (a_test,)


class PerfLab_IntSwitch:
    """Integer A/B Switch - Toggle between two integer values."""

    DESCRIPTION = """🔢 INTEGER A/B SWITCH

HOW TO USE:
1. Enter your TEST value in 'a_test' (e.g., 15 steps)
2. Enter your PRODUCTION value in 'b_production' (e.g., 30 steps)
3. Toggle 'use_b' to switch between them

INPUTS:
• a_test: Test/fast integer value
• b_production: Production/quality integer value
• use_b: Toggle (OFF = test, ON = production)

USE FOR:
• Switching step counts (15 vs 30)
• Switching resolutions (512 vs 1024)
• Switching batch sizes (1 vs 4)
• Any integer A/B comparison

EXAMPLE:
• a_test: 15 (fast testing)
• b_production: 30 (final quality)
• Connect output to KSampler steps"""

    CATEGORY = "⚡ Performance Lab/Utility"
    FUNCTION = "switch"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a_test": ("INT", {"default": 15, "min": 0, "max": 10000}),
                "b_production": ("INT", {"default": 30, "min": 0, "max": 10000}),
                "use_b": ("BOOLEAN", {"default": False}),
            }
        }

    def switch(self, a_test, b_production, use_b):
        return (b_production if use_b else a_test,)


class PerfLab_FloatSwitch:
    """Float A/B Switch - Toggle between two decimal values."""

    DESCRIPTION = """🔢 FLOAT A/B SWITCH

HOW TO USE:
1. Enter your TEST value in 'a_test' (e.g., 1.0 CFG)
2. Enter your PRODUCTION value in 'b_production' (e.g., 7.0 CFG)
3. Toggle 'use_b' to switch between them

INPUTS:
• a_test: Test float value
• b_production: Production float value
• use_b: Toggle (OFF = test, ON = production)

USE FOR:
• Switching CFG scale (1.0 vs 7.0)
• Switching denoise strength (0.5 vs 1.0)
• Any decimal A/B comparison

EXAMPLE (Flux vs SD):
• a_test: 1.0 (Flux CFG)
• b_production: 7.0 (SD CFG)
• Toggle based on which model you're using"""

    CATEGORY = "⚡ Performance Lab/Utility"
    FUNCTION = "switch"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a_test": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "b_production": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "use_b": ("BOOLEAN", {"default": False}),
            }
        }

    def switch(self, a_test, b_production, use_b):
        return (b_production if use_b else a_test,)


# ═══════════════════════════════════════════════════════════════════════════════
# META-WORKFLOW NODES (Run & Test Other Workflows)
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_LoadWorkflow:
    """Load Workflow - Load a workflow JSON file for analysis or execution."""

    DESCRIPTION = """📂 LOAD WORKFLOW

HOW TO USE:
1. Select a workflow from dropdown OR enter custom path
2. The workflow is loaded and output as JSON string
3. Connect to Analyzer or Queue Workflow node

INPUTS:
• workflow_folder: Select from common ComfyUI folders
• file_name: Select workflow or enter filename
• custom_path: (Optional) Full path for files elsewhere

OUTPUTS:
• workflow_json: The loaded workflow as JSON string
• node_count: Number of nodes in the workflow
• status: Success/error message

USE FOR:
• Loading workflows for analysis
• Batch testing multiple workflows
• Building a workflow test suite

TIPS:
• Leave custom_path empty to use folder selection
• Workflows are auto-discovered from ComfyUI folders
• Connect to Queue Workflow to run it"""

    CATEGORY = "⚡ Performance Lab/Meta-Workflow"
    FUNCTION = "load"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("workflow_json", "node_count", "status")

    # Common workflow locations relative to ComfyUI
    WORKFLOW_FOLDERS = [
        "user/default/workflows",
        "output",
        "input",
        "custom_nodes/ComfyUI_PerformanceLab/examples",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_folder": (["user/default/workflows", "output", "input", "examples", "custom_path"],
                                   {"default": "user/default/workflows",
                                    "tooltip": "Select folder or use 'custom_path' for manual entry"}),
                "file_name": ("STRING", {
                    "default": "workflow.json",
                    "tooltip": "Workflow filename (with .json extension)"
                }),
            },
            "optional": {
                "custom_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full path (only used when folder is set to 'custom_path')"
                }),
            }
        }

    def load(self, workflow_folder: str, file_name: str, custom_path: str = ""):
        # Determine the file path
        if workflow_folder == "custom_path":
            if not custom_path:
                return ("", 0, "❌ custom_path selected but no path provided")
            file_path = custom_path
        else:
            # Build path relative to ComfyUI installation
            # Try to find ComfyUI root
            comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(MODULE_DIR)))

            if workflow_folder == "examples":
                # Our examples folder
                file_path = os.path.join(MODULE_DIR, "examples", file_name)
            else:
                file_path = os.path.join(comfy_root, workflow_folder, file_name)

        if not file_path:
            return ("", 0, "❌ No file path provided")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            workflow = json.loads(content)
            nodes = workflow.get("nodes", [])
            node_count = len(nodes)

            print(f"[Performance Lab] Loaded workflow: {file_path}")
            print(f"   Nodes: {node_count}")

            return (content, node_count, f"✅ Loaded {node_count} nodes from {os.path.basename(file_path)}")

        except FileNotFoundError:
            return ("", 0, f"❌ File not found: {file_path}")
        except json.JSONDecodeError as e:
            return ("", 0, f"❌ Invalid JSON: {e}")
        except Exception as e:
            return ("", 0, f"❌ Error: {e}")


class PerfLab_QueueWorkflow:
    """Queue Workflow - Send a workflow to ComfyUI for execution."""

    DESCRIPTION = """▶️ QUEUE WORKFLOW

HOW TO USE:
1. Connect workflow_json from Load Workflow node
2. Set your ComfyUI server URL (default: localhost:8188)
3. Toggle 'execute' to True to run
4. The workflow will be queued on ComfyUI

INPUTS:
• workflow_json: The workflow JSON to execute
• server_url: ComfyUI server address
• execute: Safety toggle - must be True to run
• client_id: Optional ID for tracking

OUTPUTS:
• prompt_id: The queued prompt ID
• status: Success/error message

USE FOR:
• Running test workflows automatically
• Batch testing multiple workflows
• Remote workflow execution
• Performance benchmarking

TIPS:
• Make sure ComfyUI is running first
• The 'execute' toggle prevents accidents
• Works with remote ComfyUI servers too"""

    CATEGORY = "⚡ Performance Lab/Meta-Workflow"
    FUNCTION = "queue"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_id", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json": ("STRING", {"forceInput": True}),
                "server_url": ("STRING", {"default": "http://127.0.0.1:8188"}),
                "execute": ("BOOLEAN", {"default": False,
                           "tooltip": "Toggle ON to actually queue the workflow"}),
            },
            "optional": {
                "client_id": ("STRING", {"default": "perflab"}),
            }
        }

    def queue(self, workflow_json: str, server_url: str, execute: bool, client_id: str = "perflab"):
        if not execute:
            return ("", "⏸️ Execute is OFF - toggle to True to run")

        if not workflow_json:
            return ("", "❌ No workflow provided")

        try:
            import urllib.request
            import urllib.error

            # Parse workflow
            workflow = json.loads(workflow_json)

            # Prepare prompt payload
            prompt_payload = {
                "prompt": workflow,
                "client_id": client_id
            }

            # Send to ComfyUI
            url = f"{server_url.rstrip('/')}/prompt"
            data = json.dumps(prompt_payload).encode('utf-8')

            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                prompt_id = result.get("prompt_id", "unknown")

                print(f"[Performance Lab] Workflow queued!")
                print(f"   Prompt ID: {prompt_id}")
                print(f"   Server: {server_url}")

                return (prompt_id, f"✅ Queued: {prompt_id}")

        except urllib.error.URLError as e:
            return ("", f"❌ Connection failed: {e}")
        except json.JSONDecodeError:
            return ("", "❌ Invalid workflow JSON")
        except Exception as e:
            return ("", f"❌ Error: {e}")


class PerfLab_EndpointHealth:
    """Endpoint Health Check - Check if a network service is running."""

    DESCRIPTION = """🏥 ENDPOINT HEALTH CHECK

HOW TO USE:
1. Enter the URL of your service (e.g., http://localhost:7860)
2. Select the service type for correct health check
3. Run to check if it's online and responding

INPUTS:
• url: Full URL to the service
• service_type: Type of service (ComfyUI, Automatic1111, Ollama, etc.)
• timeout: How long to wait (seconds)

OUTPUTS:
• is_healthy: Boolean - True if service is up
• latency_ms: Response time in milliseconds
• status: Detailed status message

SUPPORTED SERVICES:
• ComfyUI (port 8188)
• Automatic1111/Forge (port 7860)
• Ollama (port 11434)
• KoboldCpp (port 5001)
• Custom (any HTTP endpoint)

USE FOR:
• Checking if services are online before running
• Measuring network latency
• Distributed workflow health monitoring"""

    CATEGORY = "⚡ Performance Lab/Network"
    FUNCTION = "check"
    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("is_healthy", "latency_ms", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://127.0.0.1:8188"}),
                "service_type": (["ComfyUI", "Automatic1111", "Ollama", "KoboldCpp", "Whisper", "TTS", "Custom"],),
            },
            "optional": {
                "timeout": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0}),
            }
        }

    # Health check endpoints per service type
    HEALTH_ENDPOINTS = {
        "ComfyUI": "/system_stats",
        "Automatic1111": "/sdapi/v1/options",
        "Ollama": "/api/tags",
        "KoboldCpp": "/api/v1/info/version",
        "Whisper": "/",
        "TTS": "/",
        "Custom": "/",
    }

    def check(self, url: str, service_type: str, timeout: float = 5.0):
        import urllib.request
        import urllib.error

        endpoint = self.HEALTH_ENDPOINTS.get(service_type, "/")
        full_url = f"{url.rstrip('/')}{endpoint}"

        start_time = time.time()

        try:
            req = urllib.request.Request(full_url, method='GET')
            with urllib.request.urlopen(req, timeout=timeout) as response:
                latency = (time.time() - start_time) * 1000  # ms
                status_code = response.status

                status = f"✅ {service_type} online ({status_code}) - {latency:.1f}ms"
                print(f"[Performance Lab] {status}")
                return (True, latency, status)

        except urllib.error.HTTPError as e:
            latency = (time.time() - start_time) * 1000
            # Some services return errors but are still "up"
            if e.code in [401, 403, 404, 405]:
                status = f"⚠️ {service_type} responding ({e.code}) - {latency:.1f}ms"
                print(f"[Performance Lab] {status}")
                return (True, latency, status)
            return (False, latency, f"❌ HTTP {e.code}: {e.reason}")

        except urllib.error.URLError as e:
            return (False, 0.0, f"❌ Cannot connect: {e.reason}")
        except Exception as e:
            return (False, 0.0, f"❌ Error: {e}")


class PerfLab_NetworkScanner:
    """Network Scanner - Find generative AI services on your network."""

    DESCRIPTION = """🔍 NETWORK SCANNER

HOW TO USE:
1. Enter the base IP (e.g., 192.168.1) or localhost
2. Select which services to scan for
3. Run to discover running services

INPUTS:
• base_ip: Network prefix or 'localhost' for local only
• scan_comfyui: Check for ComfyUI instances
• scan_a1111: Check for Automatic1111/Forge
• scan_ollama: Check for Ollama LLM server
• scan_kobold: Check for KoboldCpp

OUTPUTS:
• found_services: JSON list of discovered services
• count: Number of services found
• report: Human-readable summary

USE FOR:
• Discovering AI services on your network
• Building distributed pipeline configurations
• Finding available compute resources

TIPS:
• Localhost scan is fast (checks common ports)
• Network scan checks .1-.254 (takes longer)
• Only scans standard ports per service"""

    CATEGORY = "⚡ Performance Lab/Network"
    FUNCTION = "scan"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("found_services", "count", "report")
    OUTPUT_NODE = True

    # Standard ports per service
    SERVICE_PORTS = {
        "ComfyUI": [8188, 8189],
        "Automatic1111": [7860, 7861],
        "Ollama": [11434],
        "KoboldCpp": [5001, 5000],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_ip": ("STRING", {"default": "localhost"}),
            },
            "optional": {
                "scan_comfyui": ("BOOLEAN", {"default": True}),
                "scan_a1111": ("BOOLEAN", {"default": True}),
                "scan_ollama": ("BOOLEAN", {"default": True}),
                "scan_kobold": ("BOOLEAN", {"default": True}),
            }
        }

    def scan(self, base_ip: str, scan_comfyui=True, scan_a1111=True,
             scan_ollama=True, scan_kobold=True):
        import socket

        found = []
        services_to_scan = {}

        if scan_comfyui:
            services_to_scan["ComfyUI"] = self.SERVICE_PORTS["ComfyUI"]
        if scan_a1111:
            services_to_scan["Automatic1111"] = self.SERVICE_PORTS["Automatic1111"]
        if scan_ollama:
            services_to_scan["Ollama"] = self.SERVICE_PORTS["Ollama"]
        if scan_kobold:
            services_to_scan["KoboldCpp"] = self.SERVICE_PORTS["KoboldCpp"]

        print(f"[Performance Lab] Scanning for services on {base_ip}...")

        # Determine hosts to scan
        if base_ip.lower() in ["localhost", "127.0.0.1", "local"]:
            hosts = ["127.0.0.1"]
        else:
            # Scan local subnet
            hosts = [f"{base_ip}.{i}" for i in range(1, 255)]

        for host in hosts:
            for service_name, ports in services_to_scan.items():
                for port in ports:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.5 if host == "127.0.0.1" else 0.2)
                        result = sock.connect_ex((host, port))
                        sock.close()

                        if result == 0:
                            entry = {
                                "service": service_name,
                                "host": host,
                                "port": port,
                                "url": f"http://{host}:{port}"
                            }
                            found.append(entry)
                            print(f"   ✅ Found {service_name} at {host}:{port}")

                    except:
                        pass

        # Build report
        if found:
            report_lines = ["═══ Services Found ═══", ""]
            for svc in found:
                report_lines.append(f"• {svc['service']}: {svc['url']}")
            report = "\n".join(report_lines)
        else:
            report = "❌ No services found"

        print(f"[Performance Lab] Scan complete: {len(found)} services found")

        return (json.dumps(found, indent=2), len(found), report)


class PerfLab_BenchmarkRunner:
    """Benchmark Runner - Run a workflow multiple times and average results."""

    DESCRIPTION = """🏁 BENCHMARK RUNNER

HOW TO USE:
1. Connect workflow_json from Load Workflow
2. Set number of runs (3-5 recommended)
3. Connect to Queue Workflow for execution
4. View averaged performance metrics

INPUTS:
• runs: Number of times to run (1-10)
• warmup_runs: Discard first N runs (0-3)
• delay_between: Seconds between runs

OUTPUTS:
• avg_duration: Average time per run
• min_duration: Fastest run
• max_duration: Slowest run
• report: Full benchmark report

USE FOR:
• Getting reliable performance metrics
• Comparing workflow optimizations
• Eliminating one-off timing variations

TIPS:
• Use 3+ runs for stable averages
• First run is often slower (warmup)
• Set delay to let GPU cool between runs"""

    CATEGORY = "⚡ Performance Lab/Meta-Workflow"
    FUNCTION = "benchmark"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("avg_duration", "min_duration", "max_duration", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runs": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
            "optional": {
                "duration_1": ("FLOAT", {"default": 0.0}),
                "duration_2": ("FLOAT", {"default": 0.0}),
                "duration_3": ("FLOAT", {"default": 0.0}),
                "duration_4": ("FLOAT", {"default": 0.0}),
                "duration_5": ("FLOAT", {"default": 0.0}),
                "warmup_runs": ("INT", {"default": 1, "min": 0, "max": 3}),
                "label": ("STRING", {"default": "Benchmark"}),
            }
        }

    def benchmark(self, runs, duration_1=0.0, duration_2=0.0, duration_3=0.0,
                  duration_4=0.0, duration_5=0.0, warmup_runs=1, label="Benchmark"):
        # Collect all non-zero durations
        all_durations = [d for d in [duration_1, duration_2, duration_3, duration_4, duration_5] if d > 0]

        if not all_durations:
            return (0.0, 0.0, 0.0, "❌ No duration data - connect Performance Report outputs")

        # Skip warmup runs
        if warmup_runs > 0 and len(all_durations) > warmup_runs:
            durations = all_durations[warmup_runs:]
        else:
            durations = all_durations

        avg_dur = sum(durations) / len(durations)
        min_dur = min(durations)
        max_dur = max(durations)
        variance = max_dur - min_dur

        report = f"""═══ {label} Results ═══

📊 Runs analyzed: {len(durations)} (after {warmup_runs} warmup)
⏱️  Average: {avg_dur:.2f}s
🚀 Fastest: {min_dur:.2f}s
🐢 Slowest: {max_dur:.2f}s
📏 Variance: {variance:.2f}s ({(variance/avg_dur*100):.1f}%)

Raw times: {', '.join(f'{d:.2f}s' for d in all_durations)}"""

        print(f"\n[Performance Lab]\n{report}\n")

        return (avg_dur, min_dur, max_dur, report)


# ═══════════════════════════════════════════════════════════════════════════════
# USER-FRIENDLY HELPER NODES
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_OneClickOptimize:
    """One-Click Optimize - Everything in one simple node!"""

    DESCRIPTION = """⚡ ONE-CLICK OPTIMIZE

THE EASIEST WAY TO OPTIMIZE!

Just connect your values and toggle Test Mode:
• Test Mode ON = Fast testing (low res, few steps)
• Test Mode OFF = Full quality production

INPUTS (connect from your workflow):
• width/height: Your resolution
• steps: Your step count
• cfg: Your CFG value
• batch_size: Your batch size

OUTPUTS (connect to your workflow):
• All values optimized based on mode

ONE TOGGLE CONTROLS EVERYTHING!
No need to understand each optimization."""

    CATEGORY = "⚡ Performance Lab/⭐ Start Here"
    FUNCTION = "optimize"
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "steps", "cfg", "batch_size", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "test_mode": ("BOOLEAN", {"default": True,
                             "tooltip": "ON = fast testing, OFF = full quality"}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192,
                         "tooltip": "Connect from Empty Latent or enter value"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192,
                          "tooltip": "Connect from Empty Latent or enter value"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200,
                         "tooltip": "Connect from KSampler or enter value"}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0,
                       "tooltip": "Connect from KSampler or enter value"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                              "tooltip": "Connect from Empty Latent or enter value"}),
                "test_resolution": ("INT", {"default": 512, "min": 256, "max": 1024,
                                   "tooltip": "Max resolution in test mode"}),
                "test_steps": ("INT", {"default": 15, "min": 4, "max": 50,
                              "tooltip": "Max steps in test mode"}),
            }
        }

    def optimize(self, test_mode, width=1024, height=1024, steps=30, cfg=7.0,
                 batch_size=1, test_resolution=512, test_steps=15):
        if test_mode:
            # Apply test optimizations
            if width > test_resolution or height > test_resolution:
                scale = test_resolution / max(width, height)
                width = int(width * scale) // 8 * 8
                height = int(height * scale) // 8 * 8
            steps = min(steps, test_steps)
            batch_size = 1
            status = f"🧪 TEST MODE: {width}x{height}, {steps} steps, batch=1"
        else:
            status = f"🎬 PRODUCTION: {width}x{height}, {steps} steps, batch={batch_size}"

        print(f"[Performance Lab] {status}")
        return (width, height, steps, cfg, batch_size, status)


class PerfLab_QuickStart:
    """Quick Start Guide - Learn how to use Performance Lab."""

    DESCRIPTION = """📚 QUICK START GUIDE

ADD THIS NODE TO SEE INSTRUCTIONS!

This node outputs helpful documentation
about how to use Performance Lab.

Select a topic to learn about:
• Getting Started - First steps
• Speed Optimization - Make it faster
• VRAM Optimization - Fix out of memory
• Quality Optimization - Better images
• Troubleshooting - Fix common issues"""

    CATEGORY = "⚡ Performance Lab/⭐ Start Here"
    FUNCTION = "guide"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("guide",)
    OUTPUT_NODE = True

    GUIDES = {
        "Getting Started": """═══ GETTING STARTED ═══

Welcome to Performance Lab! Here's how to begin:

1️⃣ ADD TIMING
   • Add ⏱️ Start Timer at the START
   • Add 📊 Performance Report at the END
   • Connect timer output to report

2️⃣ RUN YOUR WORKFLOW
   • Queue your workflow normally
   • Check console for timing results

3️⃣ OPTIMIZE
   • Add ⚡ One-Click Optimize
   • Toggle Test Mode ON for fast testing
   • Toggle OFF when you want full quality

That's it! You're optimizing! 🎉""",

        "Speed Optimization": """═══ SPEED OPTIMIZATION ═══

Make your workflow run FASTER:

🚀 QUICK WINS:
• Lower resolution (768 instead of 1024)
• Fewer steps (15-20 for testing)
• Batch size 1

📐 USE THESE NODES:
• 📐 Cap Resolution - Limit dimensions
• 🔢 Reduce Steps - Limit step count
• 🚀 Speed Test Preset - All at once

💡 TIPS:
• Resolution affects speed quadratically
  (2x resolution = 4x slower)
• Steps affect speed linearly
  (2x steps = 2x slower)
• Test at 512px, render at 1024px""",

        "VRAM Optimization": """═══ VRAM OPTIMIZATION ═══

Fix "CUDA out of memory" errors:

💾 USE THESE NODES:
• 💾 Low VRAM Preset - Select your GPU size
• 📦 Reduce Batch - Force batch=1
• 📐 Cap Resolution - Lower resolution

🎯 RECOMMENDED SETTINGS BY GPU:
• 6GB:  512px max, 20 steps, batch=1
• 8GB:  768px max, 25 steps, batch=1
• 12GB: 1024px max, 30 steps, batch=1

💡 TIPS:
• SDXL uses 2x VRAM of SD 1.5
• Batch=4 uses 4x VRAM of batch=1
• Upscalers add 2-4GB VRAM""",

        "Quality Optimization": """═══ QUALITY OPTIMIZATION ═══

Get better image quality:

🎨 KEY SETTINGS:
• Use correct CFG for your model
• Use enough steps (25-30 for quality)
• Use native resolution

🎯 OPTIMAL CFG BY MODEL:
• SD 1.5: 7.5
• SDXL: 7.0
• SD3: 4.5
• Flux Dev: 3.5
• Flux Schnell: 1.0

📐 OPTIMAL RESOLUTION:
• SD 1.5: 512x512
• SDXL/Flux: 1024x1024

💡 TIPS:
• Higher steps = more detail (diminishing returns after 30)
• Wrong CFG = black or burned images""",

        "Troubleshooting": """═══ TROUBLESHOOTING ═══

Common issues and fixes:

🖤 BLACK IMAGES?
• Flux with CFG > 4? Lower to 1-3.5
• Too few steps? Use at least 15-20
• Use 🔧 Black Image Fix node

💥 OUT OF MEMORY?
• Lower resolution
• Reduce batch to 1
• Use 💾 Low VRAM Preset

🐌 TOO SLOW?
• Lower resolution during testing
• Use fewer steps (15-20)
• Use 🚀 Speed Test Preset

❓ NODES NOT SHOWING?
• Restart ComfyUI
• Check console for errors
• Reinstall from Manager""",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "topic": (list(cls.GUIDES.keys()),
                         {"tooltip": "Select a topic to learn about"}),
            }
        }

    def guide(self, topic):
        guide_text = self.GUIDES.get(topic, "Topic not found")
        print(f"\n[Performance Lab]\n{guide_text}\n")
        return (guide_text,)


class PerfLab_AutoDetectGPU:
    """Auto Detect GPU - Automatically detect your GPU and VRAM."""

    DESCRIPTION = """🔍 AUTO DETECT GPU

Automatically detects your GPU and suggests settings!

NO INPUTS NEEDED - just add this node and run.

OUTPUTS:
• gpu_name: Your GPU model
• vram_gb: Total VRAM in GB
• suggested_preset: Recommended VRAM preset
• info: Full GPU information

USE THIS:
Connect 'suggested_preset' to Low VRAM Preset
to automatically use the right settings!"""

    CATEGORY = "⚡ Performance Lab/⭐ Start Here"
    FUNCTION = "detect"
    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("gpu_name", "vram_gb", "suggested_preset", "info")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def detect(self):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used_vram = torch.cuda.memory_allocated(0) / (1024**3)
                free_vram = total_vram - used_vram

                # Suggest preset based on VRAM
                if total_vram < 7:
                    preset = "6GB"
                elif total_vram < 10:
                    preset = "8GB"
                else:
                    preset = "12GB"

                info = f"""═══ GPU Detected ═══
🖥️  GPU: {gpu_name}
💾 Total VRAM: {total_vram:.1f} GB
📊 Used: {used_vram:.1f} GB
📊 Free: {free_vram:.1f} GB
🎯 Suggested Preset: {preset}"""

                print(f"\n[Performance Lab]\n{info}\n")
                return (gpu_name, total_vram, preset, info)
            else:
                return ("No CUDA GPU", 0.0, "8GB", "❌ No CUDA GPU detected")
        except Exception as e:
            return ("Error", 0.0, "8GB", f"❌ Error: {e}")


class PerfLab_ModelDetector:
    """Model Detector - Detect model type from checkpoint name."""

    DESCRIPTION = """🔍 MODEL DETECTOR

Detects your model type from the checkpoint name!

HOW TO USE:
1. Enter or connect your checkpoint filename
2. Get detected model type and optimal settings

DETECTS:
• SD 1.5 (sd15, v1-5, etc.)
• SDXL (sdxl, xl, etc.)
• SD3 (sd3, stable-diffusion-3)
• Flux (flux, dev, schnell)

OUTPUTS:
• model_type: Detected model
• optimal_cfg: Best CFG for this model
• optimal_resolution: Best resolution
• info: Detection details"""

    CATEGORY = "⚡ Performance Lab/⭐ Start Here"
    FUNCTION = "detect"
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("model_type", "optimal_cfg", "optimal_resolution", "info")
    OUTPUT_NODE = True

    MODEL_PATTERNS = {
        "Flux Schnell": ["schnell", "flux-schnell"],
        "Flux Dev": ["flux", "flux-dev"],
        "SD3": ["sd3", "stable-diffusion-3"],
        "SDXL": ["sdxl", "xl-base", "xl_base"],
        "SD 1.5": ["sd15", "v1-5", "sd_1.5", "1.5"],
    }

    MODEL_SETTINGS = {
        "Flux Schnell": {"cfg": 1.0, "resolution": 1024},
        "Flux Dev": {"cfg": 3.5, "resolution": 1024},
        "SD3": {"cfg": 4.5, "resolution": 1024},
        "SDXL": {"cfg": 7.0, "resolution": 1024},
        "SD 1.5": {"cfg": 7.5, "resolution": 512},
        "Unknown": {"cfg": 7.0, "resolution": 768},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_name": ("STRING", {"default": "",
                                   "tooltip": "Enter checkpoint filename or connect from loader"}),
            }
        }

    def detect(self, checkpoint_name):
        name_lower = checkpoint_name.lower()
        detected = "Unknown"

        for model_type, patterns in self.MODEL_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    detected = model_type
                    break
            if detected != "Unknown":
                break

        settings = self.MODEL_SETTINGS.get(detected, self.MODEL_SETTINGS["Unknown"])
        cfg = settings["cfg"]
        resolution = settings["resolution"]

        info = f"""═══ Model Detection ═══
📁 Checkpoint: {checkpoint_name}
🎯 Detected: {detected}
⚙️  Optimal CFG: {cfg}
📐 Optimal Resolution: {resolution}x{resolution}"""

        print(f"\n[Performance Lab]\n{info}\n")
        return (detected, cfg, resolution, info)


class PerfLab_TestModeToggle:
    """Test Mode Toggle - Simple on/off for your whole workflow."""

    DESCRIPTION = """🔘 TEST MODE TOGGLE

THE SIMPLEST OPTIMIZATION!

Just ONE toggle that outputs TRUE or FALSE.
Connect this to enable/disable inputs on
other nodes throughout your workflow.

USE:
• ON = Testing (connect to 'enabled' inputs)
• OFF = Production

CONNECT TO:
• Cap Resolution 'enabled'
• Reduce Steps 'enabled'
• Low VRAM Preset 'enabled'
• Any other 'enabled' input

ONE TOGGLE CONTROLS YOUR WHOLE WORKFLOW!"""

    CATEGORY = "⚡ Performance Lab/⭐ Start Here"
    FUNCTION = "toggle"
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("test_mode", "production_mode", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "test_mode": ("BOOLEAN", {"default": True,
                             "tooltip": "ON = testing mode, OFF = production mode"}),
            }
        }

    def toggle(self, test_mode):
        if test_mode:
            status = "🧪 TEST MODE - Fast iteration, lower quality"
        else:
            status = "🎬 PRODUCTION MODE - Full quality output"

        print(f"[Performance Lab] {status}")
        return (test_mode, not test_mode, status)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    # ⭐ Start Here (most user-friendly)
    "PerfLab_OneClickOptimize": PerfLab_OneClickOptimize,
    "PerfLab_QuickStart": PerfLab_QuickStart,
    "PerfLab_TestModeToggle": PerfLab_TestModeToggle,
    "PerfLab_AutoDetectGPU": PerfLab_AutoDetectGPU,
    "PerfLab_ModelDetector": PerfLab_ModelDetector,

    # Monitoring
    "PerfLab_Timer": PerfLab_Timer,
    "PerfLab_Report": PerfLab_Report,
    "PerfLab_VRAMMonitor": PerfLab_VRAMMonitor,

    # Quick Optimize
    "PerfLab_CapResolution": PerfLab_CapResolution,
    "PerfLab_ReduceSteps": PerfLab_ReduceSteps,
    "PerfLab_ReduceBatch": PerfLab_ReduceBatch,
    "PerfLab_OptimizeCFG": PerfLab_OptimizeCFG,
    "PerfLab_SpeedPreset": PerfLab_SpeedPreset,
    "PerfLab_LowVRAMPreset": PerfLab_LowVRAMPreset,

    # Analysis
    "PerfLab_Analyzer": PerfLab_Analyzer,
    "PerfLab_BlackImageFix": PerfLab_BlackImageFix,
    "PerfLab_Compare": PerfLab_Compare,

    # LLM
    "PerfLab_GeneratePrompt": PerfLab_GeneratePrompt,
    "PerfLab_LLMClient": PerfLab_LLMClient,
    "PerfLab_AutoOptimize": PerfLab_AutoOptimize,
    "PerfLab_ApplyOrRevert": PerfLab_ApplyOrRevert,

    # Utility
    "PerfLab_ShowText": PerfLab_ShowText,
    "PerfLab_Switch": PerfLab_Switch,
    "PerfLab_IntSwitch": PerfLab_IntSwitch,
    "PerfLab_FloatSwitch": PerfLab_FloatSwitch,

    # Meta-Workflow (test other workflows)
    "PerfLab_LoadWorkflow": PerfLab_LoadWorkflow,
    "PerfLab_QueueWorkflow": PerfLab_QueueWorkflow,
    "PerfLab_BenchmarkRunner": PerfLab_BenchmarkRunner,

    # Network (distributed services)
    "PerfLab_EndpointHealth": PerfLab_EndpointHealth,
    "PerfLab_NetworkScanner": PerfLab_NetworkScanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # ⭐ Start Here (most user-friendly - shown first!)
    "PerfLab_OneClickOptimize": "⚡ One-Click Optimize",
    "PerfLab_QuickStart": "📚 Quick Start Guide",
    "PerfLab_TestModeToggle": "🔘 Test Mode Toggle",
    "PerfLab_AutoDetectGPU": "🔍 Auto Detect GPU",
    "PerfLab_ModelDetector": "🔍 Model Detector",

    # Monitoring
    "PerfLab_Timer": "⏱️ Start Timer",
    "PerfLab_Report": "📊 Performance Report",
    "PerfLab_VRAMMonitor": "💾 VRAM Monitor",

    # Quick Optimize
    "PerfLab_CapResolution": "📐 Cap Resolution",
    "PerfLab_ReduceSteps": "🔢 Reduce Steps",
    "PerfLab_ReduceBatch": "📦 Reduce Batch",
    "PerfLab_OptimizeCFG": "🎯 Optimize CFG",
    "PerfLab_SpeedPreset": "🚀 Speed Test Preset",
    "PerfLab_LowVRAMPreset": "💾 Low VRAM Preset",

    # Analysis
    "PerfLab_Analyzer": "🔍 Workflow Analyzer",
    "PerfLab_BlackImageFix": "🔧 Black Image Fix",
    "PerfLab_Compare": "📊 Compare Results",

    # LLM
    "PerfLab_GeneratePrompt": "🤖 Generate LLM Prompt",
    "PerfLab_LLMClient": "🤖 LLM Client",
    "PerfLab_AutoOptimize": "🤖 Auto Optimize (LLM)",
    "PerfLab_ApplyOrRevert": "✅ Apply or Revert",

    # Utility
    "PerfLab_ShowText": "📝 Show Text",
    "PerfLab_Switch": "🔀 A/B Switch",
    "PerfLab_IntSwitch": "🔢 Int A/B Switch",
    "PerfLab_FloatSwitch": "🔢 Float A/B Switch",

    # Meta-Workflow
    "PerfLab_LoadWorkflow": "📂 Load Workflow",
    "PerfLab_QueueWorkflow": "▶️ Queue Workflow",
    "PerfLab_BenchmarkRunner": "🏁 Benchmark Runner",

    # Network
    "PerfLab_EndpointHealth": "🏥 Endpoint Health",
    "PerfLab_NetworkScanner": "🔍 Network Scanner",
}

# Print startup message
print(f"""
╔══════════════════════════════════════════════════════════════╗
║          ⚡ Performance Lab v{__version__} Loaded! ⚡               ║
╠══════════════════════════════════════════════════════════════╣
║  {len(NODE_CLASS_MAPPINGS)} nodes in "⚡ Performance Lab" category:               ║
║                                                              ║
║  ⭐ START HERE:    One-Click Optimize, Quick Start Guide     ║
║  📊 Monitoring:    Timer, Report, VRAM Monitor               ║
║  🚀 Optimize:      Cap Res, Steps, Batch, CFG, Presets       ║
║  🔍 Analysis:      Analyzer, Black Image Fix, Compare        ║
║  🤖 LLM:           Generate Prompt, LLM Client, Auto Optimize ║
║  🔧 Utility:       Show Text, A/B Switches                   ║
║  📂 Meta-Workflow: Load, Queue, Benchmark                    ║
║  🌐 Network:       Health Check, Scanner                     ║
║                                                              ║
║  💡 New? Add "📚 Quick Start Guide" node for instructions!   ║
╚══════════════════════════════════════════════════════════════╝
""")
