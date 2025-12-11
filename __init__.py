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

# Version
__version__ = "0.5.0"

# Ensure the module directory is in path for imports
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MONITORING NODES
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_Timer:
    """
    ⏱️ Start Timer - Place at the START of your workflow

    Begins tracking execution time. Connect the output to Performance Report
    at the end of your workflow to see how long generation took.
    """

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
    """
    📊 Performance Report - Place at the END of your workflow

    Shows execution time and VRAM usage. Connect any output from your
    workflow to trigger this at the end.
    """

    CATEGORY = "⚡ Performance Lab/Monitoring"
    FUNCTION = "report"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "duration_sec", "peak_vram_gb")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*",),  # Any input triggers report
            },
            "optional": {
                "timer": ("PERF_TIMER",),
                "label": ("STRING", {"default": "Generation"}),
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
    """
    💾 VRAM Monitor - Check VRAM usage at any point

    Place anywhere in your workflow to see current GPU memory usage.
    Useful for finding which nodes use the most VRAM.
    """

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
                "passthrough": ("*",),
                "checkpoint_name": ("STRING", {"default": ""}),
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
    """
    📐 Cap Resolution - Limit image dimensions for faster testing

    Reduces resolution to speed up generation during testing.
    Lower resolution = faster generation + less VRAM.
    """

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "cap"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "max_size": ("INT", {"default": 768, "min": 256, "max": 2048,
                            "tooltip": "Maximum dimension (width or height)"}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
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
    """
    🔢 Reduce Steps - Lower sampling steps for faster iteration

    Fewer steps = faster generation. 15-20 steps is usually enough
    for testing composition before doing final renders.
    """

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "reduce"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("steps",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "max_steps": ("INT", {"default": 20, "min": 1, "max": 100,
                             "tooltip": "Cap steps to this value during testing"}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
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
    """
    📦 Reduce Batch Size - Generate one image at a time

    Batch size 1 uses minimum VRAM. Great for testing workflows
    that run out of memory.
    """

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "reduce"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("batch_size",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64}),
            },
            "optional": {
                "force_single": ("BOOLEAN", {"default": True,
                                "tooltip": "Force batch size to 1"}),
            }
        }

    def reduce(self, batch_size: int, force_single: bool = True):
        if force_single and batch_size > 1:
            print(f"[Performance Lab] Batch size reduced: {batch_size} → 1")
            return (1,)
        return (batch_size,)


class PerfLab_OptimizeCFG:
    """
    🎯 Optimize CFG - Auto-adjust CFG for your model type

    Different models work best with different CFG values.
    This node suggests optimal CFG based on model type.
    """

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
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "model_type": (list(cls.MODEL_CFG.keys()),),
            },
            "optional": {
                "auto_adjust": ("BOOLEAN", {"default": True}),
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
    """
    🚀 Speed Test Preset - All optimizations in one node

    Applies multiple optimizations at once for maximum speed
    during testing: lower resolution, fewer steps, batch=1.
    """

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "apply"
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "steps", "cfg")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
                "target_resolution": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "target_steps": ("INT", {"default": 15, "min": 4, "max": 50}),
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
    """
    💾 8GB VRAM Preset - Optimized for low VRAM GPUs

    Applies settings optimized for 8GB VRAM cards:
    - Lower resolution
    - Reduced batch size
    - Optimized for memory efficiency
    """

    CATEGORY = "⚡ Performance Lab/Quick Optimize"
    FUNCTION = "apply"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "steps", "batch_size")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
                "vram_target": (["6GB", "8GB", "12GB"],),
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
    """
    🔍 Workflow Analyzer - Analyze workflow structure

    Paste workflow JSON to get detailed analysis:
    - Node count and types
    - Feature detection (upscaling, ControlNet, etc.)
    - Optimization suggestions
    """

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
    """
    🔧 Black Image Diagnostic - Troubleshoot dark/empty outputs

    Common causes of black images:
    - Wrong CFG for model (Flux needs 1-3, SD needs 5-8)
    - Missing VAE
    - Too few steps
    - Wrong resolution for model
    """

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
    """
    📊 Compare Results - Before/After comparison

    Connect two performance reports to see improvement.
    Shows duration change, VRAM change, and % improvement.
    """

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
    """
    🤖 Generate LLM Prompt - Create prompt for Claude/GPT/Gemini

    Generates a detailed prompt you can paste into an LLM to get
    optimization suggestions. The LLM will respond with workflow changes.
    """

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


class PerfLab_ShowText:
    """
    📝 Show Text - Display any text output in the node

    Useful for showing analysis results, prompts, or any string
    directly in the node preview.
    """

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
    """
    🔀 A/B Switch - Compare two configurations

    Easily switch between "test" and "production" settings.
    Great for comparing optimized vs original values.
    """

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
    """
    🔢 Integer A/B Switch - Switch between two integer values

    Perfect for comparing step counts, resolutions, etc.
    """

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
    """
    🔢 Float A/B Switch - Switch between two float values

    Perfect for comparing CFG scales, denoise values, etc.
    """

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
# NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
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

    # Utility
    "PerfLab_ShowText": PerfLab_ShowText,
    "PerfLab_Switch": PerfLab_Switch,
    "PerfLab_IntSwitch": PerfLab_IntSwitch,
    "PerfLab_FloatSwitch": PerfLab_FloatSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
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

    # Utility
    "PerfLab_ShowText": "📝 Show Text",
    "PerfLab_Switch": "🔀 A/B Switch",
    "PerfLab_IntSwitch": "🔢 Int A/B Switch",
    "PerfLab_FloatSwitch": "🔢 Float A/B Switch",
}

# Print startup message
print(f"""
╔══════════════════════════════════════════════════════════════╗
║          ⚡ Performance Lab v{__version__} Loaded! ⚡               ║
╠══════════════════════════════════════════════════════════════╣
║  {len(NODE_CLASS_MAPPINGS)} nodes in "⚡ Performance Lab" category:               ║
║                                                              ║
║  📊 Monitoring:    Timer, Report, VRAM Monitor               ║
║  🚀 Optimize:      Cap Res, Steps, Batch, CFG, Presets       ║
║  🔍 Analysis:      Analyzer, Black Image Fix, Compare        ║
║  🤖 LLM:           Generate Prompt                           ║
║  🔧 Utility:       Show Text, A/B Switches                   ║
╚══════════════════════════════════════════════════════════════╝
""")
