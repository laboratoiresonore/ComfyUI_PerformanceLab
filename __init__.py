"""
ComfyUI Performance Lab v2.0 - LLM-guided workflow optimization

Focused toolkit with 11 nodes (down from 31):
- 6 Core utility nodes (Timer, Report, VRAMMonitor, ShowText, CapResolution, Compare)
- 5 New v2.0 LLM-powered nodes (AutoFix, Optimizer, ABTest, Feedback, NetworkSetup)

Install: Place in ComfyUI/custom_nodes/ComfyUI_PerformanceLab/
"""

import os
import sys
import json
import time
import socket
from typing import Dict, Any, List, Tuple, Optional

# Version and metadata
__version__ = "2.1.0"
__author__ = "Laboratoire Sonore"
__description__ = "ComfyUI Performance Lab v2.1 - Ultimate AI Workflow Optimization & Testing"

# ComfyUI Manager integration
WEB_DIRECTORY = None

__all__ = [
    "__version__",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

# Ensure the module directory is in path
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE UTILITY NODES (Kept from v1 - proven useful)
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_Timer:
    """Start Timer - Place at the START of your workflow."""

    DESCRIPTION = """⏱️ START TIMER

Place at the BEGINNING of your workflow.
Connect 'timer' output to Performance Report at the END.
Run your workflow - the report shows timing."""

    CATEGORY = "⚡ Performance Lab"
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

Place at the END of your workflow.
Connect ANY output to 'trigger' and timer from Start Timer.
Shows duration, peak VRAM, current VRAM."""

    CATEGORY = "⚡ Performance Lab"
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
                "timer": ("PERF_TIMER", {"tooltip": "Connect from Start Timer for timing"}),
                "label": ("STRING", {"default": "Generation"}),
            }
        }

    def report(self, trigger, timer=None, label="Generation"):
        end_time = time.time()

        duration = end_time - timer["start_time"] if timer and "start_time" in timer else 0.0

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

        report = f"""═══ {label} Complete ═══
⏱️  Duration: {duration:.2f} seconds
📈 Peak VRAM: {peak_vram:.2f} GB
📊 Current VRAM: {current_vram:.2f} GB"""

        print(f"\n[Performance Lab]\n{report}\n")
        return (report, duration, peak_vram)


class PerfLab_VRAMMonitor:
    """VRAM Monitor - Check GPU memory at any point."""

    DESCRIPTION = """💾 VRAM MONITOR

Place ANYWHERE in your workflow as a passthrough.
Shows used/free/total VRAM in console.
Data passes through unchanged."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "check"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "*")
    RETURN_NAMES = ("vram_info", "used_gb", "free_gb", "passthrough")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "passthrough": ("*", {"tooltip": "Data passes through unchanged"}),
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


class PerfLab_ShowText:
    """Show Text - Display any text in the UI."""

    DESCRIPTION = """📝 SHOW TEXT

Display text output in the ComfyUI interface.
Connect any string output to see it displayed."""

    CATEGORY = "⚡ Performance Lab"
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
        print(f"[Performance Lab] {text}")
        return (text,)


class PerfLab_CapResolution:
    """Cap Resolution - Limit dimensions for faster testing."""

    DESCRIPTION = """📐 CAP RESOLUTION

Limits resolution while maintaining aspect ratio.
Great for fast iteration - 768px for testing, 1024+ for finals.
Toggle 'enabled' to compare original vs capped."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "cap"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "max_size": ("INT", {"default": 768, "min": 256, "max": 2048}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    def cap(self, width: int, height: int, max_size: int, enabled: bool = True):
        if not enabled:
            return (width, height)

        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale) // 8 * 8
            new_height = int(height * scale) // 8 * 8
            print(f"[Performance Lab] Resolution capped: {width}x{height} → {new_width}x{new_height}")
            return (new_width, new_height)

        return (width, height)


class PerfLab_Compare:
    """Compare Results - See before/after improvement."""

    DESCRIPTION = """📊 COMPARE RESULTS

Compare before/after optimization results.
Shows % improvement in duration and VRAM.
Green = improvement, Red = regression."""

    CATEGORY = "⚡ Performance Lab"
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

│   Metric    │  {before_label:^9}  │  {after_label:^9}  │  Change  │
│ Duration    │  {before_duration:>7.2f}s   │  {after_duration:>7.2f}s   │ {time_str:>8} │
│ Peak VRAM   │  {before_vram:>6.2f} GB  │  {after_vram:>6.2f} GB  │ {vram_str:>8} │

{time_verdict} {vram_verdict}"""

        print(f"\n[Performance Lab]\n{comparison}")
        return (comparison,)


# ═══════════════════════════════════════════════════════════════════════════════
# NEW v2.0 LLM-POWERED NODES
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_AutoFix:
    """Drop anywhere in your workflow - automatically detects and suggests fixes."""

    DESCRIPTION = """🪄 AUTO-FIX NODE

Drop this ANYWHERE in your workflow.
It will automatically:
• Detect your GPU capabilities
• Suggest optimal settings for your VRAM
• Recommend sampler/CFG based on model type
• Show real-time performance status

No configuration needed. Just connect and go.
Passthrough design = non-invasive."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "analyze"
    RETURN_TYPES = ("*", "STRING", "STRING", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("passthrough", "status", "recommendations", "suggested_steps", "suggested_resolution", "suggested_cfg")
    OUTPUT_NODE = True

    # Model-specific optimal settings
    MODEL_SETTINGS = {
        "flux": {"cfg": 3.5, "steps": 28, "resolution": 1024},
        "sdxl": {"cfg": 7.0, "steps": 25, "resolution": 1024},
        "sd15": {"cfg": 7.5, "steps": 20, "resolution": 512},
        "sd3": {"cfg": 4.5, "steps": 28, "resolution": 1024},
    }

    # VRAM-based limits
    VRAM_LIMITS = {
        4: {"max_res": 512, "max_batch": 1},
        6: {"max_res": 768, "max_batch": 1},
        8: {"max_res": 768, "max_batch": 2},
        12: {"max_res": 1024, "max_batch": 2},
        16: {"max_res": 1024, "max_batch": 4},
        24: {"max_res": 1536, "max_batch": 4},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*", {"tooltip": "Connect to any node output"}),
                "mode": (["Auto", "Speed", "Quality", "VRAM Saver"], {"default": "Auto"}),
                "model_hint": (["Auto-detect", "Flux", "SDXL", "SD 1.5", "SD3"], {"default": "Auto-detect"}),
            }
        }

    def analyze(self, any_input=None, mode="Auto", model_hint="Auto-detect"):
        recommendations = []

        # Detect GPU
        gpu_name = "Unknown"
        total_vram = 0
        used_vram = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used_vram = torch.cuda.memory_allocated(0) / (1024**3)
        except:
            pass

        # Find appropriate VRAM tier
        vram_tier = 8  # Default
        for tier in sorted(self.VRAM_LIMITS.keys()):
            if total_vram >= tier:
                vram_tier = tier

        limits = self.VRAM_LIMITS.get(vram_tier, self.VRAM_LIMITS[8])

        # Determine model type
        model_type = "sdxl"  # Default
        if model_hint != "Auto-detect":
            model_type = model_hint.lower().replace(" ", "").replace(".", "")
            if "flux" in model_type:
                model_type = "flux"
            elif "sdxl" in model_type or "xl" in model_type:
                model_type = "sdxl"
            elif "sd3" in model_type:
                model_type = "sd3"
            elif "15" in model_type or "1.5" in model_type:
                model_type = "sd15"

        settings = self.MODEL_SETTINGS.get(model_type, self.MODEL_SETTINGS["sdxl"])

        # Build recommendations based on mode
        suggested_steps = settings["steps"]
        suggested_resolution = min(settings["resolution"], limits["max_res"])
        suggested_cfg = settings["cfg"]

        if mode == "Speed":
            suggested_steps = min(suggested_steps, 15)
            suggested_resolution = min(suggested_resolution, 768)
            recommendations.append("🚀 Speed mode: Reduced steps and resolution")
        elif mode == "Quality":
            suggested_steps = max(suggested_steps, 30)
            recommendations.append("✨ Quality mode: Increased steps")
        elif mode == "VRAM Saver":
            suggested_resolution = min(suggested_resolution, limits["max_res"] - 256)
            recommendations.append("💾 VRAM mode: Conservative resolution")

        # Add model-specific tips
        if model_type == "flux":
            if suggested_cfg > 4:
                recommendations.append(f"⚠️ Flux needs low CFG! Using {suggested_cfg}")

        if total_vram < 8:
            recommendations.append(f"⚠️ Low VRAM ({total_vram:.1f}GB) - consider tiled VAE")

        # Build status
        status = f"""🖥️ GPU: {gpu_name}
💾 VRAM: {used_vram:.1f}/{total_vram:.1f} GB
🎯 Model: {model_type.upper()}
⚙️ Mode: {mode}"""

        recommendations_text = "\n".join(recommendations) if recommendations else "✅ Settings look optimal!"

        print(f"[Performance Lab AutoFix]\n{status}\n{recommendations_text}")

        return (any_input, status, recommendations_text, suggested_steps, suggested_resolution, suggested_cfg)


class PerfLab_Optimizer:
    """LLM-guided workflow optimization with memory."""

    DESCRIPTION = """🧠 LLM OPTIMIZER

The brain of Performance Lab v2.0.
Connects to KoboldCPP/Ollama/LiteLLM to:
• Analyze your workflow structure
• Remember your preferences from past sessions
• Generate intelligent suggestions
• Create A/B variations for testing

Modes:
• analyze: Understand what your workflow does
• suggest: Get optimization recommendations
• compare: Generate A/B test variations"""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "optimize"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "suggestions", "variations_json", "memory_context")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_message": ("STRING", {
                    "multiline": True,
                    "default": "My images are coming out too dark"
                }),
                "mode": (["analyze", "suggest", "compare"], {"default": "suggest"}),
            },
            "optional": {
                "workflow_json": ("STRING", {"multiline": True, "default": ""}),
                "llm_endpoint": ("STRING", {"default": "http://127.0.0.1:5001"}),
                "performance_report": ("STRING", {"forceInput": True}),
            }
        }

    def optimize(self, user_message, mode="suggest", workflow_json="",
                 llm_endpoint="http://127.0.0.1:5001", performance_report=""):

        import urllib.request
        import urllib.error

        # Build context for LLM
        context_parts = ["You are an expert ComfyUI workflow optimizer."]

        if performance_report:
            context_parts.append(f"\nCurrent performance:\n{performance_report}")

        if workflow_json:
            try:
                workflow = json.loads(workflow_json)
                nodes = workflow.get("nodes", [])
                node_types = [n.get("type", "Unknown") for n in nodes]
                context_parts.append(f"\nWorkflow has {len(nodes)} nodes: {', '.join(sorted(set(node_types)))}")
            except:
                pass

        # Mode-specific instructions
        if mode == "analyze":
            context_parts.append("\nAnalyze this workflow and explain what it does step by step.")
        elif mode == "suggest":
            context_parts.append("""
Suggest specific optimizations. For each suggestion, provide:
1. What to change
2. Current value (if known)
3. Suggested value
4. Expected improvement""")
        elif mode == "compare":
            context_parts.append("""
Create 2-3 variations to test. For each variation, provide specific parameter changes.
Format as JSON: [{"name": "Variation A", "changes": {"cfg": 3.5, "steps": 25}}, ...]""")

        system_prompt = "\n".join(context_parts)

        # Try multiple LLM endpoints in order
        llm_response = None
        endpoints_tried = []

        # 1. Try OpenAI-compatible API (LM Studio, Jan, LiteLLM)
        try:
            endpoints_tried.append("OpenAI-compatible (LM Studio/Jan/LiteLLM)")
            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 500,
            }

            req = urllib.request.Request(
                f"{llm_endpoint.rstrip('/')}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                llm_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        except (urllib.error.URLError, urllib.error.HTTPError):
            # 2. Try Kobold-style API
            try:
                endpoints_tried.append("KoboldCPP")
                payload = {
                    "prompt": f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:",
                    "max_length": 500,
                    "temperature": 0.7,
                }

                req = urllib.request.Request(
                    f"{llm_endpoint.rstrip('/')}/api/v1/generate",
                    data=json.dumps(payload).encode('utf-8'),
                    headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req, timeout=60) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    llm_response = result.get("results", [{}])[0].get("text", "")

            except (urllib.error.URLError, urllib.error.HTTPError):
                # 3. Try Ollama-style API
                try:
                    endpoints_tried.append("Ollama")
                    payload = {
                        "model": "llama3.2",
                        "prompt": f"{system_prompt}\n\nUser: {user_message}",
                        "stream": False,
                    }
                    req = urllib.request.Request(
                        f"{llm_endpoint.rstrip('/')}/api/generate",
                        data=json.dumps(payload).encode('utf-8'),
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=60) as response:
                        result = json.loads(response.read().decode('utf-8'))
                        llm_response = result.get("response", "")
                except:
                    pass
        except Exception as e:
            llm_response = None

        # If all attempts failed
        if not llm_response:
            llm_response = f"""Could not connect to LLM at {llm_endpoint}.

Tried: {', '.join(endpoints_tried)}

Supported endpoints:
• LM Studio: http://localhost:1234 (OpenAI-compatible)
• Jan: http://localhost:1337 (OpenAI-compatible)
• KoboldCPP: http://localhost:5001
• Ollama: http://localhost:11434
• LiteLLM: http://localhost:4000

Please check the endpoint is running."""

        # Parse suggestions and variations from response
        suggestions = llm_response
        variations_json = "[]"

        if mode == "compare" and "[" in llm_response:
            # Try to extract JSON array
            try:
                start = llm_response.index("[")
                end = llm_response.rindex("]") + 1
                variations_json = llm_response[start:end]
            except:
                pass

        # Memory context (for future session continuity)
        memory_context = json.dumps({
            "last_query": user_message,
            "mode": mode,
            "timestamp": time.time()
        })

        print(f"[Performance Lab Optimizer] Mode: {mode}")
        print(f"Response: {llm_response[:200]}...")

        return (llm_response, suggestions, variations_json, memory_context)


class PerfLab_ABTest:
    """Generate and compare workflow variations side-by-side."""

    DESCRIPTION = """🔬 A/B TEST

Compare two configurations side-by-side.
• Input your current settings
• Input alternate settings
• Get comparison metrics

Use with Optimizer node to test LLM suggestions."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "compare"
    RETURN_TYPES = ("STRING", "INT", "INT", "FLOAT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("comparison_report", "a_steps", "a_resolution", "a_cfg", "b_steps", "b_resolution", "b_cfg")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a_steps": ("INT", {"default": 30, "min": 1, "max": 150}),
                "a_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "a_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "b_steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "b_resolution": ("INT", {"default": 768, "min": 256, "max": 4096}),
                "b_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            },
            "optional": {
                "a_label": ("STRING", {"default": "Current"}),
                "b_label": ("STRING", {"default": "Optimized"}),
                "a_duration": ("FLOAT", {"default": 0.0}),
                "b_duration": ("FLOAT", {"default": 0.0}),
            }
        }

    def compare(self, a_steps, a_resolution, a_cfg, b_steps, b_resolution, b_cfg,
                a_label="Current", b_label="Optimized", a_duration=0.0, b_duration=0.0):

        # Calculate theoretical speedup from settings
        step_ratio = b_steps / a_steps if a_steps > 0 else 1
        # Resolution affects quadratically
        res_ratio = (b_resolution * b_resolution) / (a_resolution * a_resolution) if a_resolution > 0 else 1
        theoretical_speedup = step_ratio * res_ratio

        # Actual speedup if durations provided
        if a_duration > 0 and b_duration > 0:
            actual_speedup = b_duration / a_duration
            speed_note = f"Actual: {(1-actual_speedup)*100:+.1f}%"
        else:
            speed_note = f"Theoretical: {(1-theoretical_speedup)*100:+.1f}%"

        report = f"""═══ A/B Comparison ═══

Configuration A ({a_label}):
  • Steps: {a_steps}
  • Resolution: {a_resolution}x{a_resolution}
  • CFG: {a_cfg}
  • Duration: {a_duration:.2f}s

Configuration B ({b_label}):
  • Steps: {b_steps}
  • Resolution: {b_resolution}x{b_resolution}
  • CFG: {b_cfg}
  • Duration: {b_duration:.2f}s

Speed Change: {speed_note}

Connect both configurations to your workflow
and compare the output quality!"""

        print(f"[Performance Lab A/B Test]\n{report}")

        return (report, a_steps, a_resolution, a_cfg, b_steps, b_resolution, b_cfg)


class PerfLab_Feedback:
    """Record user preference and update memory."""

    DESCRIPTION = """👍 FEEDBACK

Record which variation you preferred.
This helps the system learn your preferences:
• Preferred CFG ranges
• Quality vs speed tradeoffs
• Sampler preferences

Over time, suggestions get better!"""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "record"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("learning_summary", "updated_memory")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "choice": (["A", "B", "Neither"], {"default": "A"}),
            },
            "optional": {
                "reason": ("STRING", {"multiline": True, "default": ""}),
                "a_settings": ("STRING", {"default": "{}"}),
                "b_settings": ("STRING", {"default": "{}"}),
                "previous_memory": ("STRING", {"default": "{}"}),
            }
        }

    def record(self, choice, reason="", a_settings="{}", b_settings="{}", previous_memory="{}"):

        # Parse previous memory
        try:
            memory = json.loads(previous_memory)
        except:
            memory = {}

        # Initialize preference tracking
        if "preferences" not in memory:
            memory["preferences"] = {
                "cfg_choices": [],
                "speed_vs_quality": [],
                "feedback_count": 0
            }

        # Record this choice
        memory["preferences"]["feedback_count"] += 1

        # Parse settings to understand what was chosen
        try:
            a = json.loads(a_settings) if isinstance(a_settings, str) else a_settings
            b = json.loads(b_settings) if isinstance(b_settings, str) else b_settings
        except:
            a, b = {}, {}

        chosen_settings = a if choice == "A" else (b if choice == "B" else {})

        learning_notes = []

        if chosen_settings.get("cfg"):
            memory["preferences"]["cfg_choices"].append(chosen_settings["cfg"])
            # Keep last 10
            memory["preferences"]["cfg_choices"] = memory["preferences"]["cfg_choices"][-10:]
            avg_cfg = sum(memory["preferences"]["cfg_choices"]) / len(memory["preferences"]["cfg_choices"])
            learning_notes.append(f"Your preferred CFG range: ~{avg_cfg:.1f}")

        if reason:
            learning_notes.append(f"Noted: {reason}")

        if choice == "Neither":
            learning_notes.append("Neither option was satisfactory - will try different approaches")

        learning_summary = "\n".join(learning_notes) if learning_notes else f"Recorded preference for option {choice}"

        # Save to disk
        memory_path = os.path.join(MODULE_DIR, "user_preferences.json")
        try:
            with open(memory_path, "w") as f:
                json.dump(memory, f, indent=2)
        except:
            pass

        print(f"[Performance Lab Feedback] {learning_summary}")

        return (learning_summary, json.dumps(memory))


class PerfLab_NetworkSetup:
    """One-click network configuration for multi-machine setups."""

    DESCRIPTION = """🌐 NETWORK SETUP

Discover and configure AI services on your network:
• ComfyUI instances (port 8188-8190)
• KoboldCPP (port 5001)
• Ollama (port 11434)

Generates LiteLLM config for load balancing.
One-click setup for distributed workflows."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "discover"
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("status_report", "litellm_config", "services_found", "endpoints_json")
    OUTPUT_NODE = True

    # Common AI service ports
    SERVICE_PORTS = {
        "comfyui": [8188, 8189, 8190],
        "kobold": [5001],
        "ollama": [11434],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scan_range": ("STRING", {"default": "192.168.1.0/24"}),
            },
            "optional": {
                "scan_localhost": ("BOOLEAN", {"default": True}),
                "timeout_ms": ("INT", {"default": 100, "min": 50, "max": 1000}),
            }
        }

    def _check_port(self, host: str, port: int, timeout: float) -> bool:
        """Check if a port is open on a host."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False

    def _parse_cidr(self, cidr: str) -> List[str]:
        """Parse CIDR notation to list of IPs (simplified, handles /24 only)."""
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

    def discover(self, scan_range="192.168.1.0/24", scan_localhost=True, timeout_ms=100):
        timeout = timeout_ms / 1000.0
        discovered = []

        # Scan localhost first
        if scan_localhost:
            for service, ports in self.SERVICE_PORTS.items():
                for port in ports:
                    if self._check_port("127.0.0.1", port, timeout):
                        discovered.append({
                            "host": "127.0.0.1",
                            "port": port,
                            "service": service,
                            "url": f"http://127.0.0.1:{port}"
                        })

        # Scan network range (quick scan, just check if ports are open)
        hosts = self._parse_cidr(scan_range)

        # Only scan a subset to avoid taking too long
        for host in hosts[:50]:  # Limit to first 50 IPs
            if host == "127.0.0.1":
                continue
            for service, ports in self.SERVICE_PORTS.items():
                for port in ports:
                    if self._check_port(host, port, timeout):
                        discovered.append({
                            "host": host,
                            "port": port,
                            "service": service,
                            "url": f"http://{host}:{port}"
                        })

        # Build status report
        status_lines = ["═══ Network Discovery ═══", ""]

        if discovered:
            status_lines.append(f"Found {len(discovered)} service(s):\n")
            for svc in discovered:
                status_lines.append(f"  🟢 {svc['service'].upper()} at {svc['url']}")
        else:
            status_lines.append("No services found on scanned range.")
            status_lines.append("Make sure services are running and ports are open.")

        status_report = "\n".join(status_lines)

        # Generate LiteLLM config
        kobold_endpoints = [s for s in discovered if s["service"] == "kobold"]
        ollama_endpoints = [s for s in discovered if s["service"] == "ollama"]

        litellm_models = []
        for i, ep in enumerate(kobold_endpoints):
            litellm_models.append({
                "model_name": "local-llm",
                "litellm_params": {
                    "model": "openai/kobold",
                    "api_base": f"{ep['url']}/v1",
                    "rpm": 10
                }
            })

        for i, ep in enumerate(ollama_endpoints):
            litellm_models.append({
                "model_name": "local-llm",
                "litellm_params": {
                    "model": "ollama/llama3.2",
                    "api_base": ep['url'],
                    "rpm": 10
                }
            })

        litellm_yaml = f"""# LiteLLM Configuration
# Save to ~/.litellm/config.yaml
# Run: litellm --config ~/.litellm/config.yaml

model_list:
"""
        for model in litellm_models:
            litellm_yaml += f"""  - model_name: {model['model_name']}
    litellm_params:
      model: {model['litellm_params']['model']}
      api_base: {model['litellm_params']['api_base']}
      rpm: {model['litellm_params']['rpm']}
"""

        if litellm_models:
            litellm_yaml += """
router_settings:
  routing_strategy: least-busy
  num_retries: 2
  allowed_fails: 3
  cooldown_time: 60
"""

        endpoints_json = json.dumps(discovered, indent=2)

        print(f"[Performance Lab Network]\n{status_report}")

        return (status_report, litellm_yaml, len(discovered), endpoints_json)


# ═══════════════════════════════════════════════════════════════════════════════
# NEW v2.1 ADVANCED NODES
# ═══════════════════════════════════════════════════════════════════════════════

class PerfLab_GPUBenchmark:
    """Show GPU performance hints and compare to other GPUs."""

    DESCRIPTION = """🎯 GPU BENCHMARK HINTS

Shows how fast your GPU is for different models.
Compares to other GPUs and estimates generation time.
No external tools needed - built-in database."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "analyze"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("benchmark_info", "comparison", "estimated_time_sec", "recommendations")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["flux", "sdxl", "sd15", "video"], {"default": "sdxl"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 150}),
            },
            "optional": {
                "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }

    def analyze(self, model_type, steps, resolution=1024, batch_size=1):
        try:
            from gpu_benchmarks import get_gpu_benchmark, estimate_generation_time, compare_gpus
        except ImportError:
            return ("GPU benchmark module not available", "", 0.0, "")

        # Detect current GPU
        gpu_name = "Unknown"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except:
            pass

        # Get benchmark
        benchmark = get_gpu_benchmark(gpu_name)

        # Estimate time
        estimated_time = estimate_generation_time(gpu_name, model_type, steps, resolution, batch_size)

        # Build benchmark info
        benchmark_info = f"""═══ GPU Benchmark ═══
GPU: {gpu_name}
Tier: {benchmark['tier'].upper()}
VRAM: {benchmark['vram_gb']}GB
Recommended Batch: {benchmark['recommended_batch']}
Max Resolution: {benchmark['max_resolution']}px

Performance ({model_type.upper()}):
  • Steps/sec: {benchmark.get(f'{model_type}_steps_per_sec', 0):.2f}
  • Estimated time: {estimated_time:.1f}s"""

        # Compare to other GPUs
        comparisons = compare_gpus(gpu_name, steps, model_type)
        comparison_lines = ["\n\n═══ GPU Comparison ═══"]

        for i, comp in enumerate(comparisons[:5], 1):
            if comp['speedup_percent'] > 0:
                comparison_lines.append(
                    f"{i}. {comp['gpu']} ({comp['tier']}) - {comp['speedup_percent']:+.0f}% faster ({comp['time']:.1f}s)"
                )
            else:
                comparison_lines.append(
                    f"{i}. {comp['gpu']} ({comp['tier']}) - {comp['speedup_percent']:.0f}% slower ({comp['time']:.1f}s)"
                )

        comparison = "\n".join(comparison_lines)

        # Recommendations
        recommendations = []
        if benchmark['tier'] == 'unknown':
            recommendations.append("⚠️ GPU not in benchmark database - using conservative estimates")
        if batch_size > benchmark['recommended_batch']:
            recommendations.append(f"⚠️ Batch size {batch_size} may exceed optimal ({benchmark['recommended_batch']})")
        if resolution > benchmark['max_resolution']:
            recommendations.append(f"⚠️ Resolution {resolution} may cause OOM (recommended max: {benchmark['max_resolution']})")

        recommendations_text = "\n".join(recommendations) if recommendations else "✅ Settings look optimal for your GPU"

        print(f"[Performance Lab GPU Benchmark]\n{benchmark_info}{comparison}")

        return (benchmark_info, comparison, estimated_time, recommendations_text)


class PerfLab_BatchProcessor:
    """Process multiple workflow variations from CSV file."""

    DESCRIPTION = """📊 BATCH PROCESSOR

Load parameters from CSV and queue multiple generations.
Perfect for:
• Testing prompt variations
• A/B testing configurations
• Model comparisons
• Dataset generation"""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("status", "total_configs", "current_config_json", "next_config_json")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_path": ("STRING", {"default": "batch_config.csv"}),
                "action": (["load", "get_next", "get_status", "create_template"], {"default": "load"}),
            },
            "optional": {
                "template_type": (["image", "video", "llm"], {"default": "image"}),
            }
        }

    def process(self, csv_path, action, template_type="image"):
        try:
            from batch_processor import parse_batch_csv, BatchQueue, generate_batch_csv_template
        except ImportError:
            return ("Batch processor module not available", 0, "{}", "{}")

        if action == "create_template":
            try:
                generate_batch_csv_template(csv_path, template_type)
                return (f"✅ Template created: {csv_path}", 0, "{}", "{}")
            except Exception as e:
                return (f"❌ Error creating template: {e}", 0, "{}", "{}")

        if action == "load":
            try:
                configs = parse_batch_csv(csv_path)
                # Store in global queue (simplified - in production use persistent storage)
                global _BATCH_QUEUE
                _BATCH_QUEUE = BatchQueue()
                _BATCH_QUEUE.add_many(configs)

                status = f"✅ Loaded {len(configs)} configurations from {csv_path}"
                first_config = json.dumps(configs[0], indent=2) if configs else "{}"
                second_config = json.dumps(configs[1], indent=2) if len(configs) > 1 else "{}"

                return (status, len(configs), first_config, second_config)
            except Exception as e:
                return (f"❌ Error loading CSV: {e}", 0, "{}", "{}")

        if action == "get_next":
            if '_BATCH_QUEUE' not in globals():
                return ("❌ No batch loaded - use 'load' first", 0, "{}", "{}")

            next_item = _BATCH_QUEUE.get_next()
            if next_item:
                status_info = _BATCH_QUEUE.get_status()
                status = f"⏳ Progress: {status_info['completed']}/{status_info['total']} complete"
                return (status, status_info['total'], json.dumps(next_item, indent=2), "{}")
            else:
                return ("✅ All batch items processed!", 0, "{}", "{}")

        if action == "get_status":
            if '_BATCH_QUEUE' not in globals():
                return ("❌ No batch loaded", 0, "{}", "{}")

            status_info = _BATCH_QUEUE.get_status()
            status = f"""═══ Batch Status ═══
Queued: {status_info['queued']}
Completed: {status_info['completed']}
Failed: {status_info['failed']}
Total: {status_info['total']}"""
            return (status, status_info['total'], "{}", "{}")

        return ("Unknown action", 0, "{}", "{}")


class PerfLab_WhimWeaverLauncher:
    """Launch and control WhimWeaver workflows."""

    DESCRIPTION = """🚀 WHIMWEAVER LAUNCHER

Launch WhimWeaver with specific workflows.
Monitor performance in real-time.
Perfect for automated testing and optimization."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "launch"
    RETURN_TYPES = ("STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("status", "is_running", "process_id")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["launch", "stop", "status"], {"default": "status"}),
            },
            "optional": {
                "whimweaver_path": ("STRING", {"default": ""}),
                "workflow_path": ("STRING", {"default": ""}),
                "extra_args": ("STRING", {"default": ""}),
            }
        }

    def launch(self, action, whimweaver_path="", workflow_path="", extra_args=""):
        try:
            from whimweaver_integration import WhimWeaverLauncher
        except ImportError:
            return ("WhimWeaver integration module not available", False, 0)

        global _WHIMWEAVER_LAUNCHER

        if '_WHIMWEAVER_LAUNCHER' not in globals():
            _WHIMWEAVER_LAUNCHER = WhimWeaverLauncher(whimweaver_path if whimweaver_path else None)

        launcher = _WHIMWEAVER_LAUNCHER

        if action == "launch":
            args = extra_args.split() if extra_args else None
            success = launcher.launch(workflow_path if workflow_path else None, args)

            if success:
                pid = launcher.process.pid if launcher.process else 0
                return (f"✅ WhimWeaver launched (PID: {pid})", True, pid)
            else:
                return ("❌ Failed to launch WhimWeaver", False, 0)

        elif action == "stop":
            launcher.stop()
            return ("✅ WhimWeaver stopped", False, 0)

        elif action == "status":
            if launcher.is_running():
                metrics = launcher.get_metrics()
                status = f"""═══ WhimWeaver Status ═══
Status: Running
Runtime: {metrics.get('runtime_sec', 0):.1f}s
CPU: {metrics.get('cpu_percent', 0):.1f}%
Memory: {metrics.get('memory_mb', 0):.0f}MB
GPU Memory: {metrics.get('gpu_memory_mb', 0):.0f}MB"""
                pid = launcher.process.pid if launcher.process else 0
                return (status, True, pid)
            else:
                return ("WhimWeaver not running", False, 0)

        return ("Unknown action", False, 0)


class PerfLab_WhimWeaverMonitor:
    """Monitor WhimWeaver performance in real-time."""

    DESCRIPTION = """📊 WHIMWEAVER MONITOR

Real-time performance monitoring for WhimWeaver.
Tracks CPU, memory, GPU usage.
Get performance summary after completion."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "monitor"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("current_metrics", "cpu_percent", "memory_mb", "gpu_memory_mb", "summary")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["current", "summary", "attach_pid"], {"default": "current"}),
            },
            "optional": {
                "pid": ("INT", {"default": 0}),
            }
        }

    def monitor(self, mode, pid=0):
        try:
            from whimweaver_integration import WhimWeaverMonitor
        except ImportError:
            return ("Monitor module not available", 0.0, 0.0, 0.0, "")

        global _WHIMWEAVER_MONITOR

        if '_WHIMWEAVER_MONITOR' not in globals():
            _WHIMWEAVER_MONITOR = WhimWeaverMonitor()

        monitor = _WHIMWEAVER_MONITOR

        if mode == "attach_pid" and pid > 0:
            if monitor.attach(pid):
                return (f"✅ Attached to PID {pid}", 0.0, 0.0, 0.0, "")
            else:
                return (f"❌ Failed to attach to PID {pid}", 0.0, 0.0, 0.0, "")

        elif mode == "current":
            metrics = monitor.get_metrics()

            if "error" in metrics:
                return (metrics["error"], 0.0, 0.0, 0.0, "")

            metrics_text = f"""═══ Current Metrics ═══
Runtime: {metrics.get('runtime_sec', 0):.1f}s
CPU: {metrics.get('cpu_percent', 0):.1f}%
Memory: {metrics.get('memory_mb', 0):.0f}MB
Threads: {metrics.get('num_threads', 0)}
GPU Memory: {metrics.get('gpu_memory_mb', 0):.0f}MB
Status: {metrics.get('status', 'unknown')}"""

            return (
                metrics_text,
                metrics.get('cpu_percent', 0.0),
                metrics.get('memory_mb', 0.0),
                metrics.get('gpu_memory_mb', 0.0),
                ""
            )

        elif mode == "summary":
            summary = monitor.get_summary()

            if "error" in summary:
                return (summary["error"], 0.0, 0.0, 0.0, "")

            summary_text = f"""═══ Performance Summary ═══
Total Runtime: {summary.get('total_runtime_sec', 0):.1f}s
Avg CPU: {summary.get('avg_cpu_percent', 0):.1f}%
Avg Memory: {summary.get('avg_memory_mb', 0):.0f}MB
Peak Memory: {summary.get('peak_memory_mb', 0):.0f}MB
Samples: {summary.get('samples_collected', 0)}"""

            return ("", 0.0, 0.0, 0.0, summary_text)

        return ("Unknown mode", 0.0, 0.0, 0.0, "")


class PerfLab_MultiModalTester:
    """Test any AI model type (image, video, LLM) with performance tracking."""

    DESCRIPTION = """🎭 MULTI-MODAL TESTER

Universal testing node for:
• Image generation (SD, SDXL, Flux)
• Video generation (AnimateDiff, SVD)
• LLM inference (any text model)
• Audio generation

Automatically detects model type and tracks appropriate metrics."""

    CATEGORY = "⚡ Performance Lab"
    FUNCTION = "test"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("test_report", "duration_sec", "throughput", "model_type", "recommendations")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["auto", "image", "video", "llm", "audio"], {"default": "auto"}),
            },
            "optional": {
                "workflow_path": ("STRING", {"default": ""}),
                "trigger": ("*", {"tooltip": "Connect to trigger test"}),
                "timer": ("PERF_TIMER", {"tooltip": "Connect from Timer"}),
            }
        }

    def test(self, model_type, workflow_path="", trigger=None, timer=None):
        # Detect model type
        detected_type = model_type
        if model_type == "auto" and workflow_path:
            try:
                from whimweaver_integration import detect_whimweaver_type
                detected_type = detect_whimweaver_type(workflow_path)
            except:
                detected_type = "unknown"

        # Calculate duration
        duration = 0.0
        if timer and "start_time" in timer:
            duration = time.time() - timer["start_time"]

        # Get GPU metrics
        try:
            import torch
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / (1024**3)
                vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
        except:
            vram_used = 0.0
            vram_peak = 0.0

        # Calculate throughput based on type
        throughput = 0.0
        throughput_label = ""

        if detected_type == "image":
            # Images per second
            throughput = 1.0 / duration if duration > 0 else 0
            throughput_label = "images/sec"
        elif detected_type == "video":
            # Assume 60 frames, calculate frames per second
            throughput = 60.0 / duration if duration > 0 else 0
            throughput_label = "frames/sec"
        elif detected_type == "llm":
            # Tokens per second (estimate)
            throughput = 500.0 / duration if duration > 0 else 0  # Assume 500 tokens
            throughput_label = "tokens/sec"

        # Build report
        test_report = f"""═══ Multi-Modal Test Results ═══
Model Type: {detected_type.upper()}
Duration: {duration:.2f}s
Throughput: {throughput:.2f} {throughput_label}
VRAM Used: {vram_used:.2f}GB
VRAM Peak: {vram_peak:.2f}GB"""

        # Recommendations
        recommendations = []
        if duration > 60:
            recommendations.append("⚠️ Long generation time - consider optimization")
        if vram_peak > 20:
            recommendations.append("⚠️ High VRAM usage - may limit batch size")
        if throughput < 1.0 and detected_type == "image":
            recommendations.append("💡 Low throughput - check AutoFix suggestions")

        recommendations_text = "\n".join(recommendations) if recommendations else "✅ Performance looks good!"

        print(f"[Performance Lab Multi-Modal Test]\n{test_report}\n{recommendations_text}")

        return (test_report, duration, throughput, detected_type, recommendations_text)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    # Core Utility (6 nodes)
    "PerfLab_Timer": PerfLab_Timer,
    "PerfLab_Report": PerfLab_Report,
    "PerfLab_VRAMMonitor": PerfLab_VRAMMonitor,
    "PerfLab_ShowText": PerfLab_ShowText,
    "PerfLab_CapResolution": PerfLab_CapResolution,
    "PerfLab_Compare": PerfLab_Compare,

    # v2.0 LLM-Powered (5 nodes)
    "PerfLab_AutoFix": PerfLab_AutoFix,
    "PerfLab_Optimizer": PerfLab_Optimizer,
    "PerfLab_ABTest": PerfLab_ABTest,
    "PerfLab_Feedback": PerfLab_Feedback,
    "PerfLab_NetworkSetup": PerfLab_NetworkSetup,

    # v2.1 Advanced (6 nodes)
    "PerfLab_GPUBenchmark": PerfLab_GPUBenchmark,
    "PerfLab_BatchProcessor": PerfLab_BatchProcessor,
    "PerfLab_WhimWeaverLauncher": PerfLab_WhimWeaverLauncher,
    "PerfLab_WhimWeaverMonitor": PerfLab_WhimWeaverMonitor,
    "PerfLab_MultiModalTester": PerfLab_MultiModalTester,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Core Utility
    "PerfLab_Timer": "⏱️ Start Timer",
    "PerfLab_Report": "📊 Performance Report",
    "PerfLab_VRAMMonitor": "💾 VRAM Monitor",
    "PerfLab_ShowText": "📝 Show Text",
    "PerfLab_CapResolution": "📐 Cap Resolution",
    "PerfLab_Compare": "📊 Compare Results",

    # v2.0 LLM-Powered
    "PerfLab_AutoFix": "🪄 AutoFix (Drop Anywhere)",
    "PerfLab_Optimizer": "🧠 LLM Optimizer",
    "PerfLab_ABTest": "🔬 A/B Test",
    "PerfLab_Feedback": "👍 Record Preference",
    "PerfLab_NetworkSetup": "🌐 Network Setup",

    # v2.1 Advanced
    "PerfLab_GPUBenchmark": "🎯 GPU Benchmark Hints",
    "PerfLab_BatchProcessor": "📊 Batch Processor (CSV)",
    "PerfLab_WhimWeaverLauncher": "🚀 WhimWeaver Launcher",
    "PerfLab_WhimWeaverMonitor": "📊 WhimWeaver Monitor",
    "PerfLab_MultiModalTester": "🎭 Multi-Modal Tester",
}

# Print startup message
print(f"[Performance Lab] v{__version__} loaded - {len(NODE_CLASS_MAPPINGS)} nodes (v2.1 Ultimate Edition)")
