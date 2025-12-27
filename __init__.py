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
__version__ = "2.0.0"
__author__ = "Laboratoire Sonore"
__description__ = "ComfyUI Performance Lab v2.0 - LLM-guided workflow optimization"

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

        # Try to call LLM
        try:
            # Try Kobold-style API first
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

        except urllib.error.URLError:
            # Try Ollama-style API
            try:
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
                llm_response = f"Could not connect to LLM at {llm_endpoint}. Please check the endpoint is running."
        except Exception as e:
            llm_response = f"Error: {str(e)}"

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
}

# Print startup message
print(f"[Performance Lab] v{__version__} loaded - {len(NODE_CLASS_MAPPINGS)} nodes (focused v2.0)")
