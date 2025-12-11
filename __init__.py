"""
ComfyUI Performance Lab - Iterative workflow optimization with AI assistance.

This node pack provides:
- Performance monitoring nodes for tracking VRAM and execution time
- Integration with the Performance Lab CLI for advanced optimization
- Workflow analysis and smart suggestions

Install: Place in ComfyUI/custom_nodes/ComfyUI_PerformanceLab/
"""

import os
import sys
import json
import time
import subprocess
import threading
from typing import Dict, Any, Tuple, Optional

# Version
__version__ = "0.4.2"

# Ensure the module directory is in path for imports
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)


class PerformanceMonitor:
    """
    Monitor workflow execution performance.
    Tracks execution time and reports metrics after each run.
    """

    CATEGORY = "Performance Lab"
    FUNCTION = "monitor"
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("report", "duration_seconds")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": ("*",),  # Pass-through to trigger at end
                "label": ("STRING", {"default": "Generation", "multiline": False}),
            },
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    # Class variable to track start times
    _start_times: Dict[str, float] = {}

    def monitor(self, any_input, label: str, start_time: float = 0.0, unique_id: str = ""):
        end_time = time.time()

        # Calculate duration
        if start_time > 0:
            duration = end_time - start_time
        elif unique_id in self._start_times:
            duration = end_time - self._start_times[unique_id]
        else:
            duration = 0.0

        # Generate report
        report = f"{label}: {duration:.2f}s"

        # Print to console
        print(f"\n[Performance Lab] {report}")

        return (report, duration)


class PerformanceTimer:
    """
    Start a performance timer.
    Connect this node at the beginning of your workflow to track execution time.
    """

    CATEGORY = "Performance Lab"
    FUNCTION = "start_timer"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("start_time",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    def start_timer(self, unique_id: str = ""):
        start_time = time.time()
        PerformanceMonitor._start_times[unique_id] = start_time
        print(f"[Performance Lab] Timer started")
        return (start_time,)


class WorkflowAnalyzer:
    """
    Analyze a workflow and provide optimization suggestions.
    Drag a workflow JSON file or paste workflow content to analyze.
    """

    CATEGORY = "Performance Lab"
    FUNCTION = "analyze"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json": ("STRING", {
                    "multiline": True,
                    "default": "Paste workflow JSON here or leave empty to analyze current workflow"
                }),
            },
        }

    def analyze(self, workflow_json: str):
        try:
            # Try to parse the workflow
            if workflow_json.strip().startswith("{"):
                workflow = json.loads(workflow_json)
            else:
                return ("Paste a valid workflow JSON to analyze.",)

            # Basic analysis
            nodes = workflow.get("nodes", [])
            links = workflow.get("links", [])

            # Count node types
            node_types = {}
            for node in nodes:
                node_type = node.get("type", "Unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1

            # Detect features
            features = []
            node_type_lower = " ".join(node_types.keys()).lower()

            if "ksampler" in node_type_lower:
                features.append("Sampling")
            if "upscale" in node_type_lower or "esrgan" in node_type_lower:
                features.append("Upscaling")
            if "controlnet" in node_type_lower:
                features.append("ControlNet")
            if "lora" in node_type_lower:
                features.append("LoRA")
            if "sdxl" in node_type_lower or "xl" in node_type_lower:
                features.append("SDXL")
            if "flux" in node_type_lower:
                features.append("Flux")

            # Build report
            report_lines = [
                "=== Workflow Analysis ===",
                f"Total Nodes: {len(nodes)}",
                f"Total Links: {len(links)}",
                f"Features: {', '.join(features) if features else 'Basic'}",
                "",
                "Node Types:",
            ]

            for node_type, count in sorted(node_types.items(), key=lambda x: -x[1])[:10]:
                report_lines.append(f"  - {node_type}: {count}")

            report_lines.extend([
                "",
                "=== Suggestions ===",
                "Run the Performance Lab CLI for detailed optimization:",
                f"  cd {MODULE_DIR}",
                "  python performance_lab.py",
            ])

            report = "\n".join(report_lines)
            print(f"\n[Performance Lab]\n{report}")
            return (report,)

        except json.JSONDecodeError:
            return ("Invalid JSON. Please paste a valid workflow.",)
        except Exception as e:
            return (f"Error analyzing workflow: {e}",)


class LaunchPerformanceLab:
    """
    Launch the Performance Lab CLI in a new terminal.
    Use this to access the full optimization suite.
    """

    CATEGORY = "Performance Lab"
    FUNCTION = "launch"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
        }

    def launch(self, workflow_path: str):
        script_path = os.path.join(MODULE_DIR, "performance_lab.py")

        if not os.path.exists(script_path):
            return (f"Error: performance_lab.py not found at {script_path}",)

        try:
            # Try to open in a new terminal
            import platform
            system = platform.system()

            cmd = ["python", script_path]
            if workflow_path and os.path.exists(workflow_path):
                cmd.append(workflow_path)

            if system == "Linux":
                # Try common Linux terminals
                terminals = [
                    ["gnome-terminal", "--", *cmd],
                    ["konsole", "-e", *cmd],
                    ["xterm", "-e", *cmd],
                ]
                for term_cmd in terminals:
                    try:
                        subprocess.Popen(term_cmd, cwd=MODULE_DIR)
                        return (f"Launched Performance Lab in terminal",)
                    except FileNotFoundError:
                        continue
                # Fallback: run in background
                subprocess.Popen(cmd, cwd=MODULE_DIR)
                return (f"Launched Performance Lab (check console output)",)

            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", "-a", "Terminal", script_path], cwd=MODULE_DIR)
                return (f"Launched Performance Lab in Terminal",)

            elif system == "Windows":
                subprocess.Popen(["start", "cmd", "/k", "python", script_path],
                               shell=True, cwd=MODULE_DIR)
                return (f"Launched Performance Lab in Command Prompt",)

            return (f"Launch Performance Lab manually:\ncd {MODULE_DIR}\npython performance_lab.py",)

        except Exception as e:
            return (f"Error launching: {e}\n\nRun manually:\ncd {MODULE_DIR}\npython performance_lab.py",)


class ShowMetrics:
    """
    Display current system metrics from ComfyUI.
    Shows VRAM usage, queue status, and system info.
    """

    CATEGORY = "Performance Lab"
    FUNCTION = "show"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metrics",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    def show(self):
        try:
            import torch

            metrics_lines = ["=== System Metrics ==="]

            if torch.cuda.is_available():
                # Get GPU info
                gpu_name = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)

                metrics_lines.extend([
                    f"GPU: {gpu_name}",
                    f"VRAM Total: {vram_total:.2f} GB",
                    f"VRAM Allocated: {vram_used:.2f} GB",
                    f"VRAM Reserved: {vram_reserved:.2f} GB",
                    f"VRAM Free: {vram_total - vram_reserved:.2f} GB",
                ])
            else:
                metrics_lines.append("No CUDA GPU detected")

            metrics = "\n".join(metrics_lines)
            print(f"\n[Performance Lab]\n{metrics}")
            return (metrics,)

        except Exception as e:
            return (f"Error getting metrics: {e}",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PerformanceTimer": PerformanceTimer,
    "PerformanceMonitor": PerformanceMonitor,
    "WorkflowAnalyzer_PerfLab": WorkflowAnalyzer,
    "LaunchPerformanceLab": LaunchPerformanceLab,
    "ShowMetrics_PerfLab": ShowMetrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerformanceTimer": "Performance Timer (Start)",
    "PerformanceMonitor": "Performance Monitor (End)",
    "WorkflowAnalyzer_PerfLab": "Workflow Analyzer",
    "LaunchPerformanceLab": "Launch Performance Lab",
    "ShowMetrics_PerfLab": "Show System Metrics",
}

# Print startup message
print(f"\n{'='*50}")
print(f"  Performance Lab v{__version__} loaded!")
print(f"  5 nodes available in 'Performance Lab' category")
print(f"  CLI: python {os.path.join(MODULE_DIR, 'performance_lab.py')}")
print(f"{'='*50}\n")
