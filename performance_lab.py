#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             ⚡ COMFYUI PERFORMANCE LAB v0.1 - ULTIMATE EDITION ⚡             ║
║     Iterative Workflow Optimization with LLM-Assisted Analysis & More!       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  NEW IN v0.1:                                                                ║
║  • Quick Actions - One-key optimizations (bypass, cap resolution, etc.)      ║
║  • Benchmark Mode - Multiple runs for reliable metrics                       ║
║  • Smart Suggestions - AI-free workflow analysis & recommendations           ║
║  • Presets System - 8GB VRAM, Speed Test, Quality, Custom presets            ║
║  • A/B Comparison - Compare two workflow versions side-by-side               ║
║  • Progress Dashboard - Visual history of all optimizations                  ║
║  • Workflow Diff - See exactly what changed                                  ║
║  • ComfyUI Manager Integration - Auto-detect missing nodes                   ║
║  • Multi-Platform Clipboard - Windows, macOS, Linux support                  ║
║  • Configuration Persistence - Save settings between sessions                ║
║  • Built-in Mods Library - Common optimizations included                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import copy
import hashlib
import importlib.util
import urllib.request
import urllib.error
import subprocess
import platform
from datetime import datetime
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List, Any, Callable

# Try to import model_tuner for smart model detection
try:
    from model_tuner import (
        detect_model, optimize_workflow, get_model_presets,
        ModelDetector, ModelOptimizer, LoRATuner, SamplerRecommender,
        MODEL_PROFILES, ModelType
    )
    MODEL_TUNER_AVAILABLE = True
except ImportError:
    MODEL_TUNER_AVAILABLE = False

# Try to import workflow utilities for fingerprinting and beautification
try:
    from workflow_utils import (
        WorkflowFingerprinter, WorkflowFingerprint, ModelFamily,
        WorkflowBeautifier, BeautifyMode, WorkflowDiff,
        is_safe_to_overwrite, suggest_filename
    )
    WORKFLOW_UTILS_AVAILABLE = True
except ImportError:
    WORKFLOW_UTILS_AVAILABLE = False

# Try to import LLM enhancer for advanced LLM integration
try:
    from llm_enhancer import (
        LLMContextBuilder, NodeCatalog, SystemSpecsCollector,
        PromptGenerator, PromptGoal, ModValidator, ValidationResult,
        ConversationMemory, KnowledgeBase, ErrorHistory,
        WorkflowGraphExporter, PROMPT_TEMPLATES
    )
    LLM_ENHANCER_AVAILABLE = True
except ImportError:
    LLM_ENHANCER_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "0.3.0"
MODS_DIR = "mods"
COMFY_URL = "http://127.0.0.1:8188"
SESSION_LOG = "session_history.json"
CONFIG_FILE = "performance_lab_config.json"
BENCHMARK_RUNS = 3

# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL STYLING (Enhanced for v0.1)
# ═══════════════════════════════════════════════════════════════════════════════

class Style:
    """ANSI escape codes for terminal styling."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Background Colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"

    # Box drawing
    BOX_H = "─"
    BOX_V = "│"
    BOX_TL = "┌"
    BOX_TR = "┐"
    BOX_BL = "└"
    BOX_BR = "┘"
    BOX_T = "┬"
    BOX_B = "┴"
    BOX_L = "├"
    BOX_R = "┤"
    BOX_X = "┼"

    # Double box
    DBOX_H = "═"
    DBOX_V = "║"
    DBOX_TL = "╔"
    DBOX_TR = "╗"
    DBOX_BL = "╚"
    DBOX_BR = "╝"

    # Progress
    PROGRESS_FULL = "█"
    PROGRESS_EMPTY = "░"
    PROGRESS_HALF = "▓"

def styled(text: str, *styles) -> str:
    """Apply multiple styles to text."""
    prefix = "".join(styles)
    return f"{prefix}{text}{Style.RESET}" if prefix else text

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header(title: str, subtitle: str = "", version: str = ""):
    """Print a beautiful boxed header."""
    width = 78
    print()
    print(styled(Style.DBOX_TL + Style.DBOX_H * (width - 2) + Style.DBOX_TR, Style.CYAN, Style.BOLD))

    # Title with lightning bolts
    title_display = f"⚡ {title} ⚡"
    print(styled(Style.DBOX_V, Style.CYAN) + styled(title_display.center(width - 2), Style.BOLD, Style.WHITE) + styled(Style.DBOX_V, Style.CYAN))

    if subtitle:
        print(styled(Style.DBOX_V, Style.CYAN) + styled(subtitle.center(width - 2), Style.DIM) + styled(Style.DBOX_V, Style.CYAN))

    if version:
        print(styled(Style.DBOX_V, Style.CYAN) + styled(f"v{version}".center(width - 2), Style.YELLOW) + styled(Style.DBOX_V, Style.CYAN))

    print(styled(Style.DBOX_BL + Style.DBOX_H * (width - 2) + Style.DBOX_BR, Style.CYAN, Style.BOLD))

def print_box(title: str, content: List[str], color=Style.WHITE, icon: str = ""):
    """Print content in a nice box."""
    width = 76
    title_display = f"{icon} {title}" if icon else title
    print()
    print(styled(f"  {Style.BOX_TL}{Style.BOX_H} {title_display} {Style.BOX_H * (width - len(title_display) - 5)}{Style.BOX_TR}", color))
    for line in content:
        truncated = line[:width-4] if len(line) > width-4 else line
        padding = " " * (width - 4 - len(truncated))
        print(styled(f"  {Style.BOX_V} ", color) + truncated + padding + styled(f" {Style.BOX_V}", color))
    print(styled(f"  {Style.BOX_BL}{Style.BOX_H * (width - 2)}{Style.BOX_BR}", color))

def print_divider(char="─", color=Style.GRAY, width=76):
    print(styled(f"  {char * width}", color))

def print_progress_bar(current: float, total: float, width: int = 40, prefix: str = "", suffix: str = ""):
    """Print a progress bar."""
    if total == 0:
        total = 1
    percent = min(current / total, 1.0)
    filled = int(width * percent)
    bar = Style.PROGRESS_FULL * filled + Style.PROGRESS_EMPTY * (width - filled)
    pct_str = f"{percent * 100:.1f}%"
    print(f"  {prefix} {styled(bar, Style.CYAN)} {pct_str} {suffix}", end="\r")

def print_mini_chart(data: List[float], width: int = 30, label: str = ""):
    """Print a mini ASCII chart."""
    if not data:
        return
    max_val = max(data) if data else 1
    min_val = min(data) if data else 0
    range_val = max_val - min_val if max_val != min_val else 1

    chars = " ▁▂▃▄▅▆▇█"

    chart = ""
    for val in data[-width:]:
        idx = int((val - min_val) / range_val * (len(chars) - 1))
        chart += chars[idx]

    print(f"  {label}: {styled(chart, Style.CYAN)} ({min_val:.1f}-{max_val:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# CLIPBOARD UTILITIES (Multi-Platform)
# ═══════════════════════════════════════════════════════════════════════════════

class Clipboard:
    """Multi-platform clipboard handling."""

    @staticmethod
    def copy(text: str) -> bool:
        """Copy text to clipboard. Returns True if successful."""
        system = platform.system()

        try:
            if system == 'Windows':
                # Windows: use clip command
                process = subprocess.Popen(['clip'], stdin=subprocess.PIPE, shell=True)
                process.communicate(text.encode('utf-16-le'))
                return process.returncode == 0
            elif system == 'Darwin':
                # macOS: use pbcopy
                process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                process.communicate(text.encode())
                return process.returncode == 0
            else:
                # Linux: try multiple clipboard tools
                for cmd in [['xclip', '-selection', 'clipboard'], ['xsel', '--clipboard', '--input'], ['wl-copy']]:
                    try:
                        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                        process.communicate(text.encode())
                        if process.returncode == 0:
                            return True
                    except FileNotFoundError:
                        continue
        except Exception:
            pass

        return False

    @staticmethod
    def paste() -> Optional[str]:
        """Get text from clipboard. Returns None if failed."""
        system = platform.system()

        try:
            if system == 'Windows':
                process = subprocess.Popen(['powershell', '-command', 'Get-Clipboard'],
                                          stdout=subprocess.PIPE, shell=True)
                stdout, _ = process.communicate()
                return stdout.decode().strip()
            elif system == 'Darwin':
                process = subprocess.Popen(['pbpaste'], stdout=subprocess.PIPE)
                stdout, _ = process.communicate()
                return stdout.decode()
            else:
                for cmd in [['xclip', '-selection', 'clipboard', '-o'], ['xsel', '--clipboard', '--output'], ['wl-paste']]:
                    try:
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                        stdout, _ = process.communicate()
                        if process.returncode == 0:
                            return stdout.decode()
                    except FileNotFoundError:
                        continue
        except Exception:
            pass

        return None

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """Manages persistent configuration."""

    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self._load()

    def _load(self) -> Dict:
        """Load configuration from file."""
        defaults = {
            "comfy_url": COMFY_URL,
            "benchmark_runs": BENCHMARK_RUNS,
            "last_workflow": None,
            "last_goal": "",
            "presets": {},
            "quick_action_history": [],
            "theme": "default",
        }
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    defaults.update(loaded)
        except Exception:
            pass
        return defaults

    def save(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value):
        self.config[key] = value
        self.save()

# ═══════════════════════════════════════════════════════════════════════════════
# COMFYUI API INTERFACE (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

class ComfyUIMonitor:
    """Handles all ComfyUI API interactions and metric collection."""

    def __init__(self, base_url: str = COMFY_URL):
        self.base_url = base_url
        self.connected = False
        self._last_vram_readings: List[float] = []

    def api_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
        """Make an API request to ComfyUI."""
        try:
            url = f"{self.base_url}/{endpoint}"
            if data and method == "POST":
                req = urllib.request.Request(url,
                                            data=json.dumps(data).encode(),
                                            headers={'Content-Type': 'application/json'},
                                            method='POST')
            else:
                req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode())
        except Exception:
            return None

    def check_connection(self) -> bool:
        """Test if ComfyUI is running and accessible."""
        result = self.api_request("system_stats")
        self.connected = result is not None
        return self.connected

    def get_system_stats(self) -> Optional[Dict]:
        """Get current system stats including VRAM usage."""
        return self.api_request("system_stats")

    def get_queue_status(self) -> Tuple[int, int]:
        """Returns (pending_count, running_count)."""
        data = self.api_request("queue")
        if not data:
            return 0, 0
        return len(data.get('queue_pending', [])), len(data.get('queue_running', []))

    def get_history(self, max_items: int = 100) -> Dict:
        """Get execution history."""
        return self.api_request(f"history?max_items={max_items}") or {}

    def get_object_info(self) -> Optional[Dict]:
        """Get all available node types (for manager integration)."""
        return self.api_request("object_info")

    def interrupt(self) -> bool:
        """Interrupt current generation."""
        result = self.api_request("interrupt", method="POST")
        return result is not None

    def collect_metrics(self) -> Dict:
        """Collect all available metrics from ComfyUI."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "connected": False,
            "vram_used_gb": None,
            "vram_total_gb": None,
            "vram_percent": None,
            "vram_free_gb": None,
            "queue_pending": 0,
            "queue_running": 0,
            "gpu_name": None,
        }

        stats = self.get_system_stats()
        if stats:
            metrics["connected"] = True
            devices = stats.get("devices", [])
            if devices:
                gpu = devices[0]
                vram_used = gpu.get("vram_used", 0)
                vram_total = gpu.get("vram_total", 1)
                vram_free = gpu.get("vram_free", vram_total - vram_used)
                metrics["vram_used_gb"] = round(vram_used / (1024**3), 2)
                metrics["vram_total_gb"] = round(vram_total / (1024**3), 2)
                metrics["vram_free_gb"] = round(vram_free / (1024**3), 2)
                metrics["vram_percent"] = round(100 * vram_used / vram_total, 1)
                metrics["gpu_name"] = gpu.get("name", "Unknown GPU")

        pending, running = self.get_queue_status()
        metrics["queue_pending"] = pending
        metrics["queue_running"] = running

        return metrics

    def wait_for_generation(self, show_progress: bool = True) -> Dict:
        """Wait for a generation to complete and collect metrics."""
        result = {
            "success": None,
            "duration_seconds": 0,
            "error_message": None,
            "peak_vram_gb": 0,
            "baseline_vram_gb": 0,
            "avg_vram_gb": 0,
            "node_errors": [],
            "prompt_id": None,
            "vram_readings": [],
        }

        if show_progress:
            print(f"\n  {styled('📡', Style.CYAN)} Connecting to ComfyUI at {styled(self.base_url, Style.BLUE)}...")

        baseline = self.collect_metrics()
        if not baseline["connected"]:
            if show_progress:
                print(f"  {styled('⚠', Style.YELLOW)}  ComfyUI not detected. Is it running?")
            result["error_message"] = "ComfyUI not connected"
            return result

        result["baseline_vram_gb"] = baseline.get("vram_used_gb", 0)
        initial_history_ids = set(self.get_history().keys())

        if show_progress:
            print(f"  {styled('⏳', Style.YELLOW)} Waiting for generation...")
            print(f"     {styled('Load the _experimental file and click Queue Prompt', Style.DIM)}")
            print(f"     {styled('Press Ctrl+C to abort', Style.DIM)}")

        start_time = None
        is_running = False
        vram_readings = []
        peak_vram = baseline.get("vram_used_gb", 0)

        try:
            while True:
                time.sleep(0.25)

                current = self.collect_metrics()
                if not current["connected"]:
                    continue

                # Track VRAM
                if current["vram_used_gb"]:
                    vram_readings.append(current["vram_used_gb"])
                    peak_vram = max(peak_vram, current["vram_used_gb"])

                pending = current["queue_pending"]
                running = current["queue_running"]

                # Detect start
                if not is_running and (pending > 0 or running > 0):
                    if show_progress:
                        print(f"\n  {styled('🚀', Style.GREEN)} Generation STARTED!")
                    start_time = time.time()
                    is_running = True
                    vram_readings = []  # Reset readings at start

                # Show progress
                if is_running and show_progress:
                    elapsed = time.time() - start_time
                    vram_str = f"{current['vram_used_gb']:.1f}GB" if current['vram_used_gb'] else "N/A"
                    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(elapsed * 4) % 10]
                    print(f"\r  {styled(spinner, Style.CYAN)}  Running... {elapsed:.1f}s | VRAM: {vram_str} | Peak: {peak_vram:.1f}GB    ", end="", flush=True)

                # Detect finish
                current_history = self.get_history()
                current_ids = set(current_history.keys())
                new_ids = current_ids - initial_history_ids

                if is_running and new_ids:
                    if show_progress:
                        print()  # Newline
                    end_time = time.time()
                    result["duration_seconds"] = round(end_time - start_time, 2)
                    result["peak_vram_gb"] = round(peak_vram, 2)
                    result["vram_readings"] = vram_readings
                    result["avg_vram_gb"] = round(sum(vram_readings) / len(vram_readings), 2) if vram_readings else 0

                    # Check result
                    latest_id = sorted(new_ids)[-1]
                    result["prompt_id"] = latest_id
                    run_data = current_history.get(latest_id, {})

                    status = run_data.get("status", {})
                    status_str = status.get("status_str", "success")

                    if status_str == "error":
                        result["success"] = False
                        messages = status.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, list) and len(msg) >= 2:
                                if msg[0] == "execution_error":
                                    error_info = msg[1]
                                    result["error_message"] = error_info.get("exception_message", "Unknown error")
                                    result["node_errors"].append({
                                        "node_id": error_info.get("node_id"),
                                        "node_type": error_info.get("node_type"),
                                        "message": error_info.get("exception_message"),
                                    })
                    else:
                        result["success"] = True

                    return result

        except KeyboardInterrupt:
            if show_progress:
                print(f"\n  {styled('🛑', Style.RED)} Aborted by user")
            result["success"] = None
            result["error_message"] = "Aborted by user"
            return result

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW ANALYZER (Enhanced for v0.1)
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowAnalyzer:
    """Analyzes ComfyUI workflow structure for optimization."""

    # Known VRAM-heavy node patterns
    VRAM_HEAVY_PATTERNS = [
        "upscale", "esrgan", "4x", "8x", "video", "animate", "wan", "svd",
        "animatediff", "controlnet", "ipadapter", "instantid", "inpaint",
        "outpaint", "flux", "sdxl", "cascade"
    ]

    # Known node categories for suggestions
    NODE_CATEGORIES = {
        "sampler": ["ksampler", "sampler", "sample"],
        "model_loader": ["checkpoint", "model", "loader", "unet"],
        "vae": ["vae", "encode", "decode"],
        "upscaler": ["upscale", "esrgan", "4x", "8x", "ultimate"],
        "video": ["video", "animate", "animatediff", "vhs", "frame"],
        "controlnet": ["controlnet", "preprocessor", "openpose", "depth"],
        "ip_adapter": ["ipadapter", "instantid", "faceswap"],
    }

    @staticmethod
    def analyze(workflow: Dict) -> Dict:
        """Extract comprehensive information from a workflow."""
        analysis = {
            "node_count": 0,
            "node_types": {},
            "node_ids": [],
            "groups": [],
            "features": {
                "has_video": False,
                "has_upscale": False,
                "has_controlnet": False,
                "has_ip_adapter": False,
                "has_inpaint": False,
                "has_sdxl": False,
                "has_flux": False,
            },
            "vram_heavy_nodes": [],
            "model_loaders": [],
            "samplers": [],
            "resolution_hints": [],
            "step_counts": [],
            "batch_sizes": [],
            "connections_count": 0,
            "widget_stats": {},
            "optimization_potential": "low",
            "estimated_vram_gb": 4,  # Base estimate
        }

        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        groups = workflow.get("groups", [])

        analysis["node_count"] = len(nodes)
        analysis["connections_count"] = len(links)

        for node in nodes:
            node_id = node.get("id", 0)
            node_type = node.get("type", "Unknown")
            node_title = node.get("title", node_type)
            type_lower = node_type.lower()

            analysis["node_ids"].append(node_id)
            analysis["node_types"][node_type] = analysis["node_types"].get(node_type, 0) + 1

            # Check for VRAM-heavy nodes
            for pattern in WorkflowAnalyzer.VRAM_HEAVY_PATTERNS:
                if pattern in type_lower:
                    analysis["vram_heavy_nodes"].append({
                        "id": node_id,
                        "type": node_type,
                        "title": node_title,
                    })
                    break

            # Feature detection
            if "video" in type_lower or "vhs" in type_lower or "animate" in type_lower:
                analysis["features"]["has_video"] = True
            if "upscale" in type_lower or "esrgan" in type_lower:
                analysis["features"]["has_upscale"] = True
            if "controlnet" in type_lower:
                analysis["features"]["has_controlnet"] = True
            if "ipadapter" in type_lower or "instantid" in type_lower:
                analysis["features"]["has_ip_adapter"] = True
            if "inpaint" in type_lower:
                analysis["features"]["has_inpaint"] = True
            if "sdxl" in type_lower:
                analysis["features"]["has_sdxl"] = True
            if "flux" in type_lower:
                analysis["features"]["has_flux"] = True

            # Identify key nodes
            if "loader" in type_lower and ("checkpoint" in type_lower or "model" in type_lower):
                analysis["model_loaders"].append({"id": node_id, "type": node_type})
            if "sampler" in type_lower or "ksampler" in type_lower:
                analysis["samplers"].append({"id": node_id, "type": node_type})

            # Analyze widget values
            widgets = node.get("widgets_values", [])
            for i, w in enumerate(widgets):
                if isinstance(w, int):
                    # Resolution detection (256-4096, divisible by 8)
                    if 256 <= w <= 4096 and w % 8 == 0:
                        if w not in analysis["resolution_hints"]:
                            analysis["resolution_hints"].append(w)
                    # Steps detection (typically 1-150)
                    if 1 <= w <= 150 and "sampler" in type_lower:
                        analysis["step_counts"].append(w)
                    # Batch size detection (1-64)
                    if 1 <= w <= 64 and i == len(widgets) - 1:  # Often last widget
                        analysis["batch_sizes"].append(w)

        # Groups
        for group in groups:
            analysis["groups"].append({
                "title": group.get("title", "Untitled"),
                "color": group.get("color", ""),
            })

        # Estimate VRAM and optimization potential
        analysis["estimated_vram_gb"] = WorkflowAnalyzer._estimate_vram(analysis)
        analysis["optimization_potential"] = WorkflowAnalyzer._assess_potential(analysis)

        return analysis

    @staticmethod
    def _estimate_vram(analysis: Dict) -> float:
        """Estimate VRAM usage based on workflow analysis."""
        base = 4.0  # Base model

        if analysis["features"]["has_sdxl"]:
            base = 6.5
        if analysis["features"]["has_flux"]:
            base = 8.0

        # Add for features
        if analysis["features"]["has_controlnet"]:
            base += 1.5
        if analysis["features"]["has_ip_adapter"]:
            base += 2.0
        if analysis["features"]["has_video"]:
            base += 3.0
        if analysis["features"]["has_upscale"]:
            base += 2.0

        # Resolution impact
        max_res = max(analysis["resolution_hints"]) if analysis["resolution_hints"] else 512
        if max_res > 1024:
            base *= 1.5
        if max_res > 1536:
            base *= 1.5

        return round(base, 1)

    @staticmethod
    def _assess_potential(analysis: Dict) -> str:
        """Assess optimization potential."""
        score = 0

        if analysis["features"]["has_upscale"]:
            score += 2
        if len(analysis["vram_heavy_nodes"]) > 3:
            score += 2
        if analysis["resolution_hints"] and max(analysis["resolution_hints"]) > 1024:
            score += 1
        if analysis["step_counts"] and max(analysis["step_counts"]) > 30:
            score += 1
        if analysis["batch_sizes"] and max(analysis["batch_sizes"]) > 1:
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"

    @staticmethod
    def summarize(analysis: Dict) -> str:
        """Create a human-readable summary."""
        lines = []
        lines.append(f"• {analysis['node_count']} nodes, {analysis['connections_count']} connections")

        # Top node types
        sorted_types = sorted(analysis['node_types'].items(), key=lambda x: -x[1])[:6]
        type_str = ", ".join([f"{t}({c})" for t, c in sorted_types])
        lines.append(f"• Top types: {type_str}")

        # Features
        features = []
        for feat, enabled in analysis["features"].items():
            if enabled:
                feat_name = feat.replace("has_", "").replace("_", " ").title()
                features.append(feat_name)
        if features:
            lines.append(f"• Features: {', '.join(features)}")

        # VRAM-heavy nodes
        if analysis["vram_heavy_nodes"]:
            heavy = [n["type"] for n in analysis["vram_heavy_nodes"][:5]]
            lines.append(f"• VRAM-heavy: {', '.join(heavy)}")

        # Estimates
        lines.append(f"• Est. VRAM: ~{analysis['estimated_vram_gb']} GB")
        lines.append(f"• Optimization potential: {analysis['optimization_potential'].upper()}")

        if analysis["resolution_hints"]:
            lines.append(f"• Resolutions: {sorted(analysis['resolution_hints'])}")

        return "\n".join(lines)

    @staticmethod
    def get_smart_suggestions(analysis: Dict) -> List[Dict]:
        """Generate smart optimization suggestions without LLM."""
        suggestions = []

        # High resolution warning
        if analysis["resolution_hints"]:
            max_res = max(analysis["resolution_hints"])
            if max_res > 1024:
                suggestions.append({
                    "priority": "high",
                    "type": "resolution",
                    "title": f"Cap resolution from {max_res}px to 1024px",
                    "reason": "Higher resolutions use quadratically more VRAM",
                    "impact": "~40% VRAM reduction",
                    "quick_action": "cap_resolution_1024",
                })
            if max_res > 1536:
                suggestions.append({
                    "priority": "critical",
                    "type": "resolution",
                    "title": f"Cap resolution from {max_res}px to 768px for testing",
                    "reason": "Very high resolution - cap to 768 for faster iteration",
                    "impact": "~60% faster generation",
                    "quick_action": "cap_resolution_768",
                })

        # Upscaler bypass
        if analysis["features"]["has_upscale"]:
            suggestions.append({
                "priority": "high",
                "type": "upscaler",
                "title": "Bypass upscalers during testing",
                "reason": "Upscalers are VRAM-heavy and not needed for iteration",
                "impact": "2-4GB VRAM saved, 3-5x faster",
                "quick_action": "bypass_upscalers",
            })

        # High step count
        if analysis["step_counts"]:
            max_steps = max(analysis["step_counts"])
            if max_steps > 30:
                suggestions.append({
                    "priority": "medium",
                    "type": "steps",
                    "title": f"Reduce steps from {max_steps} to 20 for testing",
                    "reason": "20 steps often sufficient for testing composition",
                    "impact": f"~{int((max_steps - 20) / max_steps * 100)}% faster",
                    "quick_action": "reduce_steps_20",
                })

        # Batch size reduction
        if analysis["batch_sizes"]:
            max_batch = max(analysis["batch_sizes"])
            if max_batch > 1:
                suggestions.append({
                    "priority": "high",
                    "type": "batch",
                    "title": f"Reduce batch size from {max_batch} to 1",
                    "reason": "Batch processing multiplies VRAM usage",
                    "impact": f"~{int((max_batch - 1) / max_batch * 100)}% VRAM reduction",
                    "quick_action": "reduce_batch_1",
                })

        # Video workflow optimization
        if analysis["features"]["has_video"]:
            suggestions.append({
                "priority": "high",
                "type": "video",
                "title": "Reduce frame count for testing",
                "reason": "Video generation is extremely VRAM-intensive",
                "impact": "Significant VRAM and time savings",
                "quick_action": "reduce_frames",
            })

        # ControlNet optimization
        if analysis["features"]["has_controlnet"]:
            suggestions.append({
                "priority": "medium",
                "type": "controlnet",
                "title": "Reduce ControlNet strength for testing",
                "reason": "Lower strength = faster processing",
                "impact": "Moderate speed improvement",
                "quick_action": "reduce_controlnet",
            })

        # SDXL specific
        if analysis["features"]["has_sdxl"]:
            suggestions.append({
                "priority": "info",
                "type": "model",
                "title": "Consider SD 1.5 for rapid iteration",
                "reason": "SDXL uses ~2x VRAM vs SD 1.5",
                "impact": "~50% VRAM reduction (requires workflow change)",
                "quick_action": None,
            })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 5))

        return suggestions

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION HISTORY TRACKER (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

class SessionHistory:
    """Tracks all modifications and their results within a session."""

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries: List[Dict] = []
        self.baseline_metrics: Optional[Dict] = None
        self.target_workflow: Optional[str] = None
        self.workflow_analysis: Optional[Dict] = None
        self.baseline_result: Optional[Dict] = None
        self.benchmark_results: List[Dict] = []

    def set_baseline(self, workflow_path: str, analysis: Dict, metrics: Dict):
        """Set the baseline for comparison."""
        self.target_workflow = workflow_path
        self.workflow_analysis = analysis
        self.baseline_metrics = metrics

    def set_baseline_result(self, result: Dict):
        """Set baseline generation result for comparison."""
        self.baseline_result = result

    def add_entry(self, mod_name: str, mod_description: str, result: Dict, kept: bool,
                  diff: Optional[Dict] = None):
        """Add a modification entry."""
        entry = {
            "iteration": len(self.entries) + 1,
            "timestamp": datetime.now().isoformat(),
            "mod_name": mod_name,
            "mod_description": mod_description,
            "result": result,
            "kept": kept,
            "diff": diff,
        }

        # Compare to baseline
        if self.baseline_result and result.get("duration_seconds"):
            baseline_dur = self.baseline_result.get("duration_seconds", 0)
            if baseline_dur > 0:
                change = ((result["duration_seconds"] - baseline_dur) / baseline_dur) * 100
                entry["speed_change_percent"] = round(change, 1)

        self.entries.append(entry)

    def add_benchmark(self, name: str, runs: List[Dict]):
        """Add benchmark results."""
        if not runs:
            return

        durations = [r["duration_seconds"] for r in runs if r.get("success")]
        if durations:
            self.benchmark_results.append({
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "runs": len(durations),
                "avg": round(sum(durations) / len(durations), 2),
                "min": round(min(durations), 2),
                "max": round(max(durations), 2),
                "std": round(self._std_dev(durations), 2),
            })

    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def get_summary(self) -> str:
        """Get a summary of all modifications."""
        if not self.entries:
            return "No modifications applied yet."

        lines = []
        for e in self.entries:
            status = styled("✓ KEPT", Style.GREEN) if e["kept"] else styled("✗ REVERTED", Style.RED)
            duration = e["result"].get("duration_seconds", "N/A")
            success = styled("OK", Style.GREEN) if e["result"].get("success") else styled("FAIL", Style.RED)

            speed_change = ""
            if "speed_change_percent" in e:
                pct = e["speed_change_percent"]
                if pct < 0:
                    speed_change = styled(f" ({pct:.1f}%)", Style.GREEN)
                else:
                    speed_change = styled(f" (+{pct:.1f}%)", Style.RED)

            lines.append(f"  {e['iteration']}. [{status}] {e['mod_name']} → {success} ({duration}s){speed_change}")

        return "\n".join(lines)

    def get_performance_trend(self) -> List[Tuple[str, float]]:
        """Get performance trend of kept modifications."""
        trend = []
        for e in self.entries:
            if e["kept"] and e["result"].get("success"):
                trend.append((e["mod_name"], e["result"]["duration_seconds"]))
        return trend

    def to_dict(self) -> Dict:
        """Export session as dictionary."""
        return {
            "session_id": self.session_id,
            "target_workflow": self.target_workflow,
            "workflow_analysis": self.workflow_analysis,
            "baseline_metrics": self.baseline_metrics,
            "baseline_result": self.baseline_result,
            "entries": self.entries,
            "benchmark_results": self.benchmark_results,
        }

    def export(self, filepath: str) -> bool:
        """Export session to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception:
            return False

# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROMPT GENERATOR (Enhanced with multi-LLM support)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMPromptGenerator:
    """Generates copy-paste prompts for external LLM analysis."""

    LLM_PRESETS = {
        "claude": {
            "name": "Claude",
            "style": "detailed",
            "prefix": "I'm working on optimizing a ComfyUI workflow. Please analyze this and provide specific, actionable suggestions.\n\n",
        },
        "gpt4": {
            "name": "GPT-4",
            "style": "structured",
            "prefix": "Task: Analyze and optimize this ComfyUI workflow.\nOutput: Python mod code.\n\n",
        },
        "gemini": {
            "name": "Gemini",
            "style": "concise",
            "prefix": "Optimize this ComfyUI workflow. Respond with Python mod code:\n\n",
        },
        "llama": {
            "name": "Llama/Mistral",
            "style": "explicit",
            "prefix": "You are a ComfyUI workflow optimization expert. Analyze this workflow and output a Python mod.\n\n",
        },
    }

    @staticmethod
    def generate_analysis_prompt(
        session: SessionHistory,
        latest_result: Dict,
        workflow_content: Dict,
        user_goal: str = "",
        llm_type: str = "claude"
    ) -> str:
        """Generate a comprehensive prompt for LLM analysis."""

        preset = LLMPromptGenerator.LLM_PRESETS.get(llm_type, LLMPromptGenerator.LLM_PRESETS["claude"])

        prompt_parts = []

        # LLM-specific prefix
        prompt_parts.append(preset["prefix"])

        # Header
        prompt_parts.append("=" * 70)
        prompt_parts.append("COMFYUI WORKFLOW OPTIMIZATION REQUEST")
        prompt_parts.append("=" * 70)
        prompt_parts.append("")

        # Context
        prompt_parts.append("## CONTEXT")
        prompt_parts.append(f"I'm optimizing a ComfyUI workflow using an iterative mod system.")
        if user_goal:
            prompt_parts.append(f"**My Goal:** {user_goal}")
        prompt_parts.append("")

        # Workflow analysis
        if session.workflow_analysis:
            prompt_parts.append("## WORKFLOW STRUCTURE")
            prompt_parts.append(WorkflowAnalyzer.summarize(session.workflow_analysis))
            prompt_parts.append("")

        # Latest result
        prompt_parts.append("## LATEST TEST RESULT")
        if latest_result.get("success"):
            prompt_parts.append(f"✓ Status: SUCCESS")
            prompt_parts.append(f"• Duration: {latest_result.get('duration_seconds', 'N/A')} seconds")
            prompt_parts.append(f"• Peak VRAM: {latest_result.get('peak_vram_gb', 'N/A')} GB")
            prompt_parts.append(f"• Avg VRAM: {latest_result.get('avg_vram_gb', 'N/A')} GB")
            prompt_parts.append(f"• Baseline VRAM: {latest_result.get('baseline_vram_gb', 'N/A')} GB")
        elif latest_result.get("success") is False:
            prompt_parts.append(f"✗ Status: FAILED")
            prompt_parts.append(f"• Error: {latest_result.get('error_message', 'Unknown')}")
            if latest_result.get("node_errors"):
                for err in latest_result["node_errors"]:
                    prompt_parts.append(f"  - Node {err.get('node_id')} ({err.get('node_type')}): {err.get('message')}")
        else:
            prompt_parts.append("• Status: NOT YET TESTED")
        prompt_parts.append("")

        # Modification history
        if session.entries:
            prompt_parts.append("## MODIFICATION HISTORY")
            for e in session.entries[-5:]:  # Last 5 entries
                status = "KEPT" if e["kept"] else "REVERTED"
                result = e["result"]
                if result.get("success"):
                    perf = f"{result['duration_seconds']}s"
                elif result.get("success") is False:
                    perf = f"FAILED: {result.get('error_message', 'Unknown')[:50]}"
                else:
                    perf = "Not tested"
                prompt_parts.append(f"{e['iteration']}. [{status}] {e['mod_name']}: {e['mod_description'][:60]}")
                prompt_parts.append(f"   Result: {perf}")
            prompt_parts.append("")

        # Node type distribution
        if session.workflow_analysis:
            prompt_parts.append("## KEY NODE TYPES (count)")
            types = session.workflow_analysis.get("node_types", {})
            sorted_types = sorted(types.items(), key=lambda x: -x[1])[:15]
            type_lines = [f"{t}: {c}" for t, c in sorted_types]
            prompt_parts.append(", ".join(type_lines))
            prompt_parts.append("")

        # Request
        prompt_parts.append("## REQUEST")
        prompt_parts.append("Based on this information, suggest a specific optimization mod.")
        prompt_parts.append("Provide Python code following this structure:")
        prompt_parts.append("")
        prompt_parts.append("```python")
        prompt_parts.append('description = "Brief description of what this mod does"')
        prompt_parts.append("")
        prompt_parts.append("def apply(content):")
        prompt_parts.append('    """')
        prompt_parts.append("    content: dict (parsed JSON workflow)")
        prompt_parts.append("    returns: modified dict, or None if no changes")
        prompt_parts.append('    """')
        prompt_parts.append("    nodes = content.get('nodes', [])")
        prompt_parts.append("    # Your modification logic here")
        prompt_parts.append("    return content")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("Focus on: VRAM optimization, speed improvements, or fixing errors.")
        prompt_parts.append("=" * 70)

        return "\n".join(prompt_parts)

    @staticmethod
    def generate_initial_prompt(workflow_content: Dict, user_goal: str = "",
                               llm_type: str = "claude", include_full_workflow: bool = False) -> str:
        """Generate the initial analysis prompt."""

        preset = LLMPromptGenerator.LLM_PRESETS.get(llm_type, LLMPromptGenerator.LLM_PRESETS["claude"])
        analysis = WorkflowAnalyzer.analyze(workflow_content)

        prompt_parts = []
        prompt_parts.append(preset["prefix"])

        prompt_parts.append("=" * 70)
        prompt_parts.append("COMFYUI WORKFLOW OPTIMIZATION - INITIAL ANALYSIS")
        prompt_parts.append("=" * 70)
        prompt_parts.append("")

        prompt_parts.append("## CONTEXT")
        prompt_parts.append("I have a ComfyUI workflow I'd like to optimize.")
        if user_goal:
            prompt_parts.append(f"**My Goal:** {user_goal}")
        prompt_parts.append("")

        prompt_parts.append("## WORKFLOW ANALYSIS")
        prompt_parts.append(WorkflowAnalyzer.summarize(analysis))
        prompt_parts.append("")

        # All node types
        prompt_parts.append("## ALL NODE TYPES")
        types = analysis.get("node_types", {})
        sorted_types = sorted(types.items(), key=lambda x: -x[1])
        for t, c in sorted_types:
            prompt_parts.append(f"  • {t}: {c}")
        prompt_parts.append("")

        # Smart suggestions
        suggestions = WorkflowAnalyzer.get_smart_suggestions(analysis)
        if suggestions:
            prompt_parts.append("## AUTO-DETECTED ISSUES")
            for s in suggestions[:5]:
                prompt_parts.append(f"  [{s['priority'].upper()}] {s['title']}")
                prompt_parts.append(f"     → {s['reason']}")
            prompt_parts.append("")

        # Condensed workflow structure
        if include_full_workflow:
            prompt_parts.append("## WORKFLOW STRUCTURE (condensed)")
            nodes = workflow_content.get("nodes", [])
            for node in nodes[:50]:
                node_id = node.get("id", "?")
                node_type = node.get("type", "Unknown")
                widgets = node.get("widgets_values", [])
                widgets_preview = str(widgets)[:80] + "..." if len(str(widgets)) > 80 else str(widgets)
                prompt_parts.append(f"[{node_id}] {node_type}: {widgets_preview}")
            if len(nodes) > 50:
                prompt_parts.append(f"... and {len(nodes) - 50} more nodes")
            prompt_parts.append("")

        # Request
        prompt_parts.append("## REQUEST")
        prompt_parts.append("Please suggest an optimization mod. Respond with Python code:")
        prompt_parts.append("")
        prompt_parts.append("```python")
        prompt_parts.append('description = "Brief description"')
        prompt_parts.append("def apply(content):")
        prompt_parts.append("    # content is the parsed JSON workflow dict")
        prompt_parts.append("    # Return modified dict or None if no changes")
        prompt_parts.append("    return content")
        prompt_parts.append("```")
        prompt_parts.append("=" * 70)

        return "\n".join(prompt_parts)

# ═══════════════════════════════════════════════════════════════════════════════
# BUILT-IN MODS LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

class BuiltInMods:
    """Collection of built-in optimization mods."""

    @staticmethod
    def cap_resolution(content: Dict, max_res: int = 1024) -> Dict:
        """Cap all resolutions to a maximum value."""
        nodes = content.get("nodes", [])
        changed = False
        for node in nodes:
            widgets = node.get("widgets_values", [])
            for i, w in enumerate(widgets):
                if isinstance(w, int) and w > max_res and w % 8 == 0 and 256 <= w <= 4096:
                    widgets[i] = max_res
                    changed = True
        return content if changed else None

    @staticmethod
    def bypass_upscalers(content: Dict) -> Dict:
        """Bypass all upscaler nodes."""
        nodes = content.get("nodes", [])
        changed = False
        for node in nodes:
            node_type = node.get("type", "").lower()
            if "upscale" in node_type or "esrgan" in node_type:
                if node.get("mode") != 1:  # 1 = bypass
                    node["mode"] = 1
                    changed = True
        return content if changed else None

    @staticmethod
    def reduce_steps(content: Dict, max_steps: int = 20) -> Dict:
        """Reduce sampling steps."""
        nodes = content.get("nodes", [])
        changed = False
        for node in nodes:
            node_type = node.get("type", "").lower()
            if "sampler" in node_type or "ksampler" in node_type:
                widgets = node.get("widgets_values", [])
                for i, w in enumerate(widgets):
                    if isinstance(w, int) and 20 < w <= 150:
                        widgets[i] = max_steps
                        changed = True
        return content if changed else None

    @staticmethod
    def reduce_batch_size(content: Dict, batch_size: int = 1) -> Dict:
        """Reduce batch sizes to 1."""
        nodes = content.get("nodes", [])
        changed = False
        for node in nodes:
            widgets = node.get("widgets_values", [])
            # Batch size is often the last integer widget
            if widgets and isinstance(widgets[-1], int) and 1 < widgets[-1] <= 64:
                widgets[-1] = batch_size
                changed = True
        return content if changed else None

    @staticmethod
    def mute_node_type(content: Dict, type_pattern: str) -> Dict:
        """Mute all nodes matching a type pattern."""
        nodes = content.get("nodes", [])
        changed = False
        pattern_lower = type_pattern.lower()
        for node in nodes:
            node_type = node.get("type", "").lower()
            if pattern_lower in node_type:
                if node.get("mode") != 2:  # 2 = mute
                    node["mode"] = 2
                    changed = True
        return content if changed else None

    @staticmethod
    def speed_test_preset(content: Dict) -> Dict:
        """Apply all speed optimizations for testing."""
        # Chain multiple optimizations
        content = BuiltInMods.cap_resolution(content, 512) or content
        content = BuiltInMods.bypass_upscalers(content) or content
        content = BuiltInMods.reduce_steps(content, 15) or content
        content = BuiltInMods.reduce_batch_size(content, 1) or content
        return content

    @staticmethod
    def vram_8gb_preset(content: Dict) -> Dict:
        """Optimize for 8GB VRAM cards."""
        content = BuiltInMods.cap_resolution(content, 768) or content
        content = BuiltInMods.bypass_upscalers(content) or content
        content = BuiltInMods.reduce_batch_size(content, 1) or content
        return content

# ═══════════════════════════════════════════════════════════════════════════════
# MOD LOADER (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

class ModLoader:
    """Handles loading and managing mod files."""

    def __init__(self, mods_dir: str = MODS_DIR):
        self.mods_dir = mods_dir
        if not os.path.exists(mods_dir):
            os.makedirs(mods_dir)
        self._ensure_example_mods()

    def _ensure_example_mods(self):
        """Ensure example mods exist."""
        example_mods = {
            "vram_optimizer.py": '''description = "Reduce VRAM usage (cap resolution, reduce batch)"

def apply(content):
    nodes = content.get("nodes", [])
    changed = False

    for node in nodes:
        widgets = node.get("widgets_values", [])

        # Cap resolutions to 1024
        for i, w in enumerate(widgets):
            if isinstance(w, int) and w > 1024 and w % 8 == 0:
                widgets[i] = 1024
                changed = True

        # Reduce batch sizes
        if widgets and isinstance(widgets[-1], int) and widgets[-1] > 1:
            widgets[-1] = 1
            changed = True

    return content if changed else None
''',
            "bypass_upscalers.py": '''description = "Bypass all upscaler nodes for faster testing"

def apply(content):
    nodes = content.get("nodes", [])
    changed = False

    for node in nodes:
        node_type = node.get("type", "").lower()
        if "upscale" in node_type or "esrgan" in node_type:
            if node.get("mode") != 1:
                node["mode"] = 1  # 1 = bypass
                changed = True

    return content if changed else None
''',
            "reduce_steps.py": '''description = "Reduce sampling steps to 20 for faster iteration"

def apply(content):
    nodes = content.get("nodes", [])
    changed = False

    for node in nodes:
        node_type = node.get("type", "").lower()
        if "sampler" in node_type:
            widgets = node.get("widgets_values", [])
            for i, w in enumerate(widgets):
                if isinstance(w, int) and 20 < w <= 150:
                    widgets[i] = 20
                    changed = True

    return content if changed else None
''',
        }

        for filename, content in example_mods.items():
            filepath = os.path.join(self.mods_dir, filename)
            if not os.path.exists(filepath):
                try:
                    with open(filepath, 'w') as f:
                        f.write(content)
                except Exception:
                    pass

    def list_mods(self) -> List[str]:
        """List available mod files."""
        return [f for f in os.listdir(self.mods_dir)
                if f.endswith('.py') and f != '__init__.py']

    def load_mod(self, mod_name: str):
        """Load a mod module."""
        mod_path = os.path.join(self.mods_dir, mod_name)
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def get_mod_description(self, mod_name: str) -> str:
        """Get the description from a mod."""
        try:
            mod = self.load_mod(mod_name)
            return getattr(mod, 'description', 'No description available')
        except Exception as e:
            return f"Error loading: {e}"

    def save_mod(self, name: str, content: str) -> bool:
        """Save a new mod."""
        if not name.endswith('.py'):
            name += '.py'
        path = os.path.join(self.mods_dir, name)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False

    def delete_mod(self, mod_name: str) -> bool:
        """Delete a mod."""
        path = os.path.join(self.mods_dir, mod_name)
        try:
            if os.path.exists(path):
                os.remove(path)
                return True
        except Exception:
            pass
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# FILE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def read_workflow(filepath: str) -> Optional[Dict]:
    """Read a JSON workflow file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  {styled('✗', Style.RED)} Error reading file: {e}")
        return None

def write_workflow(filepath: str, content: Dict) -> bool:
    """Write a JSON workflow file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
        return True
    except Exception as e:
        print(f"  {styled('✗', Style.RED)} Error writing file: {e}")
        return False

def create_experimental_path(filepath: str) -> str:
    """Create the experimental version path."""
    base, ext = os.path.splitext(filepath)
    return f"{base}_experimental{ext}"

def get_workflow_diff(original: Dict, modified: Dict) -> Dict:
    """Get differences between two workflows."""
    diff = {
        "nodes_changed": [],
        "nodes_added": [],
        "nodes_removed": [],
        "widget_changes": [],
    }

    orig_nodes = {n.get("id"): n for n in original.get("nodes", [])}
    mod_nodes = {n.get("id"): n for n in modified.get("nodes", [])}

    # Find changes
    for node_id, orig_node in orig_nodes.items():
        if node_id not in mod_nodes:
            diff["nodes_removed"].append({
                "id": node_id,
                "type": orig_node.get("type"),
            })
        else:
            mod_node = mod_nodes[node_id]
            if orig_node.get("mode") != mod_node.get("mode"):
                diff["nodes_changed"].append({
                    "id": node_id,
                    "type": orig_node.get("type"),
                    "change": f"mode: {orig_node.get('mode')} → {mod_node.get('mode')}",
                })

            # Check widget changes
            orig_widgets = orig_node.get("widgets_values", [])
            mod_widgets = mod_node.get("widgets_values", [])
            for i, (ow, mw) in enumerate(zip(orig_widgets, mod_widgets)):
                if ow != mw:
                    diff["widget_changes"].append({
                        "node_id": node_id,
                        "node_type": orig_node.get("type"),
                        "widget_idx": i,
                        "old": ow,
                        "new": mw,
                    })

    for node_id, mod_node in mod_nodes.items():
        if node_id not in orig_nodes:
            diff["nodes_added"].append({
                "id": node_id,
                "type": mod_node.get("type"),
            })

    return diff

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION (v0.1 Ultimate Edition)
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceLab:
    """Main application controller."""

    def __init__(self):
        self.config = ConfigManager()
        self.monitor = ComfyUIMonitor(self.config.get("comfy_url", COMFY_URL))
        self.mod_loader = ModLoader()
        self.session = SessionHistory()
        self.target_workflow: Optional[str] = None
        self.workflow_content: Optional[Dict] = None
        self.original_content: Optional[Dict] = None  # For revert
        self.user_goal: str = self.config.get("last_goal", "")

    def run(self):
        """Main application loop."""
        clear_screen()
        print_header(
            "COMFYUI PERFORMANCE LAB",
            "Iterative Workflow Optimization with LLM-Assisted Analysis",
            VERSION
        )

        # Initial setup
        self.setup_target_workflow()
        if not self.target_workflow:
            print(f"\n  {styled('Goodbye!', Style.DIM)}")
            return

        # Main menu loop
        while True:
            self.show_main_menu()
            choice = input(f"\n  {styled('▶', Style.CYAN)} Enter choice: ").strip().lower()

            if choice == 'q':
                self._save_state()
                print(f"\n  {styled('Session saved. Goodbye!', Style.DIM)}")
                break
            elif choice == '1':
                self.apply_mod_workflow()
            elif choice == '2':
                self.quick_actions_menu()
            elif choice == '3':
                self.generate_llm_prompt()
            elif choice == '4':
                self.paste_new_mod()
            elif choice == '5':
                self.smart_suggestions()
            elif choice == '6':
                self.benchmark_mode()
            elif choice == '7':
                self.view_dashboard()
            elif choice == '8':
                self.presets_menu()
            elif choice == '9':
                self.set_optimization_goal()
            elif choice == 'c':
                self.test_comfyui_connection()
            elif choice == 't':
                self.setup_target_workflow()
            elif choice == 'e':
                self.export_session()
            elif choice == 'l':
                self.llm_enhancer_menu()
            elif choice == 'm':
                self.model_tuner_menu()
            elif choice == 'b':
                self.beautify_menu()
            elif choice == 's':
                self.save_as_menu()
            elif choice == 'u':
                self.update_performance_lab()
            else:
                print(f"  {styled('Invalid choice', Style.YELLOW)}")

    def _save_state(self):
        """Save current state to config."""
        self.config.set("last_workflow", self.target_workflow)
        self.config.set("last_goal", self.user_goal)

    def setup_target_workflow(self):
        """Set up the target workflow file."""
        last_workflow = self.config.get("last_workflow")

        print_box("Target Workflow", [
            "Enter the workflow JSON file you want to optimize.",
            "This will be your base for all modifications.",
        ], Style.BLUE, icon="📁")

        if last_workflow and os.path.exists(last_workflow):
            print(f"\n  {styled('Last workflow:', Style.DIM)} {last_workflow}")
            use_last = input(f"  {styled('▶', Style.CYAN)} Use last workflow? (y/n): ").strip().lower()
            if use_last == 'y':
                filepath = last_workflow
            else:
                filepath = input(f"  {styled('▶', Style.CYAN)} Workflow file path (or 'q' to quit): ").strip()
        else:
            filepath = input(f"\n  {styled('▶', Style.CYAN)} Workflow file path (or 'q' to quit): ").strip()

        if filepath.lower() == 'q':
            self.target_workflow = None
            return

        if not os.path.exists(filepath):
            print(f"  {styled('✗', Style.RED)} File not found: {filepath}")
            return self.setup_target_workflow()

        content = read_workflow(filepath)
        if content is None:
            return self.setup_target_workflow()

        self.target_workflow = filepath
        self.workflow_content = content
        self.original_content = copy.deepcopy(content)

        # Analyze workflow
        analysis = WorkflowAnalyzer.analyze(content)
        baseline_metrics = self.monitor.collect_metrics()
        self.session.set_baseline(filepath, analysis, baseline_metrics)

        print(f"\n  {styled('✓', Style.GREEN)} Workflow loaded: {styled(filepath, Style.BOLD)}")
        print_box("Workflow Analysis", WorkflowAnalyzer.summarize(analysis).split("\n"), Style.GREEN, icon="📊")

        # Show smart suggestions preview
        suggestions = WorkflowAnalyzer.get_smart_suggestions(analysis)
        if suggestions:
            print(f"\n  {styled('💡 Quick suggestions available! Use option [2] Quick Actions', Style.YELLOW)}")

    def show_main_menu(self):
        """Display the main menu."""
        print()
        print_divider("═", Style.CYAN)
        print(f"  {styled('⚡ MAIN MENU', Style.BOLD, Style.WHITE)}")
        print(f"  Target: {styled(self.target_workflow or 'Not set', Style.BLUE)}")
        if self.user_goal:
            print(f"  Goal: {styled(self.user_goal[:50], Style.MAGENTA)}")

        # Connection status
        connected = self.monitor.check_connection()
        status_icon = styled("●", Style.GREEN) if connected else styled("●", Style.RED)
        print(f"  ComfyUI: {status_icon} {'Connected' if connected else 'Not connected'}")

        print_divider()

        mods = self.mod_loader.list_mods()
        suggestions = []
        if self.session.workflow_analysis:
            suggestions = WorkflowAnalyzer.get_smart_suggestions(self.session.workflow_analysis)

        menu_items = [
            ("1", "Apply a Mod", f"{len(mods)} mods available", Style.WHITE),
            ("2", "⚡ Quick Actions", f"{len(suggestions)} suggestions", Style.YELLOW),
            ("3", "Generate LLM Prompt", "For Claude/GPT/Gemini", Style.WHITE),
            ("4", "Paste New Mod", "From LLM response", Style.WHITE),
            ("5", "🧠 Smart Suggestions", "AI-free analysis", Style.CYAN),
            ("6", "📊 Benchmark Mode", f"Run {self.config.get('benchmark_runs', 3)}x for metrics", Style.MAGENTA),
            ("7", "📈 View Dashboard", f"{len(self.session.entries)} modifications", Style.WHITE),
            ("8", "⚙️  Presets", "8GB VRAM, Speed Test...", Style.WHITE),
            ("9", "Set Goal", self.user_goal[:20] + "..." if len(self.user_goal) > 20 else (self.user_goal or "Not set"), Style.WHITE),
            ("L", "🤖 LLM Enhancer", "Advanced AI context", Style.GREEN),
            ("M", "🎛️  Model Tuner", "Auto-detect & optimize", Style.GREEN),
            ("B", "🎨 Beautify", "Organize & clean up", Style.MAGENTA),
            ("U", "🔄 Update", "Pull latest from GitHub", Style.CYAN),
            ("C", "Test Connection", "", Style.WHITE),
            ("T", "Change Target", "", Style.WHITE),
            ("S", "Save As...", "Save to new file", Style.WHITE),
            ("E", "Export Session", "", Style.WHITE),
            ("Q", "Quit", "", Style.RED),
        ]

        for key, label, hint, color in menu_items:
            hint_str = styled(f" ({hint})", Style.DIM) if hint else ""
            key_styled = styled(key, Style.CYAN, Style.BOLD)
            print(f"    {key_styled}  {styled(label, color)}{hint_str}")

    def quick_actions_menu(self):
        """Quick one-key actions menu."""
        if not self.session.workflow_analysis:
            print(f"\n  {styled('⚠', Style.YELLOW)} Load a workflow first.")
            return

        suggestions = WorkflowAnalyzer.get_smart_suggestions(self.session.workflow_analysis)

        print_box("⚡ Quick Actions", [
            "One-key optimizations based on your workflow analysis.",
            "These create an experimental file for testing.",
        ], Style.YELLOW, icon="")

        # Built-in quick actions
        quick_actions = [
            ("1", "Cap resolution to 768px", "cap_resolution_768", "~60% faster"),
            ("2", "Cap resolution to 1024px", "cap_resolution_1024", "~40% faster"),
            ("3", "Bypass all upscalers", "bypass_upscalers", "2-4GB VRAM saved"),
            ("4", "Reduce steps to 20", "reduce_steps_20", "Faster iteration"),
            ("5", "Reduce batch to 1", "reduce_batch_1", "VRAM reduction"),
            ("6", "🚀 Speed Test Preset", "speed_test", "All optimizations"),
            ("7", "💾 8GB VRAM Preset", "vram_8gb", "Fit on 8GB cards"),
            ("R", "↩️  Revert to Original", "revert", "Undo all changes"),
        ]

        print()
        for key, label, action, impact in quick_actions:
            print(f"    {styled(key, Style.CYAN)} {label} {styled(f'[{impact}]', Style.DIM)}")

        print(f"\n    {styled('B', Style.CYAN)} Back to main menu")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Select action: ").strip().lower()

        if choice == 'b':
            return

        action_map = {
            '1': ('cap_resolution_768', lambda c: BuiltInMods.cap_resolution(c, 768)),
            '2': ('cap_resolution_1024', lambda c: BuiltInMods.cap_resolution(c, 1024)),
            '3': ('bypass_upscalers', BuiltInMods.bypass_upscalers),
            '4': ('reduce_steps_20', lambda c: BuiltInMods.reduce_steps(c, 20)),
            '5': ('reduce_batch_1', BuiltInMods.reduce_batch_size),
            '6': ('speed_test_preset', BuiltInMods.speed_test_preset),
            '7': ('vram_8gb_preset', BuiltInMods.vram_8gb_preset),
            'r': ('revert', None),
        }

        if choice not in action_map:
            print(f"  {styled('Invalid choice', Style.YELLOW)}")
            return

        action_name, action_func = action_map[choice]

        if action_name == 'revert':
            # Revert to original
            if self.original_content:
                self.workflow_content = copy.deepcopy(self.original_content)
                write_workflow(self.target_workflow, self.workflow_content)
                print(f"  {styled('✓', Style.GREEN)} Reverted to original workflow!")
            return

        # Apply quick action
        print(f"\n  {styled('⚙', Style.CYAN)} Applying: {action_name}")

        content_copy = copy.deepcopy(self.workflow_content)
        new_content = action_func(content_copy)

        if new_content is None:
            print(f"  {styled('⚠', Style.YELLOW)} No changes needed (already optimized).")
            return

        # Show diff
        diff = get_workflow_diff(self.workflow_content, new_content)
        if diff["widget_changes"] or diff["nodes_changed"]:
            print(f"\n  {styled('Changes:', Style.DIM)}")
            for change in diff["widget_changes"][:5]:
                print(f"    • Node {change['node_id']} ({change['node_type']}): {change['old']} → {change['new']}")
            for change in diff["nodes_changed"][:5]:
                print(f"    • Node {change['id']} ({change['type']}): {change['change']}")

        # Save experimental
        exp_path = create_experimental_path(self.target_workflow)
        if write_workflow(exp_path, new_content):
            print(f"  {styled('✓', Style.GREEN)} Created: {styled(exp_path, Style.BOLD)}")

            # Ask to test
            test = input(f"\n  {styled('▶', Style.CYAN)} Test in ComfyUI now? (y/n): ").strip().lower()

            if test == 'y':
                result = self.monitor.wait_for_generation()
                self.show_generation_result(result)

                if result.get("success") is not None:
                    keep = self.ask_keep_changes()

                    if keep:
                        os.replace(exp_path, self.target_workflow)
                        self.workflow_content = new_content
                        print(f"  {styled('✓', Style.GREEN)} Changes applied!")
                    else:
                        if os.path.exists(exp_path):
                            os.remove(exp_path)
                        print(f"  {styled('↩', Style.YELLOW)} Changes reverted.")

                    self.session.add_entry(f"quick:{action_name}", f"Quick action: {action_name}", result, keep, diff)
            else:
                print(f"  {styled('📄', Style.DIM)} Experimental file ready for manual testing.")

    def apply_mod_workflow(self):
        """Apply a mod to the workflow."""
        mods = self.mod_loader.list_mods()

        if not mods:
            print(f"\n  {styled('⚠', Style.YELLOW)} No mods found in '{MODS_DIR}/' directory.")
            print(f"  {styled('Tip:', Style.DIM)} Use option 4 to paste a mod from an LLM response.")
            return

        print_box("Available Mods", [], Style.MAGENTA, icon="🔧")
        for i, mod_name in enumerate(mods):
            desc = self.mod_loader.get_mod_description(mod_name)
            print(f"    {styled(str(i+1), Style.CYAN)}  {mod_name}")
            print(f"       {styled(desc[:60], Style.DIM)}")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Select mod # (or 'b' to go back): ").strip()

        if choice.lower() == 'b':
            return

        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(mods)):
                print(f"  {styled('Invalid selection', Style.YELLOW)}")
                return
        except ValueError:
            print(f"  {styled('Please enter a number', Style.YELLOW)}")
            return

        mod_name = mods[idx]

        print(f"\n  {styled('⚙', Style.CYAN)} Loading mod: {mod_name}")

        try:
            mod = self.mod_loader.load_mod(mod_name)
            mod_desc = getattr(mod, 'description', 'No description')

            content_copy = copy.deepcopy(self.workflow_content)
            new_content = mod.apply(content_copy)

            if new_content is None or new_content == self.workflow_content:
                print(f"  {styled('⚠', Style.YELLOW)} Mod made no changes.")
                return

            # Show diff
            diff = get_workflow_diff(self.workflow_content, new_content)

            # Save experimental
            exp_path = create_experimental_path(self.target_workflow)
            if not write_workflow(exp_path, new_content):
                return

            print(f"  {styled('✓', Style.GREEN)} Created: {styled(exp_path, Style.BOLD)}")

            # Wait for generation
            result = self.monitor.wait_for_generation()

            # Show results
            self.show_generation_result(result)

            # Ask to keep
            if result.get("success") is not None:
                keep = self.ask_keep_changes()

                if keep:
                    os.replace(exp_path, self.target_workflow)
                    self.workflow_content = new_content
                    print(f"  {styled('✓', Style.GREEN)} Changes applied to: {self.target_workflow}")
                else:
                    if os.path.exists(exp_path):
                        os.remove(exp_path)
                    print(f"  {styled('↩', Style.YELLOW)} Changes reverted.")

                self.session.add_entry(mod_name, mod_desc, result, keep, diff)
            else:
                if os.path.exists(exp_path):
                    os.remove(exp_path)

        except Exception as e:
            print(f"  {styled('✗', Style.RED)} Error applying mod: {e}")

    def show_generation_result(self, result: Dict):
        """Display generation results."""
        print()
        print_divider("═")

        if result.get("success"):
            print(f"  {styled('✨ GENERATION SUCCESSFUL', Style.GREEN, Style.BOLD)}")
            print()
            duration = result.get("duration_seconds", 0)
            peak_vram = result.get("peak_vram_gb", 0)
            avg_vram = result.get("avg_vram_gb", 0)
            base_vram = result.get("baseline_vram_gb", 0)

            print(f"    Duration:    {styled(f'{duration:.2f}s', Style.BOLD)}")
            print(f"    Peak VRAM:   {styled(f'{peak_vram:.2f} GB', Style.BOLD)}")
            print(f"    Avg VRAM:    {avg_vram:.2f} GB")
            print(f"    Base VRAM:   {base_vram:.2f} GB")

            # Compare to baseline
            if self.session.baseline_result:
                baseline_dur = self.session.baseline_result.get("duration_seconds", 0)
                if baseline_dur > 0:
                    change = ((duration - baseline_dur) / baseline_dur) * 100
                    if change < 0:
                        print(f"    {styled(f'⚡ {abs(change):.1f}% FASTER than baseline!', Style.GREEN)}")
                    else:
                        print(f"    {styled(f'⚠ {change:.1f}% slower than baseline', Style.YELLOW)}")

        elif result.get("success") is False:
            print(f"  {styled('💥 GENERATION FAILED', Style.RED, Style.BOLD)}")
            print()
            print(f"    Error: {styled(result.get('error_message', 'Unknown'), Style.RED)}")
            if result.get("node_errors"):
                for err in result["node_errors"]:
                    print(f"    • Node {err.get('node_id')} ({err.get('node_type')})")
        else:
            print(f"  {styled('⚠ TEST NOT COMPLETED', Style.YELLOW, Style.BOLD)}")

        print_divider("═")

    def ask_keep_changes(self) -> bool:
        """Ask user whether to keep changes."""
        while True:
            choice = input(f"\n  {styled('▶', Style.CYAN)} Keep these changes? (y/n): ").strip().lower()
            if choice in ('y', 'yes'):
                return True
            elif choice in ('n', 'no'):
                return False
            print(f"  {styled('Please enter y or n', Style.DIM)}")

    def smart_suggestions(self):
        """Show smart AI-free suggestions."""
        if not self.session.workflow_analysis:
            print(f"\n  {styled('⚠', Style.YELLOW)} Load a workflow first.")
            return

        suggestions = WorkflowAnalyzer.get_smart_suggestions(self.session.workflow_analysis)

        if not suggestions:
            print(f"\n  {styled('✓', Style.GREEN)} No obvious optimizations detected!")
            print(f"  {styled('Your workflow looks well-optimized.', Style.DIM)}")
            return

        print_box("🧠 Smart Suggestions (AI-Free)", [
            "Automatic analysis of your workflow structure.",
            "These suggestions don't require an external LLM.",
        ], Style.CYAN, icon="")

        for i, s in enumerate(suggestions):
            priority_colors = {
                "critical": Style.RED,
                "high": Style.YELLOW,
                "medium": Style.CYAN,
                "low": Style.WHITE,
                "info": Style.DIM,
            }
            color = priority_colors.get(s["priority"], Style.WHITE)

            print(f"\n  {styled(f'[{s["priority"].upper()}]', color, Style.BOLD)} {s['title']}")
            print(f"     {styled('Reason:', Style.DIM)} {s['reason']}")
            print(f"     {styled('Impact:', Style.DIM)} {s['impact']}")
            if s.get("quick_action"):
                print(f"     {styled('→ Available in Quick Actions menu', Style.GREEN)}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def benchmark_mode(self):
        """Run benchmark mode with multiple runs."""
        runs = self.config.get("benchmark_runs", 3)

        print_box("📊 Benchmark Mode", [
            f"Run the workflow {runs} times for reliable metrics.",
            "This helps identify consistent performance characteristics.",
            "",
            "Make sure to:",
            "• Have the workflow open in ComfyUI",
            "• Be ready to click 'Queue Prompt' for each run",
        ], Style.MAGENTA, icon="")

        # Option to change run count
        print(f"\n  Current runs: {runs}")
        change = input(f"  {styled('▶', Style.CYAN)} Change run count? (Enter number or press Enter to keep): ").strip()
        if change.isdigit():
            runs = int(change)
            self.config.set("benchmark_runs", runs)

        print(f"\n  {styled('Starting benchmark...', Style.BOLD)}")
        print(f"  {styled(f'Will run {runs} iterations', Style.DIM)}")

        results = []

        for i in range(runs):
            print(f"\n  {styled(f'═══ Run {i+1}/{runs} ═══', Style.CYAN, Style.BOLD)}")

            result = self.monitor.wait_for_generation(show_progress=True)

            if result.get("success"):
                results.append(result)
                print(f"  {styled('✓', Style.GREEN)} Run {i+1}: {result['duration_seconds']:.2f}s | Peak VRAM: {result.get('peak_vram_gb', 0):.2f}GB")
            elif result.get("success") is False:
                print(f"  {styled('✗', Style.RED)} Run {i+1}: FAILED - {result.get('error_message', 'Unknown')}")
            else:
                print(f"  {styled('⚠', Style.YELLOW)} Benchmark aborted")
                break

            if i < runs - 1:
                input(f"\n  {styled('Press Enter for next run...', Style.DIM)}")

        # Show benchmark summary
        if results:
            print()
            print_divider("═", Style.MAGENTA)
            print(f"  {styled('📊 BENCHMARK RESULTS', Style.BOLD, Style.MAGENTA)}")
            print_divider("═", Style.MAGENTA)

            durations = [r["duration_seconds"] for r in results]
            vram_peaks = [r.get("peak_vram_gb", 0) for r in results]

            avg_dur = sum(durations) / len(durations)
            min_dur = min(durations)
            max_dur = max(durations)
            avg_vram = sum(vram_peaks) / len(vram_peaks)

            print(f"\n  Runs: {len(results)}/{runs} successful")
            print(f"\n  {styled('Duration:', Style.BOLD)}")
            print(f"    Average: {styled(f'{avg_dur:.2f}s', Style.CYAN, Style.BOLD)}")
            print(f"    Min:     {min_dur:.2f}s")
            print(f"    Max:     {max_dur:.2f}s")
            print(f"    Range:   ±{(max_dur - min_dur) / 2:.2f}s")

            print(f"\n  {styled('Peak VRAM:', Style.BOLD)}")
            print(f"    Average: {styled(f'{avg_vram:.2f} GB', Style.CYAN, Style.BOLD)}")

            # Mini chart
            print(f"\n  {styled('Duration trend:', Style.DIM)}")
            print_mini_chart(durations, label="  ")

            # Save benchmark
            self.session.add_benchmark("benchmark", results)

            # Set as baseline?
            if not self.session.baseline_result:
                set_base = input(f"\n  {styled('▶', Style.CYAN)} Set this as baseline? (y/n): ").strip().lower()
                if set_base == 'y':
                    # Use average result as baseline
                    baseline_result = {
                        "success": True,
                        "duration_seconds": avg_dur,
                        "peak_vram_gb": avg_vram,
                        "baseline_vram_gb": results[0].get("baseline_vram_gb", 0),
                    }
                    self.session.set_baseline_result(baseline_result)
                    print(f"  {styled('✓', Style.GREEN)} Baseline set!")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def view_dashboard(self):
        """View the session dashboard."""
        clear_screen()
        print_header("📈 SESSION DASHBOARD", f"Session: {self.session.session_id}")

        # Summary stats
        total_mods = len(self.session.entries)
        kept_mods = sum(1 for e in self.session.entries if e["kept"])

        print(f"\n  {styled('Session Summary:', Style.BOLD)}")
        print(f"    Total modifications: {total_mods}")
        print(f"    Kept: {styled(str(kept_mods), Style.GREEN)} | Reverted: {styled(str(total_mods - kept_mods), Style.RED)}")

        # Performance trend
        if self.session.entries:
            print(f"\n  {styled('Modification History:', Style.BOLD)}")
            print(self.session.get_summary())

            # Chart
            trend = self.session.get_performance_trend()
            if trend:
                print(f"\n  {styled('Performance Trend:', Style.BOLD)}")
                durations = [t[1] for t in trend]
                print_mini_chart(durations, label="  Duration")

        # Benchmark results
        if self.session.benchmark_results:
            print(f"\n  {styled('Benchmark History:', Style.BOLD)}")
            for bench in self.session.benchmark_results[-3:]:
                print(f"    • {bench['name']}: {bench['avg']:.2f}s avg ({bench['runs']} runs)")

        # Baseline comparison
        if self.session.baseline_result:
            print(f"\n  {styled('Baseline:', Style.BOLD)}")
            print(f"    Duration: {self.session.baseline_result.get('duration_seconds', 'N/A')}s")
            print(f"    Peak VRAM: {self.session.baseline_result.get('peak_vram_gb', 'N/A')} GB")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def presets_menu(self):
        """Presets menu for common optimization profiles."""
        print_box("⚙️ Optimization Presets", [
            "Quick-apply optimization profiles for common scenarios.",
        ], Style.WHITE, icon="")

        presets = [
            ("1", "🚀 Speed Test", "Max speed for iteration (512px, 15 steps, no upscale)"),
            ("2", "💾 8GB VRAM", "Fit on 8GB GPUs (768px, batch 1, no upscale)"),
            ("3", "⚖️  Balanced", "Good quality/speed balance (1024px, 25 steps)"),
            ("4", "🎨 Quality", "Production quality (original settings)"),
        ]

        print()
        for key, name, desc in presets:
            print(f"    {styled(key, Style.CYAN)} {name}")
            print(f"       {styled(desc, Style.DIM)}")

        print(f"\n    {styled('B', Style.CYAN)} Back to main menu")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Select preset: ").strip().lower()

        if choice == 'b':
            return

        preset_funcs = {
            '1': ('speed_test', BuiltInMods.speed_test_preset),
            '2': ('8gb_vram', BuiltInMods.vram_8gb_preset),
            '3': ('balanced', lambda c: BuiltInMods.cap_resolution(c, 1024) or c),
            '4': ('quality', lambda c: c),  # No changes
        }

        if choice not in preset_funcs:
            print(f"  {styled('Invalid choice', Style.YELLOW)}")
            return

        preset_name, preset_func = preset_funcs[choice]

        if preset_name == 'quality':
            # Revert to original
            if self.original_content:
                self.workflow_content = copy.deepcopy(self.original_content)
                write_workflow(self.target_workflow, self.workflow_content)
                print(f"  {styled('✓', Style.GREEN)} Restored original quality settings!")
            return

        print(f"\n  {styled('⚙', Style.CYAN)} Applying preset: {preset_name}")

        content_copy = copy.deepcopy(self.workflow_content)
        new_content = preset_func(content_copy)

        if new_content:
            exp_path = create_experimental_path(self.target_workflow)
            if write_workflow(exp_path, new_content):
                print(f"  {styled('✓', Style.GREEN)} Created: {styled(exp_path, Style.BOLD)}")
                print(f"  {styled('📄', Style.DIM)} Load in ComfyUI and test!")

    def model_tuner_menu(self):
        """Model Tuner - Auto-detect model and apply optimizations."""
        if not MODEL_TUNER_AVAILABLE:
            print(f"\n  {styled('⚠', Style.YELLOW)} Model Tuner not available.")
            print(f"  {styled('Make sure model_tuner.py is in the same directory.', Style.DIM)}")
            return

        if not self.workflow_content:
            print(f"\n  {styled('⚠', Style.YELLOW)} Load a workflow first.")
            return

        # Detect model
        model_info = detect_model(self.workflow_content)
        model_type_str = model_info["type"]
        model_name = model_info["name"]
        confidence = model_info["confidence"]
        profile = model_info["profile"]

        print_box("🎛️ Model Tuner", [
            "Smart optimization based on detected model type.",
            "Automatically adjusts settings for best results.",
        ], Style.GREEN, icon="")

        # Show detection results
        print(f"\n  {styled('📊 Model Detection', Style.BOLD)}")
        print(f"  {'─' * 50}")

        if model_type_str == "unknown":
            print(f"  {styled('⚠ Could not detect model type', Style.YELLOW)}")
            print(f"  {styled('Using default SD1.5 settings', Style.DIM)}")
        else:
            conf_pct = int(confidence * 100)
            conf_color = Style.GREEN if conf_pct > 70 else Style.YELLOW
            print(f"  Detected: {styled(model_name, Style.BOLD, Style.CYAN)}")
            print(f"  Confidence: {styled(f'{conf_pct}%', conf_color)}")

        # Show optimal settings
        if profile:
            print(f"\n  {styled('⚙️ Optimal Settings', Style.BOLD)}")
            print(f"  Resolution:  {styled(f'{profile.base_resolution[0]}x{profile.base_resolution[1]}', Style.CYAN)}")
            print(f"  Steps:       {styled(str(profile.optimal_steps), Style.CYAN)} (range: {profile.min_steps}-{profile.max_steps})")
            print(f"  CFG:         {styled(str(profile.default_cfg), Style.CYAN)} (range: {profile.cfg_range[0]}-{profile.cfg_range[1]})")
            print(f"  Samplers:    {styled(', '.join(profile.best_samplers[:3]), Style.DIM)}")
            print(f"  Schedulers:  {styled(', '.join(profile.best_schedulers[:3]), Style.DIM)}")

            if profile.tips:
                print(f"\n  {styled('💡 Tips:', Style.YELLOW)}")
                for tip in profile.tips[:3]:
                    print(f"    • {tip}")

        # Show LoRA info
        try:
            lora_nodes = LoRATuner.find_lora_nodes(self.workflow_content)
            if lora_nodes:
                loras = LoRATuner.extract_lora_info(lora_nodes)
                if loras:
                    print(f"\n  {styled('🎨 LoRAs Detected:', Style.BOLD)} {len(loras)}")
                    for lora in loras[:3]:
                        name = os.path.basename(lora["name"])[:35]
                        strength = lora.get("strength_model", "N/A")
                        print(f"    • {name} (strength: {strength})")
        except Exception:
            pass

        # Menu options
        print(f"\n  {styled('Actions:', Style.BOLD)}")
        print(f"    {styled('1', Style.CYAN)} Apply optimal settings (balanced)")
        print(f"    {styled('2', Style.CYAN)} Create speed variant")
        print(f"    {styled('3', Style.CYAN)} Create quality variant")
        print(f"    {styled('4', Style.CYAN)} Show all sampler recommendations")
        print(f"    {styled('B', Style.CYAN)} Back to main menu")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Choice: ").strip().lower()

        if choice == 'b':
            return

        try:
            model_type = ModelType(model_type_str) if model_type_str != "unknown" else ModelType.SD15
        except (ValueError, KeyError):
            model_type = ModelType.SD15

        if choice == '1':
            optimized = optimize_workflow(self.workflow_content, "balanced")
            exp_path = create_experimental_path(self.target_workflow).replace("_experimental", "_model_optimized")
            if write_workflow(exp_path, optimized):
                print(f"  {styled('✓', Style.GREEN)} Created: {styled(exp_path, Style.BOLD)}")
                print(f"  {styled('📄', Style.DIM)} Load in ComfyUI and test!")

        elif choice == '2':
            optimized = optimize_workflow(self.workflow_content, "speed")
            exp_path = create_experimental_path(self.target_workflow).replace("_experimental", "_speed")
            if write_workflow(exp_path, optimized):
                print(f"  {styled('✓', Style.GREEN)} Created: {styled(exp_path, Style.BOLD)}")

        elif choice == '3':
            optimized = optimize_workflow(self.workflow_content, "quality")
            exp_path = create_experimental_path(self.target_workflow).replace("_experimental", "_quality")
            if write_workflow(exp_path, optimized):
                print(f"  {styled('✓', Style.GREEN)} Created: {styled(exp_path, Style.BOLD)}")

        elif choice == '4':
            # Show all recommendations
            presets = get_model_presets(model_type_str)
            print(f"\n  {styled('📋 All Recommendations for ' + model_name, Style.BOLD)}")
            print(f"  {'─' * 50}")

            for use_case, rec in presets.items():
                print(f"\n  {styled(use_case.upper(), Style.CYAN)} - {rec['description']}")
                print(f"    Sampler:   {rec['recommended_sampler']}")
                print(f"    Scheduler: {rec['recommended_scheduler']}")
                print(f"    Steps:     {rec['recommended_steps']}")
                print(f"    CFG:       {rec['recommended_cfg']}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def beautify_menu(self):
        """Workflow beautification menu."""
        if not WORKFLOW_UTILS_AVAILABLE:
            print(f"\n  {styled('⚠', Style.YELLOW)} Workflow utilities not available.")
            print(f"  {styled('Make sure workflow_utils.py is in the same directory.', Style.DIM)}")
            return

        if not self.workflow_content:
            print(f"\n  {styled('⚠', Style.YELLOW)} Load a workflow first.")
            return

        print_box("🎨 Workflow Beautifier", [
            "Organize and clean up your workflow layout.",
            "Choose a beautification style:",
        ], Style.MAGENTA, icon="")

        modes = [
            ("1", "📐 Organize by Category", "Group nodes logically", BeautifyMode.ORGANIZE),
            ("2", "📏 Align to Grid", "Snap all nodes to grid", BeautifyMode.ALIGN_GRID),
            ("3", "➡️  Flow Left-to-Right", "Arrange in L→R flow", BeautifyMode.FLOW_LEFT_RIGHT),
            ("4", "⬇️  Flow Top-to-Down", "Arrange in T→D flow", BeautifyMode.FLOW_TOP_DOWN),
            ("5", "📦 Compact Layout", "Minimize space usage", BeautifyMode.COMPACT),
            ("6", "📭 Expand Layout", "Add breathing room", BeautifyMode.EXPAND),
            ("7", "🎨 Color Code Nodes", "Color by function", BeautifyMode.COLOR_CODE),
            ("8", "📁 Create Groups", "Add group boxes", None),  # Special case
        ]

        print()
        for key, name, desc, _ in modes:
            print(f"    {styled(key, Style.CYAN)} {name}")
            print(f"       {styled(desc, Style.DIM)}")

        print(f"\n    {styled('B', Style.CYAN)} Back to main menu")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Select style: ").strip().lower()

        if choice == 'b':
            return

        mode_map = {str(i+1): modes[i][3] for i in range(len(modes))}

        if choice not in mode_map:
            print(f"  {styled('Invalid choice', Style.YELLOW)}")
            return

        selected_mode = mode_map[choice]

        try:
            if choice == '8':
                # Special: create groups
                beautified = WorkflowBeautifier.create_groups_by_category(self.workflow_content)
            else:
                beautified = WorkflowBeautifier.beautify(self.workflow_content, selected_mode)

            # Show preview
            old_fp = WorkflowFingerprinter.fingerprint(self.workflow_content)
            new_fp = WorkflowFingerprinter.fingerprint(beautified)

            print(f"\n  {styled('✓', Style.GREEN)} Beautification complete!")
            print(f"    Nodes: {new_fp.node_count}")

            # Ask to apply
            apply = input(f"\n  {styled('▶', Style.CYAN)} Apply changes to workflow? (y/n): ").strip().lower()

            if apply == 'y':
                # Check compatibility before overwrite
                is_safe, warnings = is_safe_to_overwrite(self.original_content, beautified)

                if not is_safe:
                    print(f"\n  {styled('⚠ INCOMPATIBLE CHANGES DETECTED:', Style.RED, Style.BOLD)}")
                    for w in warnings:
                        print(f"    {styled('!', Style.RED)} {w}")
                    print(f"\n  {styled('Cannot overwrite - model family changed.', Style.YELLOW)}")
                    alt_path = suggest_filename(self.target_workflow, beautified)
                    save_alt = input(f"  {styled('▶', Style.CYAN)} Save to {alt_path} instead? (y/n): ").strip().lower()
                    if save_alt == 'y':
                        write_workflow(alt_path, beautified)
                        print(f"  {styled('✓', Style.GREEN)} Saved to: {alt_path}")
                else:
                    if warnings:
                        print(f"\n  {styled('⚠ Warnings:', Style.YELLOW)}")
                        for w in warnings:
                            print(f"    {w}")

                    # Overwrite original
                    self.workflow_content = beautified
                    write_workflow(self.target_workflow, beautified)
                    print(f"  {styled('✓', Style.GREEN)} Applied and saved to: {self.target_workflow}")

                    # Re-analyze
                    self.session.workflow_analysis = WorkflowAnalyzer.analyze(beautified)

        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def save_as_menu(self):
        """Save current workflow to a new file."""
        if not self.workflow_content:
            print(f"\n  {styled('⚠', Style.YELLOW)} No workflow loaded.")
            return

        print_box("💾 Save As", [
            "Save the current workflow to a new file.",
            "The original file will remain unchanged.",
        ], Style.BLUE, icon="")

        # Suggest filename
        if self.target_workflow:
            base = self.target_workflow.rsplit(".", 1)[0]
            suggested = f"{base}_v2.json"
        else:
            suggested = "workflow_optimized.json"

        print(f"\n  {styled('Suggested:', Style.DIM)} {suggested}")
        path = input(f"  {styled('▶', Style.CYAN)} Save path (Enter for suggested): ").strip()

        if not path:
            path = suggested

        if not path.endswith(".json"):
            path += ".json"

        # Check if file exists
        if os.path.exists(path):
            overwrite = input(f"  {styled('⚠', Style.YELLOW)} File exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print(f"  {styled('Cancelled', Style.DIM)}")
                return

        if write_workflow(path, self.workflow_content):
            print(f"  {styled('✓', Style.GREEN)} Saved to: {styled(path, Style.BOLD)}")

            # Ask if user wants to switch to this file
            switch = input(f"  {styled('▶', Style.CYAN)} Switch to this file as target? (y/n): ").strip().lower()
            if switch == 'y':
                self.target_workflow = path
                self.original_content = copy.deepcopy(self.workflow_content)
                print(f"  {styled('✓', Style.GREEN)} Now targeting: {path}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def llm_enhancer_menu(self):
        """LLM Enhancer - Advanced AI context generation."""
        if not LLM_ENHANCER_AVAILABLE:
            print(f"\n  {styled('⚠', Style.YELLOW)} LLM Enhancer not available.")
            print(f"  {styled('Make sure llm_enhancer.py is in the same directory.', Style.DIM)}")
            return

        # Initialize LLM context builder
        try:
            llm_builder = LLMContextBuilder()
        except Exception as e:
            print(f"\n  {styled('⚠', Style.YELLOW)} Error initializing LLM Enhancer: {e}")
            return

        print_box("🤖 LLM Enhancer", [
            "Advanced AI context generation for better LLM assistance.",
            "Provides comprehensive context including node catalog,",
            "system specs, knowledge base, and conversation memory.",
        ], Style.GREEN, icon="")

        # Show available features
        print(f"\n  {styled('Features:', Style.BOLD)}")
        print(f"    {styled('1', Style.CYAN)} 📋 Generate Full Context Prompt")
        print(f"       {styled('Comprehensive prompt with all available context', Style.DIM)}")
        print(f"    {styled('2', Style.CYAN)} 🔧 Debug Workflow")
        print(f"       {styled('Generate debug-focused prompt for fixing errors', Style.DIM)}")
        print(f"    {styled('3', Style.CYAN)} ⚡ Optimize Speed")
        print(f"       {styled('Generate speed optimization prompt', Style.DIM)}")
        print(f"    {styled('4', Style.CYAN)} 🎨 Improve Quality")
        print(f"       {styled('Generate quality improvement prompt', Style.DIM)}")
        print(f"    {styled('5', Style.CYAN)} 💾 Reduce VRAM")
        print(f"       {styled('Generate VRAM optimization prompt', Style.DIM)}")
        print(f"    {styled('6', Style.CYAN)} 📖 Explain Workflow")
        print(f"       {styled('Generate explanation request prompt', Style.DIM)}")
        print(f"    {styled('7', Style.CYAN)} ✅ Validate LLM Response")
        print(f"       {styled('Validate and parse LLM mod response', Style.DIM)}")
        print(f"    {styled('8', Style.CYAN)} 🔍 View Node Catalog")
        print(f"       {styled('Browse installed ComfyUI nodes', Style.DIM)}")
        print(f"    {styled('9', Style.CYAN)} 💻 View System Specs")
        print(f"       {styled('Show current hardware context', Style.DIM)}")
        print(f"    {styled('K', Style.CYAN)} 📚 Knowledge Base")
        print(f"       {styled('Browse and search common solutions', Style.DIM)}")
        print(f"    {styled('H', Style.CYAN)} 📜 Conversation History")
        print(f"       {styled('View past LLM interactions', Style.DIM)}")
        print(f"    {styled('G', Style.CYAN)} 🗺️  Workflow Graph")
        print(f"       {styled('ASCII visualization of workflow', Style.DIM)}")
        print(f"\n    {styled('B', Style.CYAN)} Back to main menu")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Choice: ").strip().lower()

        if choice == 'b':
            return

        if choice == '1':
            self._llm_generate_full_context(llm_builder)
        elif choice == '2':
            self._llm_generate_goal_prompt(llm_builder, PromptGoal.DEBUG)
        elif choice == '3':
            self._llm_generate_goal_prompt(llm_builder, PromptGoal.SPEED)
        elif choice == '4':
            self._llm_generate_goal_prompt(llm_builder, PromptGoal.QUALITY)
        elif choice == '5':
            self._llm_generate_goal_prompt(llm_builder, PromptGoal.VRAM)
        elif choice == '6':
            self._llm_generate_goal_prompt(llm_builder, PromptGoal.EXPLAIN)
        elif choice == '7':
            self._llm_validate_response(llm_builder)
        elif choice == '8':
            self._llm_view_node_catalog(llm_builder)
        elif choice == '9':
            self._llm_view_system_specs()
        elif choice == 'k':
            self._llm_knowledge_base(llm_builder)
        elif choice == 'h':
            self._llm_conversation_history(llm_builder)
        elif choice == 'g':
            self._llm_workflow_graph()
        else:
            print(f"  {styled('Invalid choice', Style.YELLOW)}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def _llm_generate_full_context(self, llm_builder):
        """Generate comprehensive LLM context."""
        if not self.workflow_content:
            print(f"\n  {styled('⚠', Style.YELLOW)} Load a workflow first.")
            return

        print(f"\n  {styled('⚙', Style.CYAN)} Generating comprehensive context...")

        # Get custom request
        print(f"\n  {styled('Enter your request/question for the LLM:', Style.DIM)}")
        custom_request = input(f"  {styled('▶', Style.CYAN)} ").strip()

        if not custom_request:
            custom_request = "Please analyze this workflow and suggest optimizations."

        # Build context
        try:
            context = llm_builder.build_full_context(
                workflow=self.workflow_content,
                goal=PromptGoal.CUSTOM,
                custom_request=custom_request,
                include_nodes=True,
                include_history=True,
                include_knowledge=True,
                include_errors=True,
                include_graph=True,
                metrics=self.session.metrics_baseline
            )

            # Copy to clipboard
            success = copy_to_clipboard(context)

            print(f"\n  {styled('✓', Style.GREEN)} Full context generated!")
            print(f"  {styled(f'Length: {len(context)} characters', Style.DIM)}")

            if success:
                print(f"  {styled('📋 Copied to clipboard!', Style.CYAN)}")
            else:
                print(f"\n  {styled('Context preview (first 500 chars):', Style.DIM)}")
                print(f"  {context[:500]}...")

            # Record interaction
            try:
                llm_builder.record_interaction(
                    workflow=self.workflow_content,
                    goal="custom",
                    request=custom_request,
                    response="[prompt generated]",
                    applied=False
                )
            except Exception:
                pass

        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

    def _llm_generate_goal_prompt(self, llm_builder, goal: 'PromptGoal'):
        """Generate goal-specific LLM prompt."""
        if not self.workflow_content:
            print(f"\n  {styled('⚠', Style.YELLOW)} Load a workflow first.")
            return

        goal_names = {
            PromptGoal.DEBUG: "Debug",
            PromptGoal.SPEED: "Speed Optimization",
            PromptGoal.QUALITY: "Quality Improvement",
            PromptGoal.VRAM: "VRAM Reduction",
            PromptGoal.EXPLAIN: "Explanation",
        }

        print(f"\n  {styled('⚙', Style.CYAN)} Generating {goal_names.get(goal, 'prompt')}...")

        # Additional inputs for specific goals
        error_log = ""
        error_desc = ""
        target_vram = 6.0

        if goal == PromptGoal.DEBUG:
            print(f"\n  {styled('Paste error message (Enter twice when done):', Style.DIM)}")
            lines = []
            while True:
                line = input()
                if line == "":
                    if lines and lines[-1] == "":
                        break
                    lines.append("")
                else:
                    lines.append(line)
            error_log = "\n".join(lines).strip()

            error_desc = input(f"  {styled('▶', Style.CYAN)} Brief description of the problem: ").strip()

        elif goal == PromptGoal.VRAM:
            vram_input = input(f"  {styled('▶', Style.CYAN)} Target VRAM (GB, default 6): ").strip()
            try:
                target_vram = float(vram_input) if vram_input else 6.0
            except ValueError:
                target_vram = 6.0

        # Generate prompt
        try:
            prompt_gen = PromptGenerator(llm_builder.node_catalog)
            prompt = prompt_gen.generate(
                goal=goal,
                workflow=self.workflow_content,
                metrics=self.session.metrics_baseline,
                error_log=error_log,
                error_description=error_desc,
                target_vram=target_vram
            )

            # Copy to clipboard
            success = copy_to_clipboard(prompt)

            print(f"\n  {styled('✓', Style.GREEN)} Prompt generated!")
            print(f"  {styled(f'Length: {len(prompt)} characters', Style.DIM)}")

            if success:
                print(f"  {styled('📋 Copied to clipboard!', Style.CYAN)}")
            else:
                print(f"\n  {styled('Prompt preview (first 500 chars):', Style.DIM)}")
                print(f"  {prompt[:500]}...")

        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

    def _llm_validate_response(self, llm_builder):
        """Validate LLM-generated workflow modification."""
        print(f"\n  {styled('Paste LLM response JSON (Enter twice when done):', Style.DIM)}")

        lines = []
        while True:
            line = input()
            if line == "":
                if lines and lines[-1] == "":
                    break
                lines.append("")
            else:
                lines.append(line)

        response_json = "\n".join(lines).strip()

        if not response_json:
            print(f"  {styled('No input provided', Style.YELLOW)}")
            return

        print(f"\n  {styled('⚙', Style.CYAN)} Validating response...")

        try:
            result = llm_builder.validate_response(response_json, self.workflow_content)

            print(f"\n{result.to_string()}")

            if result.is_valid and result.fixed_workflow:
                apply = input(f"\n  {styled('▶', Style.CYAN)} Apply validated workflow? (y/n): ").strip().lower()

                if apply == 'y':
                    # Check safety
                    if WORKFLOW_UTILS_AVAILABLE and self.original_content:
                        is_safe, warnings = is_safe_to_overwrite(self.original_content, result.fixed_workflow)

                        if not is_safe:
                            print(f"\n  {styled('⚠ INCOMPATIBLE CHANGES:', Style.RED)}")
                            for w in warnings:
                                print(f"    {w}")
                            alt_path = suggest_filename(self.target_workflow, result.fixed_workflow)
                            save_alt = input(f"  {styled('▶', Style.CYAN)} Save to {alt_path}? (y/n): ").strip().lower()
                            if save_alt == 'y':
                                write_workflow(alt_path, result.fixed_workflow)
                                print(f"  {styled('✓', Style.GREEN)} Saved to: {alt_path}")
                            return

                    # Apply changes
                    self.workflow_content = result.fixed_workflow
                    write_workflow(self.target_workflow, result.fixed_workflow)
                    print(f"  {styled('✓', Style.GREEN)} Applied and saved!")

                    # Re-analyze
                    self.session.workflow_analysis = WorkflowAnalyzer.analyze(result.fixed_workflow)

                    # Record success
                    try:
                        llm_builder.record_interaction(
                            workflow=self.original_content,
                            goal="validated",
                            request="[LLM mod applied]",
                            response=response_json[:500],
                            applied=True,
                            success=True
                        )
                    except Exception:
                        pass

        except Exception as e:
            print(f"  {styled('✗ Validation error:', Style.RED)} {e}")

    def _llm_view_node_catalog(self, llm_builder):
        """View installed ComfyUI nodes."""
        print(f"\n  {styled('⚙', Style.CYAN)} Fetching node catalog from ComfyUI...")

        try:
            if llm_builder.node_catalog.fetch_catalog():
                categories = llm_builder.node_catalog.get_all_categories()

                print(f"\n  {styled('📋 Node Catalog', Style.BOLD)}")
                print(f"  Total nodes: {styled(str(len(llm_builder.node_catalog._cache)), Style.CYAN)}")
                print(f"  Categories: {styled(str(len(categories)), Style.CYAN)}")

                print(f"\n  {styled('Categories:', Style.DIM)}")
                for i, cat in enumerate(categories[:20]):
                    count = len(llm_builder.node_catalog._categories[cat])
                    print(f"    • {cat} ({count})")

                if len(categories) > 20:
                    print(f"    ... and {len(categories) - 20} more")

                # Option to export
                export = input(f"\n  {styled('▶', Style.CYAN)} Export to clipboard? (y/n): ").strip().lower()
                if export == 'y':
                    catalog_text = llm_builder.node_catalog.export_for_llm(compact=True, max_nodes=200)
                    if copy_to_clipboard(catalog_text):
                        print(f"  {styled('📋 Copied!', Style.GREEN)}")
            else:
                print(f"  {styled('⚠', Style.YELLOW)} Could not fetch catalog. Is ComfyUI running?")

        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

    def _llm_view_system_specs(self):
        """View current system specifications."""
        print(f"\n  {styled('⚙', Style.CYAN)} Collecting system specs...")

        try:
            specs = SystemSpecsCollector.collect()
            print(f"\n{specs.to_llm_context()}")

            export = input(f"\n  {styled('▶', Style.CYAN)} Copy to clipboard? (y/n): ").strip().lower()
            if export == 'y':
                if copy_to_clipboard(specs.to_llm_context()):
                    print(f"  {styled('📋 Copied!', Style.GREEN)}")

        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

    def _llm_knowledge_base(self, llm_builder):
        """Browse and search the knowledge base."""
        print(f"\n  {styled('📚 Knowledge Base', Style.BOLD)}")

        print(f"\n    {styled('1', Style.CYAN)} Browse all entries")
        print(f"    {styled('2', Style.CYAN)} Search entries")
        print(f"    {styled('3', Style.CYAN)} Find relevant for current workflow")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Choice: ").strip()

        try:
            if choice == '1':
                entries = llm_builder.knowledge_base.entries
                print(f"\n  Found {len(entries)} entries:")
                for entry in entries:
                    print(f"\n    {styled(entry.title, Style.BOLD)} [{entry.category}]")
                    print(f"    {styled(entry.problem[:100], Style.DIM)}")

            elif choice == '2':
                query = input(f"  {styled('▶', Style.CYAN)} Search: ").strip()
                results = llm_builder.knowledge_base.search(query)
                print(f"\n  Found {len(results)} results:")
                for entry in results:
                    print(f"\n{entry.to_llm_context()}")

            elif choice == '3':
                if not self.workflow_content:
                    print(f"  {styled('⚠', Style.YELLOW)} Load a workflow first.")
                    return
                results = llm_builder.knowledge_base.get_relevant_for_workflow(self.workflow_content)
                print(f"\n  Found {len(results)} relevant entries:")
                for entry in results:
                    print(f"\n{entry.to_llm_context()}")

        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

    def _llm_conversation_history(self, llm_builder):
        """View conversation history."""
        print(f"\n  {styled('📜 Conversation History', Style.BOLD)}")

        try:
            stats = llm_builder.memory.get_stats()

            print(f"\n  {styled('Statistics:', Style.DIM)}")
            print(f"    Total conversations: {stats['total_conversations']}")
            print(f"    Applied: {stats['applied']}")
            print(f"    Successful: {stats['successful']}")
            print(f"    Success rate: {stats['success_rate']:.1%}")
            print(f"    Unique workflows: {stats['unique_workflows']}")

            if stats['goals']:
                print(f"\n  {styled('By goal:', Style.DIM)}")
                for goal, count in stats['goals'].items():
                    print(f"    • {goal}: {count}")

            # Show recent history
            history = llm_builder.memory.get_history(limit=5)
            if history:
                print(f"\n  {styled('Recent entries:', Style.DIM)}")
                for entry in history:
                    status = "✓" if entry.success else "✗" if entry.applied else "?"
                    print(f"    {status} [{entry.timestamp[:10]}] {entry.goal}: {entry.user_message[:40]}...")

        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

    def _llm_workflow_graph(self):
        """Show ASCII workflow graph."""
        if not self.workflow_content:
            print(f"\n  {styled('⚠', Style.YELLOW)} Load a workflow first.")
            return

        print(f"\n  {styled('🗺️ Workflow Graph', Style.BOLD)}")

        try:
            # Show summary
            summary = WorkflowGraphExporter.export_summary(self.workflow_content)
            print(f"\n{summary}")

            # Option for ASCII graph
            show_graph = input(f"\n  {styled('▶', Style.CYAN)} Show full ASCII graph? (y/n): ").strip().lower()
            if show_graph == 'y':
                graph = WorkflowGraphExporter.export_ascii(self.workflow_content)
                print(f"\n{graph}")

            # Option for Mermaid
            show_mermaid = input(f"\n  {styled('▶', Style.CYAN)} Export Mermaid diagram? (y/n): ").strip().lower()
            if show_mermaid == 'y':
                mermaid = WorkflowGraphExporter.export_mermaid(self.workflow_content)
                if copy_to_clipboard(mermaid):
                    print(f"  {styled('📋 Mermaid diagram copied!', Style.GREEN)}")
                else:
                    print(f"\n{mermaid}")

        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

    def update_performance_lab(self):
        """Update Performance Lab from GitHub."""
        print_box("🔄 Update Performance Lab", [
            "Check for and install updates from GitHub.",
            "Requires git to be installed.",
        ], Style.CYAN, icon="")

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Check if it's a git repo
        git_dir = os.path.join(script_dir, ".git")
        is_git_repo = os.path.exists(git_dir)

        print(f"\n  {styled('Location:', Style.DIM)} {script_dir}")
        print(f"  {styled('Git repo:', Style.DIM)} {'Yes' if is_git_repo else 'No'}")

        if not is_git_repo:
            print(f"\n  {styled('⚠', Style.YELLOW)} Not a git repository.")
            print(f"  {styled('Options:', Style.DIM)}")
            print(f"    1. Run: git clone https://github.com/laboratoiresonore/ComfyUI_PerformanceLab.git")
            print(f"    2. Or download and replace files manually")

            # Offer to initialize git
            init = input(f"\n  {styled('▶', Style.CYAN)} Initialize git repo and set remote? (y/n): ").strip().lower()
            if init == 'y':
                try:
                    # Initialize git
                    subprocess.run(["git", "init"], cwd=script_dir, check=True, capture_output=True)
                    # Add remote
                    subprocess.run([
                        "git", "remote", "add", "origin",
                        "https://github.com/laboratoiresonore/ComfyUI_PerformanceLab.git"
                    ], cwd=script_dir, check=True, capture_output=True)
                    # Fetch
                    subprocess.run(["git", "fetch", "origin"], cwd=script_dir, check=True, capture_output=True)
                    # Reset to origin/main
                    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=script_dir, check=True, capture_output=True)

                    print(f"  {styled('✓', Style.GREEN)} Git initialized and synced with GitHub!")
                    print(f"  {styled('Restart Performance Lab to use the new version.', Style.YELLOW)}")

                except subprocess.CalledProcessError as e:
                    print(f"  {styled('✗ Git error:', Style.RED)} {e}")
                except FileNotFoundError:
                    print(f"  {styled('✗ Git not found:', Style.RED)} Please install git first.")
            return

        # Check current version
        print(f"\n  {styled('Current version:', Style.DIM)} v{VERSION}")

        # Fetch updates
        print(f"\n  {styled('⚙', Style.CYAN)} Fetching updates...")

        try:
            # Fetch from origin
            result = subprocess.run(
                ["git", "fetch", "origin"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"  {styled('⚠', Style.YELLOW)} Fetch failed: {result.stderr}")
                return

            # Check if behind
            result = subprocess.run(
                ["git", "status", "-uno"],
                cwd=script_dir,
                capture_output=True,
                text=True
            )

            status = result.stdout

            if "Your branch is behind" in status:
                # Get commit count
                result = subprocess.run(
                    ["git", "rev-list", "--count", "HEAD..origin/main"],
                    cwd=script_dir,
                    capture_output=True,
                    text=True
                )
                commits_behind = result.stdout.strip()

                print(f"  {styled(f'Updates available! ({commits_behind} commits behind)', Style.GREEN)}")

                # Show what will change
                result = subprocess.run(
                    ["git", "log", "--oneline", "HEAD..origin/main"],
                    cwd=script_dir,
                    capture_output=True,
                    text=True
                )
                print(f"\n  {styled('New commits:', Style.DIM)}")
                for line in result.stdout.strip().split("\n")[:5]:
                    print(f"    • {line}")

                # Confirm update
                confirm = input(f"\n  {styled('▶', Style.CYAN)} Pull updates? (y/n): ").strip().lower()

                if confirm == 'y':
                    # Stash local changes
                    subprocess.run(["git", "stash"], cwd=script_dir, capture_output=True)

                    # Pull
                    result = subprocess.run(
                        ["git", "pull", "origin", "main"],
                        cwd=script_dir,
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        print(f"  {styled('✓', Style.GREEN)} Updated successfully!")
                        print(f"  {styled('Restart Performance Lab to use the new version.', Style.YELLOW)}")

                        # Offer to restart
                        restart = input(f"\n  {styled('▶', Style.CYAN)} Restart now? (y/n): ").strip().lower()
                        if restart == 'y':
                            print(f"\n  {styled('Restarting...', Style.CYAN)}")
                            os.execvp(sys.executable, [sys.executable, __file__])
                    else:
                        print(f"  {styled('✗ Pull failed:', Style.RED)} {result.stderr}")
                        # Restore stash
                        subprocess.run(["git", "stash", "pop"], cwd=script_dir, capture_output=True)

            elif "Your branch is up to date" in status:
                print(f"  {styled('✓', Style.GREEN)} Already up to date!")

            else:
                print(f"  {styled('Status:', Style.DIM)} {status.strip()}")

        except subprocess.TimeoutExpired:
            print(f"  {styled('✗ Timeout:', Style.RED)} Could not reach GitHub.")
        except FileNotFoundError:
            print(f"  {styled('✗ Git not found:', Style.RED)} Please install git.")
        except Exception as e:
            print(f"  {styled('✗ Error:', Style.RED)} {e}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def generate_llm_prompt(self):
        """Generate a prompt for external LLM analysis."""
        print_box("Generate LLM Prompt", [
            "Create a prompt for external LLM analysis.",
            "Supports: Claude, GPT-4, Gemini, Llama/Mistral",
        ], Style.BLUE, icon="🤖")

        # LLM selection
        print(f"\n  {styled('Select LLM:', Style.DIM)}")
        for i, (key, preset) in enumerate(LLMPromptGenerator.LLM_PRESETS.items()):
            print(f"    {styled(str(i+1), Style.CYAN)} {preset['name']}")

        llm_choice = input(f"\n  {styled('▶', Style.CYAN)} LLM (1-4): ").strip()
        llm_keys = list(LLMPromptGenerator.LLM_PRESETS.keys())

        try:
            llm_type = llm_keys[int(llm_choice) - 1]
        except (ValueError, IndexError):
            llm_type = "claude"

        # Prompt type
        print(f"\n  {styled('Prompt type:', Style.DIM)}")
        print(f"    {styled('1', Style.CYAN)} Initial Analysis (first time)")
        print(f"    {styled('2', Style.CYAN)} After-Test Analysis (with results)")

        prompt_choice = input(f"\n  {styled('▶', Style.CYAN)} Choice: ").strip()

        latest_result = {}
        if self.session.entries:
            latest_result = self.session.entries[-1].get("result", {})

        if prompt_choice == '1':
            include_full = input(f"  {styled('▶', Style.CYAN)} Include full workflow? (y/n): ").strip().lower() == 'y'
            prompt = LLMPromptGenerator.generate_initial_prompt(
                self.workflow_content,
                self.user_goal,
                llm_type,
                include_full
            )
        else:
            prompt = LLMPromptGenerator.generate_analysis_prompt(
                self.session,
                latest_result,
                self.workflow_content,
                self.user_goal,
                llm_type
            )

        # Display and copy
        print()
        print(styled("─" * 70, Style.GRAY))
        print(styled("COPY THE FOLLOWING PROMPT:", Style.BOLD, Style.GREEN))
        print(styled("─" * 70, Style.GRAY))
        print()
        print(prompt)
        print()
        print(styled("─" * 70, Style.GRAY))

        # Copy to clipboard
        if Clipboard.copy(prompt):
            print(f"\n  {styled('📋 Copied to clipboard!', Style.GREEN)}")
        else:
            print(f"\n  {styled('📋 Select and copy the text above manually.', Style.YELLOW)}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def paste_new_mod(self):
        """Paste a new mod from LLM response."""
        print_box("Paste New Mod", [
            "Paste Python mod code from the LLM response.",
            "The code should have 'description' and 'apply(content)' function.",
            "",
            "Options:",
            "  • Paste multi-line code, then type 'END' on its own line",
            "  • Or paste from clipboard (if supported)",
        ], Style.MAGENTA, icon="📋")

        name = input(f"\n  {styled('▶', Style.CYAN)} Mod filename (e.g., 'vram_fix'): ").strip()
        if not name:
            print(f"  {styled('Cancelled', Style.YELLOW)}")
            return

        # Try clipboard first
        use_clipboard = input(f"  {styled('▶', Style.CYAN)} Paste from clipboard? (y/n): ").strip().lower()

        if use_clipboard == 'y':
            content = Clipboard.paste()
            if content:
                print(f"\n  {styled('Got content from clipboard:', Style.DIM)}")
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"  {preview}")
            else:
                print(f"  {styled('Clipboard empty or unavailable', Style.YELLOW)}")
                content = None
        else:
            content = None

        if not content:
            print(f"\n  {styled('Paste the Python code, then type END on a new line:', Style.DIM)}")
            print()

            lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == 'END':
                        break
                    lines.append(line)
                except EOFError:
                    break

            content = "\n".join(lines)

        if not content:
            print(f"  {styled('No content received', Style.YELLOW)}")
            return

        # Clean up code (remove markdown code blocks if present)
        if "```python" in content:
            content = content.split("```python")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        content = content.strip()

        # Validation
        if 'def apply' not in content:
            print(f"  {styled('⚠ Warning: No apply() function found', Style.YELLOW)}")
            confirm = input(f"  {styled('▶', Style.CYAN)} Save anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return

        if self.mod_loader.save_mod(name, content):
            full_name = name if name.endswith('.py') else f"{name}.py"
            print(f"  {styled('✓', Style.GREEN)} Saved: {styled(f'{MODS_DIR}/{full_name}', Style.BOLD)}")

            # Offer to apply immediately
            apply_now = input(f"  {styled('▶', Style.CYAN)} Apply now? (y/n): ").strip().lower()
            if apply_now == 'y':
                self.apply_mod_workflow()
        else:
            print(f"  {styled('✗', Style.RED)} Failed to save mod")

    def set_optimization_goal(self):
        """Set the optimization goal."""
        print_box("Set Optimization Goal", [
            "Describe what you're trying to achieve.",
            "This will be included in LLM prompts.",
            "",
            "Examples:",
            "• 'Reduce VRAM to fit on 8GB card'",
            "• 'Speed up generation without quality loss'",
            "• 'Fix the CUDA out of memory error'",
        ], Style.BLUE, icon="🎯")

        goal = input(f"\n  {styled('▶', Style.CYAN)} Your goal: ").strip()
        if goal:
            self.user_goal = goal
            self.config.set("last_goal", goal)
            print(f"  {styled('✓', Style.GREEN)} Goal set!")

    def test_comfyui_connection(self):
        """Test connection to ComfyUI."""
        print(f"\n  {styled('Testing connection...', Style.DIM)}")

        metrics = self.monitor.collect_metrics()

        if metrics["connected"]:
            print(f"  {styled('✓', Style.GREEN)} Connected to ComfyUI!")
            print(f"\n  {styled('System Info:', Style.BOLD)}")
            print(f"    GPU: {metrics.get('gpu_name', 'Unknown')}")
            print(f"    VRAM: {metrics['vram_used_gb']:.2f} / {metrics['vram_total_gb']:.2f} GB ({metrics['vram_percent']}%)")
            print(f"    Free: {metrics.get('vram_free_gb', 0):.2f} GB")
            print(f"    Queue: {metrics['queue_pending']} pending, {metrics['queue_running']} running")

            # Test object_info for node availability
            print(f"\n  {styled('Checking node availability...', Style.DIM)}")
            object_info = self.monitor.get_object_info()
            if object_info:
                print(f"    Available nodes: {len(object_info)}")
        else:
            print(f"  {styled('✗', Style.RED)} Could not connect to ComfyUI at {self.monitor.base_url}")
            print(f"    {styled('Make sure ComfyUI is running and accessible.', Style.DIM)}")

            # Offer to change URL
            change = input(f"\n  {styled('▶', Style.CYAN)} Change ComfyUI URL? (y/n): ").strip().lower()
            if change == 'y':
                new_url = input(f"  {styled('▶', Style.CYAN)} New URL: ").strip()
                if new_url:
                    self.config.set("comfy_url", new_url)
                    self.monitor = ComfyUIMonitor(new_url)
                    print(f"  {styled('✓', Style.GREEN)} URL updated!")

    def export_session(self):
        """Export session to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"session_{timestamp}.json"

        filepath = input(f"\n  {styled('▶', Style.CYAN)} Export filename [{default_name}]: ").strip()
        if not filepath:
            filepath = default_name

        if self.session.export(filepath):
            print(f"  {styled('✓', Style.GREEN)} Session exported to: {styled(filepath, Style.BOLD)}")
        else:
            print(f"  {styled('✗', Style.RED)} Failed to export session")

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        app = PerformanceLab()
        app.run()
    except KeyboardInterrupt:
        print(f"\n\n  {styled('Interrupted. Goodbye!', Style.DIM)}")
        sys.exit(0)

if __name__ == "__main__":
    main()
