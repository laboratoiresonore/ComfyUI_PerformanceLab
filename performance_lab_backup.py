#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMFYUI PERFORMANCE LAB v2.0                              ║
║          Iterative Workflow Optimization with LLM-Assisted Analysis          ║
╚══════════════════════════════════════════════════════════════════════════════╝

A beautiful, iterative mod manager that:
  • Applies workflow mods and tracks their impact
  • Collects rich metrics from ComfyUI (timing, memory, errors)
  • Generates copy-paste prompts for external LLM analysis
  • Maintains session history for iterative refinement
"""

import os
import sys
import json
import time
import copy
import importlib.util
import urllib.request
import urllib.error
from datetime import datetime
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List, Any

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODS_DIR = "mods"
COMFY_URL = "http://127.0.0.1:8188"
SESSION_LOG = "session_history.json"

# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL STYLING (works on most modern terminals)
# ═══════════════════════════════════════════════════════════════════════════════

class Style:
    """ANSI escape codes for terminal styling."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    
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

def styled(text: str, *styles) -> str:
    """Apply multiple styles to text."""
    prefix = "".join(styles)
    return f"{prefix}{text}{Style.RESET}" if prefix else text

def print_header(title: str, subtitle: str = ""):
    """Print a beautiful boxed header."""
    width = 70
    print()
    print(styled(Style.DBOX_TL + Style.DBOX_H * (width - 2) + Style.DBOX_TR, Style.CYAN))
    print(styled(Style.DBOX_V, Style.CYAN) + styled(title.center(width - 2), Style.BOLD, Style.WHITE) + styled(Style.DBOX_V, Style.CYAN))
    if subtitle:
        print(styled(Style.DBOX_V, Style.CYAN) + styled(subtitle.center(width - 2), Style.DIM) + styled(Style.DBOX_V, Style.CYAN))
    print(styled(Style.DBOX_BL + Style.DBOX_H * (width - 2) + Style.DBOX_BR, Style.CYAN))

def print_box(title: str, content: List[str], color=Style.WHITE):
    """Print content in a nice box."""
    width = 68
    print()
    print(styled(f"  {Style.BOX_TL}{Style.BOX_H} {title} {Style.BOX_H * (width - len(title) - 5)}{Style.BOX_TR}", color))
    for line in content:
        truncated = line[:width-4] if len(line) > width-4 else line
        padding = " " * (width - 4 - len(truncated))
        print(styled(f"  {Style.BOX_V} ", color) + truncated + padding + styled(f" {Style.BOX_V}", color))
    print(styled(f"  {Style.BOX_BL}{Style.BOX_H * (width - 2)}{Style.BOX_BR}", color))

def print_divider(char="─", color=Style.GRAY):
    print(styled(f"  {char * 66}", color))

# ═══════════════════════════════════════════════════════════════════════════════
# COMFYUI API INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class ComfyUIMonitor:
    """Handles all ComfyUI API interactions and metric collection."""
    
    def __init__(self, base_url: str = COMFY_URL):
        self.base_url = base_url
        self.connected = False
    
    def api_request(self, endpoint: str) -> Optional[Dict]:
        """Make an API request to ComfyUI."""
        try:
            url = f"{self.base_url}/{endpoint}"
            with urllib.request.urlopen(url, timeout=5) as response:
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
    
    def get_history(self) -> Dict:
        """Get execution history."""
        return self.api_request("history") or {}
    
    def collect_metrics(self) -> Dict:
        """Collect all available metrics from ComfyUI."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "connected": False,
            "vram_used_gb": None,
            "vram_total_gb": None,
            "vram_percent": None,
            "queue_pending": 0,
            "queue_running": 0,
        }
        
        # System stats
        stats = self.get_system_stats()
        if stats:
            metrics["connected"] = True
            devices = stats.get("devices", [])
            if devices:
                gpu = devices[0]
                vram_used = gpu.get("vram_used", 0)
                vram_total = gpu.get("vram_total", 1)
                metrics["vram_used_gb"] = round(vram_used / (1024**3), 2)
                metrics["vram_total_gb"] = round(vram_total / (1024**3), 2)
                metrics["vram_percent"] = round(100 * vram_used / vram_total, 1)
        
        # Queue status
        pending, running = self.get_queue_status()
        metrics["queue_pending"] = pending
        metrics["queue_running"] = running
        
        return metrics
    
    def wait_for_generation(self) -> Dict:
        """
        Wait for a generation to complete and collect metrics.
        Returns a rich metrics dictionary.
        """
        result = {
            "success": None,
            "duration_seconds": 0,
            "error_message": None,
            "peak_vram_gb": 0,
            "baseline_vram_gb": 0,
            "node_errors": [],
            "prompt_id": None,
        }
        
        print(f"\n  {styled('📡', Style.CYAN)} Connecting to ComfyUI at {styled(self.base_url, Style.BLUE)}...")
        
        # Check connection
        baseline = self.collect_metrics()
        if not baseline["connected"]:
            print(f"  {styled('⚠', Style.YELLOW)}  ComfyUI not detected. Is it running?")
            result["error_message"] = "ComfyUI not connected"
            return result
        
        result["baseline_vram_gb"] = baseline.get("vram_used_gb", 0)
        initial_history_ids = set(self.get_history().keys())
        
        print(f"  {styled('⏳', Style.YELLOW)} Waiting for generation...")
        print(f"     {styled('Load the _experimental file and click Queue Prompt', Style.DIM)}")
        print(f"     {styled('Press Ctrl+C to abort', Style.DIM)}")
        
        start_time = None
        is_running = False
        peak_vram = baseline.get("vram_used_gb", 0)
        
        try:
            while True:
                time.sleep(0.3)
                
                # Collect current metrics
                current = self.collect_metrics()
                if not current["connected"]:
                    continue
                
                # Track peak VRAM
                if current["vram_used_gb"]:
                    peak_vram = max(peak_vram, current["vram_used_gb"])
                
                pending = current["queue_pending"]
                running = current["queue_running"]
                
                # Detect start
                if not is_running and (pending > 0 or running > 0):
                    print(f"\n  {styled('🚀', Style.GREEN)} Generation STARTED!")
                    start_time = time.time()
                    is_running = True
                
                # Show progress indicator
                if is_running:
                    elapsed = time.time() - start_time
                    vram_str = f"{current['vram_used_gb']:.1f}GB" if current['vram_used_gb'] else "N/A"
                    print(f"\r  {styled('⚙', Style.CYAN)}  Running... {elapsed:.1f}s | VRAM: {vram_str}    ", end="", flush=True)
                
                # Detect finish
                current_history = self.get_history()
                current_ids = set(current_history.keys())
                new_ids = current_ids - initial_history_ids
                
                if is_running and new_ids:
                    print()  # Newline after progress
                    end_time = time.time()
                    result["duration_seconds"] = round(end_time - start_time, 2)
                    result["peak_vram_gb"] = round(peak_vram, 2)
                    
                    # Check the result
                    latest_id = sorted(new_ids)[-1]
                    result["prompt_id"] = latest_id
                    run_data = current_history.get(latest_id, {})
                    
                    status = run_data.get("status", {})
                    status_str = status.get("status_str", "success")
                    
                    if status_str == "error":
                        result["success"] = False
                        # Extract error details
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
            print(f"\n  {styled('🛑', Style.RED)} Aborted by user")
            result["success"] = None
            result["error_message"] = "Aborted by user"
            return result

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowAnalyzer:
    """Analyzes ComfyUI workflow structure for LLM context."""
    
    @staticmethod
    def analyze(workflow: Dict) -> Dict:
        """Extract key information from a workflow."""
        analysis = {
            "node_count": 0,
            "node_types": {},
            "groups": [],
            "has_video_nodes": False,
            "has_upscale_nodes": False,
            "has_controlnet": False,
            "model_loaders": [],
            "resolution_hints": [],
            "connections_count": 0,
        }
        
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        groups = workflow.get("groups", [])
        
        analysis["node_count"] = len(nodes)
        analysis["connections_count"] = len(links)
        
        # Analyze nodes
        for node in nodes:
            node_type = node.get("type", "Unknown")
            analysis["node_types"][node_type] = analysis["node_types"].get(node_type, 0) + 1
            
            type_lower = node_type.lower()
            
            # Detect patterns
            if "video" in type_lower or "vhs" in type_lower:
                analysis["has_video_nodes"] = True
            if "upscale" in type_lower or "esrgan" in type_lower:
                analysis["has_upscale_nodes"] = True
            if "controlnet" in type_lower:
                analysis["has_controlnet"] = True
            if "loader" in type_lower and "model" in type_lower:
                analysis["model_loaders"].append(node_type)
            
            # Look for resolution values in widgets
            widgets = node.get("widgets_values", [])
            for w in widgets:
                if isinstance(w, int) and 256 <= w <= 2048 and w % 8 == 0:
                    if w not in analysis["resolution_hints"]:
                        analysis["resolution_hints"].append(w)
        
        # Groups
        for group in groups:
            analysis["groups"].append({
                "title": group.get("title", "Untitled"),
                "color": group.get("color", ""),
            })
        
        return analysis
    
    @staticmethod
    def summarize(analysis: Dict) -> str:
        """Create a human-readable summary."""
        lines = []
        lines.append(f"• {analysis['node_count']} nodes, {analysis['connections_count']} connections")
        
        # Top node types
        sorted_types = sorted(analysis['node_types'].items(), key=lambda x: -x[1])[:8]
        type_str = ", ".join([f"{t}({c})" for t, c in sorted_types])
        lines.append(f"• Top node types: {type_str}")
        
        # Features
        features = []
        if analysis["has_video_nodes"]: features.append("Video")
        if analysis["has_upscale_nodes"]: features.append("Upscaling")
        if analysis["has_controlnet"]: features.append("ControlNet")
        if features:
            lines.append(f"• Features: {', '.join(features)}")
        
        if analysis["resolution_hints"]:
            lines.append(f"• Resolution hints: {analysis['resolution_hints']}")
        
        if analysis["groups"]:
            group_names = [g["title"] for g in analysis["groups"][:5]]
            lines.append(f"• Groups: {', '.join(group_names)}")
        
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION HISTORY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class SessionHistory:
    """Tracks all modifications and their results within a session."""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries: List[Dict] = []
        self.baseline_metrics: Optional[Dict] = None
        self.target_workflow: Optional[str] = None
        self.workflow_analysis: Optional[Dict] = None
    
    def set_baseline(self, workflow_path: str, analysis: Dict, metrics: Dict):
        """Set the baseline for comparison."""
        self.target_workflow = workflow_path
        self.workflow_analysis = analysis
        self.baseline_metrics = metrics
    
    def add_entry(self, mod_name: str, mod_description: str, result: Dict, kept: bool):
        """Add a modification entry."""
        entry = {
            "iteration": len(self.entries) + 1,
            "timestamp": datetime.now().isoformat(),
            "mod_name": mod_name,
            "mod_description": mod_description,
            "result": result,
            "kept": kept,
        }
        self.entries.append(entry)
    
    def get_summary(self) -> str:
        """Get a summary of all modifications in this session."""
        if not self.entries:
            return "No modifications applied yet."
        
        lines = []
        for e in self.entries:
            status = "✓ KEPT" if e["kept"] else "✗ REVERTED"
            duration = e["result"].get("duration_seconds", "N/A")
            success = "OK" if e["result"].get("success") else "FAIL"
            lines.append(f"  {e['iteration']}. [{status}] {e['mod_name']} → {success} ({duration}s)")
        
        return "\n".join(lines)
    
    def get_performance_trend(self) -> List[Tuple[str, float]]:
        """Get the performance trend of kept modifications."""
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
            "entries": self.entries,
        }

# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROMPT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class LLMPromptGenerator:
    """Generates copy-paste prompts for external LLM analysis."""
    
    @staticmethod
    def generate_analysis_prompt(
        session: SessionHistory,
        latest_result: Dict,
        workflow_content: Dict,
        user_goal: str = ""
    ) -> str:
        """Generate a comprehensive prompt for LLM analysis."""
        
        prompt_parts = []
        
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
            for e in session.entries:
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
        
        # Node type distribution (condensed)
        if session.workflow_analysis:
            prompt_parts.append("## KEY NODE TYPES (count)")
            types = session.workflow_analysis.get("node_types", {})
            sorted_types = sorted(types.items(), key=lambda x: -x[1])[:15]
            type_lines = [f"{t}: {c}" for t, c in sorted_types]
            prompt_parts.append(", ".join(type_lines))
            prompt_parts.append("")
        
        # Request
        prompt_parts.append("## REQUEST")
        prompt_parts.append("Based on this information, please suggest a specific mod I can apply to")
        prompt_parts.append("improve this workflow. Provide your suggestion as a Python mod file that")
        prompt_parts.append("follows this structure:")
        prompt_parts.append("")
        prompt_parts.append("```python")
        prompt_parts.append('description = "Brief description of what this mod does"')
        prompt_parts.append("")
        prompt_parts.append("def apply(content):")
        prompt_parts.append('    """')
        prompt_parts.append("    content: dict (parsed JSON workflow)")
        prompt_parts.append("    returns: modified dict, or None if no changes")
        prompt_parts.append('    """')
        prompt_parts.append("    # Your modification logic here")
        prompt_parts.append("    return content")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("Focus on: VRAM optimization, speed improvements, or fixing the error above.")
        prompt_parts.append("=" * 70)
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def generate_initial_prompt(workflow_content: Dict, user_goal: str = "") -> str:
        """Generate the initial analysis prompt before any mods."""
        
        analysis = WorkflowAnalyzer.analyze(workflow_content)
        
        prompt_parts = []
        prompt_parts.append("=" * 70)
        prompt_parts.append("COMFYUI WORKFLOW OPTIMIZATION - INITIAL ANALYSIS")
        prompt_parts.append("=" * 70)
        prompt_parts.append("")
        
        prompt_parts.append("## CONTEXT")
        prompt_parts.append("I have a ComfyUI workflow I'd like to optimize. Please analyze it and")
        prompt_parts.append("suggest modifications I can make.")
        if user_goal:
            prompt_parts.append(f"**My Goal:** {user_goal}")
        prompt_parts.append("")
        
        prompt_parts.append("## WORKFLOW ANALYSIS")
        prompt_parts.append(WorkflowAnalyzer.summarize(analysis))
        prompt_parts.append("")
        
        # Include node types
        prompt_parts.append("## ALL NODE TYPES")
        types = analysis.get("node_types", {})
        sorted_types = sorted(types.items(), key=lambda x: -x[1])
        for t, c in sorted_types:
            prompt_parts.append(f"  • {t}: {c}")
        prompt_parts.append("")
        
        # Include a condensed JSON representation
        prompt_parts.append("## WORKFLOW STRUCTURE (condensed)")
        prompt_parts.append("Here are the nodes with their IDs, types, and key widget values:")
        prompt_parts.append("")
        
        nodes = workflow_content.get("nodes", [])
        for node in nodes[:50]:  # Limit to first 50 nodes
            node_id = node.get("id", "?")
            node_type = node.get("type", "Unknown")
            widgets = node.get("widgets_values", [])
            # Condense widgets
            widgets_preview = str(widgets)[:80] + "..." if len(str(widgets)) > 80 else str(widgets)
            prompt_parts.append(f"[{node_id}] {node_type}: {widgets_preview}")
        
        if len(nodes) > 50:
            prompt_parts.append(f"... and {len(nodes) - 50} more nodes")
        
        prompt_parts.append("")
        prompt_parts.append("## REQUEST")
        prompt_parts.append("Please suggest an optimization mod. Respond with Python code following this structure:")
        prompt_parts.append("")
        prompt_parts.append("```python")
        prompt_parts.append('description = "Brief description"')
        prompt_parts.append("def apply(content):")
        prompt_parts.append("    # content is the parsed JSON workflow dict")
        prompt_parts.append("    # Return modified dict or None")
        prompt_parts.append("    return content")
        prompt_parts.append("```")
        prompt_parts.append("=" * 70)
        
        return "\n".join(prompt_parts)

# ═══════════════════════════════════════════════════════════════════════════════
# MOD LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class ModLoader:
    """Handles loading and managing mod files."""
    
    def __init__(self, mods_dir: str = MODS_DIR):
        self.mods_dir = mods_dir
        if not os.path.exists(mods_dir):
            os.makedirs(mods_dir)
    
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
    
    def save_mod_from_clipboard(self, name: str, content: str) -> bool:
        """Save a new mod from pasted content."""
        if not name.endswith('.py'):
            name += '.py'
        path = os.path.join(self.mods_dir, name)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
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

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceLab:
    """Main application controller."""
    
    def __init__(self):
        self.monitor = ComfyUIMonitor()
        self.mod_loader = ModLoader()
        self.session = SessionHistory()
        self.target_workflow: Optional[str] = None
        self.workflow_content: Optional[Dict] = None
        self.user_goal: str = ""
    
    def run(self):
        """Main application loop."""
        print_header(
            "COMFYUI PERFORMANCE LAB v2.0",
            "Iterative Workflow Optimization with LLM-Assisted Analysis"
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
                print(f"\n  {styled('Goodbye!', Style.DIM)}")
                break
            elif choice == '1':
                self.apply_mod_workflow()
            elif choice == '2':
                self.generate_llm_prompt()
            elif choice == '3':
                self.paste_new_mod()
            elif choice == '4':
                self.view_session_history()
            elif choice == '5':
                self.set_optimization_goal()
            elif choice == '6':
                self.test_comfyui_connection()
            elif choice == 't':
                self.setup_target_workflow()
            else:
                print(f"  {styled('Invalid choice', Style.YELLOW)}")
    
    def setup_target_workflow(self):
        """Set up the target workflow file."""
        print_box("Target Workflow", [
            "Enter the workflow JSON file you want to optimize.",
            "This will be your base for all modifications.",
        ], Style.BLUE)
        
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
        
        # Analyze workflow
        analysis = WorkflowAnalyzer.analyze(content)
        baseline_metrics = self.monitor.collect_metrics()
        self.session.set_baseline(filepath, analysis, baseline_metrics)
        
        print(f"\n  {styled('✓', Style.GREEN)} Workflow loaded: {styled(filepath, Style.BOLD)}")
        print_box("Workflow Analysis", WorkflowAnalyzer.summarize(analysis).split("\n"), Style.GREEN)
    
    def show_main_menu(self):
        """Display the main menu."""
        print()
        print_divider("═", Style.CYAN)
        print(f"  {styled('MAIN MENU', Style.BOLD)}")
        print(f"  Target: {styled(self.target_workflow or 'Not set', Style.BLUE)}")
        if self.user_goal:
            print(f"  Goal: {styled(self.user_goal[:50], Style.MAGENTA)}")
        print_divider()
        
        mods = self.mod_loader.list_mods()
        
        menu_items = [
            ("1", "Apply a Mod", f"{len(mods)} mods available"),
            ("2", "Generate LLM Prompt", "Copy-paste for external analysis"),
            ("3", "Paste New Mod", "Add mod from LLM response"),
            ("4", "View Session History", f"{len(self.session.entries)} modifications"),
            ("5", "Set Optimization Goal", self.user_goal[:30] + "..." if len(self.user_goal) > 30 else (self.user_goal or "Not set")),
            ("6", "Test ComfyUI Connection", "Check API access"),
            ("T", "Change Target Workflow", ""),
            ("Q", "Quit", ""),
        ]
        
        for key, label, hint in menu_items:
            hint_str = styled(f" ({hint})", Style.DIM) if hint else ""
            print(f"    {styled(key, Style.CYAN)}  {label}{hint_str}")
    
    def apply_mod_workflow(self):
        """Apply a mod to the workflow."""
        mods = self.mod_loader.list_mods()
        
        if not mods:
            print(f"\n  {styled('⚠', Style.YELLOW)} No mods found in '{MODS_DIR}/' directory.")
            print(f"  {styled('Tip:', Style.DIM)} Use option 3 to paste a mod from an LLM response.")
            return
        
        print_box("Available Mods", [], Style.MAGENTA)
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
        
        # Load and apply mod
        print(f"\n  {styled('⚙', Style.CYAN)} Loading mod: {mod_name}")
        
        try:
            mod = self.mod_loader.load_mod(mod_name)
            mod_desc = getattr(mod, 'description', 'No description')
            
            # Apply to a copy
            content_copy = copy.deepcopy(self.workflow_content)
            new_content = mod.apply(content_copy)
            
            if new_content is None or new_content == self.workflow_content:
                print(f"  {styled('⚠', Style.YELLOW)} Mod made no changes.")
                return
            
            # Save experimental version
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
                    # Replace original with experimental
                    os.replace(exp_path, self.target_workflow)
                    self.workflow_content = new_content
                    print(f"  {styled('✓', Style.GREEN)} Changes applied to: {self.target_workflow}")
                else:
                    if os.path.exists(exp_path):
                        os.remove(exp_path)
                    print(f"  {styled('↩', Style.YELLOW)} Changes reverted.")
                
                # Record in session
                self.session.add_entry(mod_name, mod_desc, result, keep)
            else:
                # User aborted or error
                if os.path.exists(exp_path):
                    os.remove(exp_path)
                    
        except Exception as e:
            print(f"  {styled('✗', Style.RED)} Error applying mod: {e}")
    
    def show_generation_result(self, result: Dict):
        """Display generation results nicely."""
        print()
        print_divider("═")
        
        if result.get("success"):
            print(f"  {styled('✨ GENERATION SUCCESSFUL', Style.GREEN, Style.BOLD)}")
            print()
            duration = result.get("duration_seconds", 0)
            peak_vram = result.get("peak_vram_gb", 0)
            base_vram = result.get("baseline_vram_gb", 0)
            print(f"    Duration:    {styled(f'{duration:.2f}s', Style.BOLD)}")
            print(f"    Peak VRAM:   {styled(f'{peak_vram:.2f} GB', Style.BOLD)}")
            print(f"    Base VRAM:   {base_vram:.2f} GB")
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
    
    def generate_llm_prompt(self):
        """Generate a prompt for external LLM analysis."""
        print_box("Generate LLM Prompt", [
            "This will create a prompt you can copy-paste to an external",
            "LLM (like Claude) for workflow optimization suggestions.",
        ], Style.BLUE)
        
        # Check if we have test results
        latest_result = {}
        if self.session.entries:
            latest_result = self.session.entries[-1].get("result", {})
        
        print(f"\n  {styled('Options:', Style.DIM)}")
        print(f"    {styled('1', Style.CYAN)}  Initial Analysis (first time)")
        print(f"    {styled('2', Style.CYAN)}  After-Test Analysis (with results)")
        
        choice = input(f"\n  {styled('▶', Style.CYAN)} Choice: ").strip()
        
        if choice == '1':
            prompt = LLMPromptGenerator.generate_initial_prompt(
                self.workflow_content, 
                self.user_goal
            )
        elif choice == '2':
            prompt = LLMPromptGenerator.generate_analysis_prompt(
                self.session,
                latest_result,
                self.workflow_content,
                self.user_goal
            )
        else:
            print(f"  {styled('Invalid choice', Style.YELLOW)}")
            return
        
        # Display the prompt
        print()
        print(styled("─" * 70, Style.GRAY))
        print(styled("COPY THE FOLLOWING PROMPT:", Style.BOLD, Style.GREEN))
        print(styled("─" * 70, Style.GRAY))
        print()
        print(prompt)
        print()
        print(styled("─" * 70, Style.GRAY))
        print(styled("END OF PROMPT", Style.BOLD, Style.GREEN))
        print(styled("─" * 70, Style.GRAY))
        
        # Try to copy to clipboard
        self.try_copy_to_clipboard(prompt)
        
        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")
    
    def try_copy_to_clipboard(self, text: str):
        """Attempt to copy text to clipboard."""
        try:
            # Try using xclip (Linux)
            import subprocess
            process = subprocess.Popen(['xclip', '-selection', 'clipboard'], 
                                       stdin=subprocess.PIPE)
            process.communicate(text.encode())
            print(f"\n  {styled('📋 Copied to clipboard!', Style.GREEN)}")
        except Exception:
            try:
                # Try using pbcopy (macOS)
                import subprocess
                process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                process.communicate(text.encode())
                print(f"\n  {styled('📋 Copied to clipboard!', Style.GREEN)}")
            except Exception:
                print(f"\n  {styled('📋 Select and copy the text above manually.', Style.YELLOW)}")
    
    def paste_new_mod(self):
        """Allow user to paste a new mod from LLM response."""
        print_box("Paste New Mod", [
            "Paste the Python mod code from the LLM response.",
            "The code should have a 'description' and 'apply(content)' function.",
            "End with a blank line followed by 'END' on its own line.",
        ], Style.MAGENTA)
        
        name = input(f"\n  {styled('▶', Style.CYAN)} Mod filename (e.g., 'vram_fix'): ").strip()
        if not name:
            print(f"  {styled('Cancelled', Style.YELLOW)}")
            return
        
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
        
        if not lines:
            print(f"  {styled('No content received', Style.YELLOW)}")
            return
        
        content = "\n".join(lines)
        
        # Basic validation
        if 'def apply' not in content:
            print(f"  {styled('⚠ Warning: No apply() function found in code', Style.YELLOW)}")
            confirm = input(f"  {styled('▶', Style.CYAN)} Save anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return
        
        if self.mod_loader.save_mod_from_clipboard(name, content):
            full_name = name if name.endswith('.py') else f"{name}.py"
            print(f"  {styled('✓', Style.GREEN)} Saved: {styled(f'{MODS_DIR}/{full_name}', Style.BOLD)}")
        else:
            print(f"  {styled('✗', Style.RED)} Failed to save mod")
    
    def view_session_history(self):
        """View the session modification history."""
        if not self.session.entries:
            print(f"\n  {styled('No modifications applied yet in this session.', Style.DIM)}")
            return
        
        print_box(f"Session History ({self.session.session_id})", [], Style.CYAN)
        
        for entry in self.session.entries:
            status_icon = styled("✓", Style.GREEN) if entry["kept"] else styled("✗", Style.RED)
            iteration_num = entry["iteration"]
            mod_name = entry["mod_name"]
            print(f"\n  {status_icon} {styled(f'#{iteration_num}', Style.BOLD)} {mod_name}")
            print(f"     {styled(entry['mod_description'][:60], Style.DIM)}")
            
            result = entry["result"]
            if result.get("success"):
                print(f"     Duration: {result['duration_seconds']}s | VRAM: {result.get('peak_vram_gb', 'N/A')} GB")
            elif result.get("success") is False:
                print(f"     {styled('FAILED:', Style.RED)} {result.get('error_message', 'Unknown')[:50]}")
            
            status = styled("KEPT", Style.GREEN) if entry["kept"] else styled("REVERTED", Style.YELLOW)
            print(f"     Status: {status}")
        
        # Performance trend
        trend = self.session.get_performance_trend()
        if len(trend) > 1:
            print()
            print_divider()
            print(f"  {styled('Performance Trend:', Style.BOLD)}")
            for i, (name, duration) in enumerate(trend):
                bar_len = int(duration / max(t[1] for t in trend) * 30)
                bar = styled("█" * bar_len, Style.CYAN)
                print(f"    {i+1}. {bar} {duration:.1f}s")
        
        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")
    
    def set_optimization_goal(self):
        """Set the user's optimization goal."""
        print_box("Set Optimization Goal", [
            "Describe what you're trying to achieve. This will be included",
            "in the LLM prompts to help get better suggestions.",
            "Examples: 'Reduce VRAM to fit on 8GB card'",
            "          'Speed up generation without quality loss'",
        ], Style.BLUE)
        
        goal = input(f"\n  {styled('▶', Style.CYAN)} Your goal: ").strip()
        if goal:
            self.user_goal = goal
            print(f"  {styled('✓', Style.GREEN)} Goal set!")
    
    def test_comfyui_connection(self):
        """Test connection to ComfyUI."""
        print(f"\n  {styled('Testing connection...', Style.DIM)}")
        
        metrics = self.monitor.collect_metrics()
        
        if metrics["connected"]:
            print(f"  {styled('✓', Style.GREEN)} Connected to ComfyUI!")
            print(f"    VRAM: {metrics['vram_used_gb']:.2f} / {metrics['vram_total_gb']:.2f} GB ({metrics['vram_percent']}%)")
            print(f"    Queue: {metrics['queue_pending']} pending, {metrics['queue_running']} running")
        else:
            print(f"  {styled('✗', Style.RED)} Could not connect to ComfyUI at {COMFY_URL}")
            print(f"    {styled('Make sure ComfyUI is running and accessible.', Style.DIM)}")

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
