"""
WhimWeaver Integration for Performance Lab v2.1
Launch, monitor, and optimize WhimWeaver workflows
"""

import subprocess
import os
import json
import time
import psutil
from typing import Optional, Dict, Any, List


class WhimWeaverMonitor:
    """Monitor WhimWeaver process performance."""

    def __init__(self, process_name: str = "python"):
        self.process_name = process_name
        self.process = None
        self.start_time = None
        self.metrics_history = []

    def find_process(self, command_filter: str = "whimweaver") -> Optional[psutil.Process]:
        """Find WhimWeaver process by command line."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                if command_filter.lower() in cmdline:
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def attach(self, pid: Optional[int] = None):
        """Attach to running WhimWeaver process."""
        if pid:
            try:
                self.process = psutil.Process(pid)
                self.start_time = time.time()
                return True
            except psutil.NoSuchProcess:
                return False
        else:
            self.process = self.find_process()
            if self.process:
                self.start_time = time.time()
                return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.process:
            return {"error": "No process attached"}

        try:
            # CPU and Memory
            cpu_percent = self.process.cpu_percent(interval=0.1)
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)

            # GPU metrics (if available)
            gpu_metrics = self._get_gpu_metrics()

            # Thread count
            num_threads = self.process.num_threads()

            # Runtime
            runtime = time.time() - self.start_time if self.start_time else 0

            metrics = {
                "timestamp": time.time(),
                "runtime_sec": runtime,
                "cpu_percent": cpu_percent,
                "memory_mb": mem_mb,
                "num_threads": num_threads,
                "status": self.process.status(),
                **gpu_metrics
            }

            self.metrics_history.append(metrics)
            return metrics

        except psutil.NoSuchProcess:
            return {"error": "Process terminated"}

    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics if torch is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "gpu_memory_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                    "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                }
        except:
            pass
        return {}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics_history:
            return {"error": "No metrics collected"}

        cpu_values = [m['cpu_percent'] for m in self.metrics_history if 'cpu_percent' in m]
        mem_values = [m['memory_mb'] for m in self.metrics_history if 'memory_mb' in m]

        return {
            "total_runtime_sec": self.metrics_history[-1]['runtime_sec'] if self.metrics_history else 0,
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "peak_memory_mb": max(mem_values) if mem_values else 0,
            "avg_memory_mb": sum(mem_values) / len(mem_values) if mem_values else 0,
            "samples_collected": len(self.metrics_history)
        }


class WhimWeaverLauncher:
    """Launch and manage WhimWeaver instances."""

    def __init__(self, whimweaver_path: Optional[str] = None):
        self.whimweaver_path = whimweaver_path or self._find_whimweaver()
        self.process = None
        self.monitor = WhimWeaverMonitor()

    def _find_whimweaver(self) -> Optional[str]:
        """Try to locate WhimWeaver in common locations."""
        search_paths = [
            os.path.expanduser("~/WhimWeaver"),
            os.path.expanduser("~/Projects/WhimWeaver"),
            "/opt/WhimWeaver",
            "../WhimWeaver",
            "../../WhimWeaver"
        ]

        for path in search_paths:
            if os.path.exists(path):
                # Look for main script
                for script in ['main.py', 'whimweaver.py', 'app.py', '__main__.py']:
                    full_path = os.path.join(path, script)
                    if os.path.exists(full_path):
                        return full_path

        return None

    def launch(
        self,
        workflow_path: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Launch WhimWeaver.

        Args:
            workflow_path: Path to workflow JSON to load
            args: Additional command-line arguments
            env: Environment variables

        Returns:
            True if launched successfully
        """
        if not self.whimweaver_path:
            return False

        cmd = ["python", self.whimweaver_path]

        if workflow_path:
            cmd.extend(["--workflow", workflow_path])

        if args:
            cmd.extend(args)

        # Prepare environment
        launch_env = os.environ.copy()
        if env:
            launch_env.update(env)

        try:
            self.process = subprocess.Popen(
                cmd,
                env=launch_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Attach monitor
            time.sleep(1)  # Give process time to start
            self.monitor.attach(self.process.pid)

            return True

        except Exception as e:
            print(f"Failed to launch WhimWeaver: {e}")
            return False

    def stop(self, timeout: int = 10):
        """Stop WhimWeaver gracefully."""
        if not self.process:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.monitor.get_metrics()

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.monitor.get_summary()

    def is_running(self) -> bool:
        """Check if WhimWeaver is still running."""
        if not self.process:
            return False
        return self.process.poll() is None


def detect_whimweaver_type(workflow_path: str) -> str:
    """
    Detect what type of AI generation WhimWeaver workflow does.

    Returns: "image", "video", "llm", "audio", "mixed", or "unknown"
    """
    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)

        nodes = workflow.get('nodes', [])
        node_types = [n.get('type', '').lower() for n in nodes]

        # Detection heuristics
        has_video = any('video' in t or 'animate' in t or 'frame' in t for t in node_types)
        has_image = any('image' in t or 'sampler' in t or 'checkpoint' in t for t in node_types)
        has_llm = any('llm' in t or 'text' in t or 'prompt' in t for t in node_types)
        has_audio = any('audio' in t or 'sound' in t or 'music' in t for t in node_types)

        types_found = sum([has_video, has_image, has_llm, has_audio])

        if types_found > 1:
            return "mixed"
        elif has_video:
            return "video"
        elif has_image:
            return "image"
        elif has_llm:
            return "llm"
        elif has_audio:
            return "audio"
        else:
            return "unknown"

    except Exception:
        return "unknown"
