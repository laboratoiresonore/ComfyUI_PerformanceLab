#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🎯 LORA OPTIMIZER - Optician-Style Settings Tuner 🎯            ║
║        Harvest metadata + A/B test your way to perfect generations           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
  • Harvest metadata from existing generations (LoRA Manager compatible)
  • A/B "optician" testing: "Do you prefer A or B?" to fine-tune settings
  • Auto-detect optimal CFG, steps, prompts from example images
  • Generate LLM prompts for settings that can't be visually tested
  • Works with SD 1.5, SDXL, Flux models
"""

import os
import sys
import json
import time
import copy
import random
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path

# Import from performance_lab
try:
    from performance_lab import (
        Style, styled, print_header, print_box, print_divider,
        ComfyUIMonitor, read_workflow, write_workflow, create_experimental_path,
        Clipboard, clear_screen
    )
except ImportError:
    # Fallback styling if performance_lab not found
    class Style:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"
        GRAY = "\033[90m"

    def styled(text, *styles):
        return f"{''.join(styles)}{text}{Style.RESET}"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

COMFY_URL = "http://127.0.0.1:8188"
COMFY_OUTPUT_DIR = None  # Auto-detected
LORA_MANAGER_DIR = None  # Auto-detected

# Default parameter ranges for A/B testing
DEFAULT_RANGES = {
    "cfg": {"min": 1.0, "max": 15.0, "step": 0.5, "default": 7.0},
    "steps": {"min": 10, "max": 50, "step": 5, "default": 20},
    "denoise": {"min": 0.3, "max": 1.0, "step": 0.1, "default": 1.0},
    "sampler": ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim"],
    "scheduler": ["normal", "karras", "exponential", "sgm_uniform"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# METADATA HARVESTER
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataHarvester:
    """
    Harvests generation metadata from:
    - PNG EXIF data (ComfyUI embeds workflow in PNG)
    - LoRA Manager's saved configurations
    - CivitAI model pages (if URL provided)
    - Workflow JSON files
    """

    def __init__(self):
        self.harvested_settings: List[Dict] = []

    def harvest_from_png(self, png_path: str) -> Optional[Dict]:
        """Extract embedded metadata from ComfyUI-generated PNG."""
        try:
            with open(png_path, 'rb') as f:
                data = f.read()

            # Look for tEXt chunk with "prompt" or "workflow"
            settings = {}

            # ComfyUI embeds JSON in PNG tEXt chunks
            # Simple extraction - look for JSON patterns
            text_start = data.find(b'{"prompt"')
            if text_start == -1:
                text_start = data.find(b'{"workflow"')

            if text_start != -1:
                # Find the end of JSON
                brace_count = 0
                end_idx = text_start
                for i in range(text_start, len(data)):
                    if data[i:i+1] == b'{':
                        brace_count += 1
                    elif data[i:i+1] == b'}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

                try:
                    json_str = data[text_start:end_idx].decode('utf-8', errors='ignore')
                    metadata = json.loads(json_str)
                    settings = self._extract_settings_from_workflow(metadata)
                except json.JSONDecodeError:
                    pass

            if settings:
                settings["source"] = png_path
                settings["source_type"] = "png_metadata"
                return settings

        except Exception as e:
            print(f"  {styled('⚠', Style.YELLOW)} Could not read {png_path}: {e}")

        return None

    def harvest_from_workflow(self, workflow_path: str) -> Optional[Dict]:
        """Extract settings from a workflow JSON file."""
        try:
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)

            settings = self._extract_settings_from_workflow(workflow)
            if settings:
                settings["source"] = workflow_path
                settings["source_type"] = "workflow_json"
                return settings

        except Exception as e:
            print(f"  {styled('⚠', Style.YELLOW)} Could not read {workflow_path}: {e}")

        return None

    def _extract_settings_from_workflow(self, workflow: Dict) -> Dict:
        """Extract relevant settings from workflow structure."""
        settings = {
            "cfg": None,
            "steps": None,
            "sampler": None,
            "scheduler": None,
            "denoise": None,
            "positive_prompt": None,
            "negative_prompt": None,
            "model": None,
            "loras": [],
            "width": None,
            "height": None,
            "seed": None,
        }

        # Handle both prompt format and workflow format
        nodes = workflow.get("nodes", [])
        prompt_data = workflow.get("prompt", {})

        # If it's prompt format, convert
        if prompt_data and not nodes:
            nodes = self._prompt_to_nodes(prompt_data)

        for node in nodes:
            node_type = node.get("type", "").lower()
            widgets = node.get("widgets_values", [])
            inputs = node.get("inputs", {})

            # KSampler / KSamplerAdvanced
            if "ksampler" in node_type or "sampler" in node_type:
                # Typical order: seed, steps, cfg, sampler_name, scheduler, denoise
                if len(widgets) >= 5:
                    try:
                        settings["seed"] = widgets[0] if isinstance(widgets[0], int) else None
                        settings["steps"] = widgets[1] if isinstance(widgets[1], int) else None
                        settings["cfg"] = widgets[2] if isinstance(widgets[2], (int, float)) else None
                        settings["sampler"] = widgets[3] if isinstance(widgets[3], str) else None
                        settings["scheduler"] = widgets[4] if isinstance(widgets[4], str) else None
                        if len(widgets) > 5:
                            settings["denoise"] = widgets[5] if isinstance(widgets[5], (int, float)) else None
                    except (IndexError, TypeError):
                        pass

            # CLIP Text Encode (prompts)
            elif "cliptext" in node_type or "text" in node_type.lower():
                if widgets:
                    text = widgets[0] if isinstance(widgets[0], str) else None
                    if text:
                        # Heuristic: negative prompts usually contain certain keywords
                        negative_keywords = ["ugly", "bad", "worst", "blurry", "low quality", "deformed"]
                        is_negative = any(kw in text.lower() for kw in negative_keywords)

                        if is_negative and not settings["negative_prompt"]:
                            settings["negative_prompt"] = text
                        elif not settings["positive_prompt"]:
                            settings["positive_prompt"] = text

            # Checkpoint Loader
            elif "checkpoint" in node_type or "loader" in node_type:
                if widgets:
                    for w in widgets:
                        if isinstance(w, str) and (".safetensors" in w or ".ckpt" in w):
                            settings["model"] = w
                            break

            # LoRA Loader
            elif "lora" in node_type:
                if widgets:
                    for w in widgets:
                        if isinstance(w, str) and ".safetensors" in w:
                            settings["loras"].append(w)

            # Empty Latent Image (resolution)
            elif "emptylatent" in node_type or "latent" in node_type:
                if len(widgets) >= 2:
                    try:
                        w = widgets[0]
                        h = widgets[1]
                        if isinstance(w, int) and isinstance(h, int):
                            settings["width"] = w
                            settings["height"] = h
                    except (IndexError, TypeError):
                        pass

        return settings

    def _prompt_to_nodes(self, prompt_data: Dict) -> List[Dict]:
        """Convert ComfyUI prompt format to nodes format."""
        nodes = []
        for node_id, node_data in prompt_data.items():
            node = {
                "id": node_id,
                "type": node_data.get("class_type", ""),
                "widgets_values": list(node_data.get("inputs", {}).values()),
                "inputs": node_data.get("inputs", {}),
            }
            nodes.append(node)
        return nodes

    def harvest_from_directory(self, directory: str, file_types: List[str] = None) -> List[Dict]:
        """Harvest metadata from all compatible files in a directory."""
        if file_types is None:
            file_types = [".png", ".json"]

        results = []
        path = Path(directory)

        for file_path in path.rglob("*"):
            if file_path.suffix.lower() in file_types:
                if file_path.suffix.lower() == ".png":
                    settings = self.harvest_from_png(str(file_path))
                elif file_path.suffix.lower() == ".json":
                    settings = self.harvest_from_workflow(str(file_path))
                else:
                    continue

                if settings and any(v is not None for k, v in settings.items() if k not in ["source", "source_type", "loras"]):
                    results.append(settings)

        self.harvested_settings.extend(results)
        return results

    def get_best_practices(self) -> Dict:
        """Analyze harvested settings and determine best practices."""
        if not self.harvested_settings:
            return {}

        best_practices = {
            "cfg": {"values": [], "recommended": None},
            "steps": {"values": [], "recommended": None},
            "sampler": {"values": [], "recommended": None},
            "scheduler": {"values": [], "recommended": None},
            "denoise": {"values": [], "recommended": None},
            "negative_prompt_patterns": [],
            "positive_prompt_patterns": [],
        }

        for settings in self.harvested_settings:
            if settings.get("cfg") is not None:
                best_practices["cfg"]["values"].append(settings["cfg"])
            if settings.get("steps") is not None:
                best_practices["steps"]["values"].append(settings["steps"])
            if settings.get("sampler") is not None:
                best_practices["sampler"]["values"].append(settings["sampler"])
            if settings.get("scheduler") is not None:
                best_practices["scheduler"]["values"].append(settings["scheduler"])
            if settings.get("denoise") is not None:
                best_practices["denoise"]["values"].append(settings["denoise"])
            if settings.get("negative_prompt"):
                best_practices["negative_prompt_patterns"].append(settings["negative_prompt"])
            if settings.get("positive_prompt"):
                best_practices["positive_prompt_patterns"].append(settings["positive_prompt"])

        # Calculate recommendations
        for key in ["cfg", "steps", "denoise"]:
            values = best_practices[key]["values"]
            if values:
                best_practices[key]["recommended"] = sum(values) / len(values)
                best_practices[key]["min"] = min(values)
                best_practices[key]["max"] = max(values)

        for key in ["sampler", "scheduler"]:
            values = best_practices[key]["values"]
            if values:
                # Most common value
                from collections import Counter
                counter = Counter(values)
                best_practices[key]["recommended"] = counter.most_common(1)[0][0]
                best_practices[key]["distribution"] = dict(counter)

        return best_practices

# ═══════════════════════════════════════════════════════════════════════════════
# A/B OPTICIAN TESTER
# ═══════════════════════════════════════════════════════════════════════════════

class OpticianTester:
    """
    "Better A or B?" style testing for visual settings.
    Uses binary search to efficiently find optimal values.
    """

    def __init__(self, monitor: ComfyUIMonitor, workflow_path: str):
        self.monitor = monitor
        self.workflow_path = workflow_path
        self.workflow_content = None
        self.test_history: List[Dict] = []
        self.current_settings: Dict = {}

        # Load workflow
        self._load_workflow()

    def _load_workflow(self):
        """Load the target workflow."""
        try:
            with open(self.workflow_path, 'r') as f:
                self.workflow_content = json.load(f)
        except Exception as e:
            print(f"  {styled('✗', Style.RED)} Error loading workflow: {e}")

    def _find_sampler_node(self) -> Optional[Dict]:
        """Find the KSampler node in the workflow."""
        if not self.workflow_content:
            return None

        nodes = self.workflow_content.get("nodes", [])
        for node in nodes:
            node_type = node.get("type", "").lower()
            if "ksampler" in node_type or "sampler" in node_type:
                return node
        return None

    def _create_variant(self, setting_name: str, value: Any) -> Dict:
        """Create a workflow variant with a modified setting."""
        variant = copy.deepcopy(self.workflow_content)

        nodes = variant.get("nodes", [])
        for node in nodes:
            node_type = node.get("type", "").lower()
            widgets = node.get("widgets_values", [])

            if "ksampler" in node_type or "sampler" in node_type:
                # Widget indices for typical KSampler:
                # 0: seed, 1: steps, 2: cfg, 3: sampler_name, 4: scheduler, 5: denoise
                if setting_name == "steps" and len(widgets) > 1:
                    widgets[1] = int(value)
                elif setting_name == "cfg" and len(widgets) > 2:
                    widgets[2] = float(value)
                elif setting_name == "sampler" and len(widgets) > 3:
                    widgets[3] = str(value)
                elif setting_name == "scheduler" and len(widgets) > 4:
                    widgets[4] = str(value)
                elif setting_name == "denoise" and len(widgets) > 5:
                    widgets[5] = float(value)
                elif setting_name == "seed" and len(widgets) > 0:
                    widgets[0] = int(value)

        return variant

    def _queue_workflow(self, workflow: Dict, client_id: str = "optician") -> Optional[str]:
        """Queue a workflow for execution via API."""
        try:
            # Convert workflow to API prompt format
            prompt = self._workflow_to_prompt(workflow)

            data = {
                "prompt": prompt,
                "client_id": client_id,
            }

            url = f"{self.monitor.base_url}/prompt"
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                return result.get("prompt_id")

        except Exception as e:
            print(f"  {styled('✗', Style.RED)} Error queueing workflow: {e}")
            return None

    def _workflow_to_prompt(self, workflow: Dict) -> Dict:
        """Convert workflow format to API prompt format."""
        # This is a simplified conversion - may need adjustment
        # ComfyUI's prompt format is different from the workflow save format
        prompt = {}

        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])

        # Build link lookup
        link_lookup = {}
        for link in links:
            # link format: [link_id, from_node, from_slot, to_node, to_slot, type]
            if len(link) >= 5:
                link_id, from_node, from_slot, to_node, to_slot = link[:5]
                link_lookup[link_id] = {
                    "from_node": from_node,
                    "from_slot": from_slot,
                }

        for node in nodes:
            node_id = str(node.get("id"))
            node_type = node.get("type")
            widgets = node.get("widgets_values", [])
            inputs = node.get("inputs", [])

            prompt[node_id] = {
                "class_type": node_type,
                "inputs": {}
            }

            # Add widget values as inputs
            # This requires knowing the input names for each node type
            # For now, we'll use a simplified approach

        return prompt

    def run_ab_test(self, setting_name: str, value_a: Any, value_b: Any,
                    base_seed: int = None) -> Dict:
        """
        Run an A/B test for a specific setting.
        Returns the user's preference and records the test.
        """
        if base_seed is None:
            base_seed = random.randint(0, 2**32 - 1)

        print(f"\n  {styled('🔬 A/B TEST:', Style.CYAN, Style.BOLD)} {setting_name}")
        print(f"     A: {setting_name} = {value_a}")
        print(f"     B: {setting_name} = {value_b}")
        print(f"     Seed: {base_seed} (same for both)")
        print_divider()

        # Create variants with same seed
        variant_a = self._create_variant(setting_name, value_a)
        variant_a = self._create_variant("seed", base_seed)

        variant_b = self._create_variant(setting_name, value_b)
        variant_b = self._create_variant("seed", base_seed)

        # Save as experimental files
        base_path = self.workflow_path
        path_a = base_path.replace(".json", "_test_A.json")
        path_b = base_path.replace(".json", "_test_B.json")

        with open(path_a, 'w') as f:
            json.dump(variant_a, f, indent=2)
        with open(path_b, 'w') as f:
            json.dump(variant_b, f, indent=2)

        print(f"\n  {styled('📄', Style.CYAN)} Created test files:")
        print(f"     A: {path_a}")
        print(f"     B: {path_b}")

        print(f"\n  {styled('⏳', Style.YELLOW)} Please:")
        print(f"     1. Open both files in ComfyUI")
        print(f"     2. Generate image A, then image B")
        print(f"     3. Compare the results")

        # Get user preference
        while True:
            choice = input(f"\n  {styled('▶', Style.CYAN)} Which do you prefer? (A/B/Same/Skip): ").strip().upper()

            if choice in ['A', 'B', 'SAME', 'S', 'SKIP']:
                break
            print(f"  {styled('Please enter A, B, Same, or Skip', Style.DIM)}")

        # Record result
        result = {
            "setting": setting_name,
            "value_a": value_a,
            "value_b": value_b,
            "seed": base_seed,
            "preference": choice,
            "timestamp": datetime.now().isoformat(),
        }

        self.test_history.append(result)

        # Clean up test files
        try:
            os.remove(path_a)
            os.remove(path_b)
        except:
            pass

        return result

    def binary_search_optimal(self, setting_name: str, min_val: float, max_val: float,
                              precision: float = None, max_iterations: int = 5) -> float:
        """
        Use binary search with A/B testing to find optimal value.
        Like an optician: "Better with lens 1... or lens 2?"
        """
        if precision is None:
            precision = (max_val - min_val) / 20

        print(f"\n  {styled('🎯 OPTICIAN MODE:', Style.MAGENTA, Style.BOLD)} Finding optimal {setting_name}")
        print(f"     Range: {min_val} to {max_val}")
        print(f"     Precision: {precision}")
        print(f"     Max iterations: {max_iterations}")
        print_divider()

        low = min_val
        high = max_val
        current_best = (low + high) / 2

        for iteration in range(max_iterations):
            print(f"\n  {styled(f'Iteration {iteration + 1}/{max_iterations}', Style.CYAN)}")

            # Test current vs higher
            mid = (low + high) / 2
            test_low = mid - (high - low) / 4
            test_high = mid + (high - low) / 4

            result = self.run_ab_test(setting_name, round(test_low, 2), round(test_high, 2))

            if result["preference"] == "A":
                high = mid
                current_best = test_low
                print(f"  {styled('→', Style.GREEN)} Prefer lower value, narrowing range to {low:.2f}-{mid:.2f}")
            elif result["preference"] == "B":
                low = mid
                current_best = test_high
                print(f"  {styled('→', Style.GREEN)} Prefer higher value, narrowing range to {mid:.2f}-{high:.2f}")
            elif result["preference"] in ["SAME", "S"]:
                current_best = mid
                print(f"  {styled('→', Style.YELLOW)} No preference, keeping middle value {mid:.2f}")
                # Optionally could break early
            else:
                print(f"  {styled('→', Style.YELLOW)} Skipped, keeping current best")
                continue

            # Check if we've reached desired precision
            if high - low <= precision * 2:
                print(f"\n  {styled('✓', Style.GREEN)} Reached desired precision!")
                break

        final_value = round(current_best, 2)
        print(f"\n  {styled('🎯 OPTIMAL VALUE FOUND:', Style.GREEN, Style.BOLD)} {setting_name} = {final_value}")

        return final_value

    def test_discrete_options(self, setting_name: str, options: List[str]) -> str:
        """
        A/B test through discrete options (like sampler names).
        Uses tournament-style elimination.
        """
        print(f"\n  {styled('🏆 TOURNAMENT MODE:', Style.MAGENTA, Style.BOLD)} Finding best {setting_name}")
        print(f"     Options: {', '.join(options)}")
        print_divider()

        remaining = list(options)

        while len(remaining) > 1:
            print(f"\n  {styled(f'Round: {len(options) - len(remaining) + 1}', Style.CYAN)} ({len(remaining)} remaining)")

            # Pair up options
            random.shuffle(remaining)
            next_round = []

            i = 0
            while i < len(remaining):
                if i + 1 < len(remaining):
                    # A/B test pair
                    option_a = remaining[i]
                    option_b = remaining[i + 1]

                    result = self.run_ab_test(setting_name, option_a, option_b)

                    if result["preference"] == "A":
                        next_round.append(option_a)
                    elif result["preference"] == "B":
                        next_round.append(option_b)
                    else:
                        # Same or skip - keep first option
                        next_round.append(option_a)

                    i += 2
                else:
                    # Odd one out - advances automatically
                    next_round.append(remaining[i])
                    i += 1

            remaining = next_round

        winner = remaining[0] if remaining else options[0]
        print(f"\n  {styled('🏆 WINNER:', Style.GREEN, Style.BOLD)} {setting_name} = {winner}")

        return winner

# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROMPT GENERATOR FOR NON-VISUAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

class SettingsLLMPromptGenerator:
    """
    Generates prompts for LLMs to help with settings that can't be
    visually A/B tested (like prompt engineering, negative prompts, etc.)
    """

    @staticmethod
    def generate_prompt_optimization_request(
        current_positive: str,
        current_negative: str,
        model_name: str,
        loras: List[str],
        style_goal: str,
        harvested_examples: List[Dict] = None
    ) -> str:
        """Generate an LLM prompt for optimizing prompts."""

        prompt_parts = []

        prompt_parts.append("=" * 70)
        prompt_parts.append("STABLE DIFFUSION PROMPT OPTIMIZATION REQUEST")
        prompt_parts.append("=" * 70)
        prompt_parts.append("")

        prompt_parts.append("## CURRENT SETUP")
        prompt_parts.append(f"**Model:** {model_name}")
        if loras:
            prompt_parts.append(f"**LoRAs:** {', '.join(loras)}")
        prompt_parts.append("")

        prompt_parts.append("## CURRENT PROMPTS")
        prompt_parts.append("**Positive Prompt:**")
        prompt_parts.append(f"```")
        prompt_parts.append(current_positive or "(empty)")
        prompt_parts.append(f"```")
        prompt_parts.append("")
        prompt_parts.append("**Negative Prompt:**")
        prompt_parts.append(f"```")
        prompt_parts.append(current_negative or "(empty)")
        prompt_parts.append(f"```")
        prompt_parts.append("")

        if style_goal:
            prompt_parts.append("## STYLE GOAL")
            prompt_parts.append(style_goal)
            prompt_parts.append("")

        if harvested_examples:
            prompt_parts.append("## EXAMPLE PROMPTS FROM SIMILAR GENERATIONS")
            for i, example in enumerate(harvested_examples[:5]):
                if example.get("positive_prompt"):
                    prompt_parts.append(f"\n**Example {i+1} Positive:**")
                    prompt_parts.append(f"```")
                    prompt_parts.append(example["positive_prompt"][:500])
                    prompt_parts.append(f"```")
                if example.get("negative_prompt"):
                    prompt_parts.append(f"**Example {i+1} Negative:**")
                    prompt_parts.append(f"```")
                    prompt_parts.append(example["negative_prompt"][:300])
                    prompt_parts.append(f"```")
            prompt_parts.append("")

        prompt_parts.append("## REQUEST")
        prompt_parts.append("Please analyze my current prompts and suggest improvements.")
        prompt_parts.append("")
        prompt_parts.append("Provide:")
        prompt_parts.append("1. An improved positive prompt (with explanation)")
        prompt_parts.append("2. An improved negative prompt (with explanation)")
        prompt_parts.append("3. Any recommended prompt techniques for this model/LoRA combination")
        prompt_parts.append("4. Tips for achieving the style goal")
        prompt_parts.append("")
        prompt_parts.append("Consider:")
        prompt_parts.append("- Prompt weighting syntax: (word:1.2) for emphasis")
        prompt_parts.append("- Token efficiency (staying under ~75 tokens if possible)")
        prompt_parts.append("- Model-specific keywords that work well")
        prompt_parts.append("- Negative prompt best practices for this model type")
        prompt_parts.append("=" * 70)

        return "\n".join(prompt_parts)

    @staticmethod
    def generate_settings_analysis_request(
        current_settings: Dict,
        test_results: List[Dict],
        model_type: str,
        optimization_goal: str
    ) -> str:
        """Generate an LLM prompt for analyzing optimal settings."""

        prompt_parts = []

        prompt_parts.append("=" * 70)
        prompt_parts.append("COMFYUI SETTINGS ANALYSIS REQUEST")
        prompt_parts.append("=" * 70)
        prompt_parts.append("")

        prompt_parts.append("## MODEL INFORMATION")
        prompt_parts.append(f"**Model Type:** {model_type}")
        prompt_parts.append("")

        prompt_parts.append("## CURRENT SETTINGS")
        for key, value in current_settings.items():
            if value is not None:
                prompt_parts.append(f"• {key}: {value}")
        prompt_parts.append("")

        if test_results:
            prompt_parts.append("## A/B TEST RESULTS")
            prompt_parts.append("The user performed optician-style A/B tests with these results:")
            prompt_parts.append("")
            for result in test_results[-10:]:  # Last 10 tests
                prompt_parts.append(f"• {result['setting']}: {result['value_a']} vs {result['value_b']}")
                prompt_parts.append(f"  Preferred: {result['preference']}")
            prompt_parts.append("")

        if optimization_goal:
            prompt_parts.append("## OPTIMIZATION GOAL")
            prompt_parts.append(optimization_goal)
            prompt_parts.append("")

        prompt_parts.append("## REQUEST")
        prompt_parts.append("Based on the test results and current settings, please provide:")
        prompt_parts.append("")
        prompt_parts.append("1. **Analysis** of the user's preferences (what patterns do you see?)")
        prompt_parts.append("2. **Recommended final settings** for their goal")
        prompt_parts.append("3. **Explanation** of why these settings work together")
        prompt_parts.append("4. **Additional tips** for this model type")
        prompt_parts.append("")
        prompt_parts.append("Focus on settings that can't be easily A/B tested:")
        prompt_parts.append("- Clip skip values")
        prompt_parts.append("- VAE selection")
        prompt_parts.append("- Prompt structure and syntax")
        prompt_parts.append("- Model-specific quirks")
        prompt_parts.append("=" * 70)

        return "\n".join(prompt_parts)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class LoRAOptimizer:
    """Main application for LoRA/Model optimization."""

    def __init__(self):
        self.monitor = ComfyUIMonitor(COMFY_URL)
        self.harvester = MetadataHarvester()
        self.tester: Optional[OpticianTester] = None
        self.workflow_path: Optional[str] = None
        self.optimized_settings: Dict = {}
        self.style_goal: str = ""

    def run(self):
        """Main application loop."""
        clear_screen()
        print_header(
            "LORA OPTIMIZER",
            "Optician-Style Settings Tuner",
            "0.1"
        )

        # Setup workflow
        self.setup_workflow()
        if not self.workflow_path:
            print(f"\n  {styled('Goodbye!', Style.DIM)}")
            return

        # Initialize tester
        self.tester = OpticianTester(self.monitor, self.workflow_path)

        # Main menu loop
        while True:
            self.show_main_menu()
            choice = input(f"\n  {styled('▶', Style.CYAN)} Enter choice: ").strip().lower()

            if choice == 'q':
                print(f"\n  {styled('Goodbye!', Style.DIM)}")
                break
            elif choice == '1':
                self.harvest_metadata_menu()
            elif choice == '2':
                self.quick_ab_test_menu()
            elif choice == '3':
                self.optician_mode_menu()
            elif choice == '4':
                self.view_recommendations()
            elif choice == '5':
                self.generate_llm_prompt_menu()
            elif choice == '6':
                self.apply_settings()
            elif choice == '7':
                self.set_style_goal()
            elif choice == 't':
                self.setup_workflow()
                if self.workflow_path:
                    self.tester = OpticianTester(self.monitor, self.workflow_path)
            else:
                print(f"  {styled('Invalid choice', Style.YELLOW)}")

    def setup_workflow(self):
        """Setup target workflow."""
        print_box("Target Workflow", [
            "Enter the workflow JSON file you want to optimize.",
        ], Style.BLUE, icon="📁")

        filepath = input(f"\n  {styled('▶', Style.CYAN)} Workflow path (or 'q' to quit): ").strip()

        if filepath.lower() == 'q':
            self.workflow_path = None
            return

        if not os.path.exists(filepath):
            print(f"  {styled('✗', Style.RED)} File not found: {filepath}")
            return self.setup_workflow()

        self.workflow_path = filepath
        print(f"  {styled('✓', Style.GREEN)} Workflow loaded: {styled(filepath, Style.BOLD)}")

    def show_main_menu(self):
        """Display main menu."""
        print()
        print_divider("═", Style.CYAN)
        print(f"  {styled('🎯 LORA OPTIMIZER', Style.BOLD, Style.WHITE)}")
        print(f"  Workflow: {styled(self.workflow_path or 'Not set', Style.BLUE)}")
        if self.style_goal:
            print(f"  Goal: {styled(self.style_goal[:40], Style.MAGENTA)}")
        print_divider()

        harvested = len(self.harvester.harvested_settings)
        tests = len(self.tester.test_history) if self.tester else 0

        menu_items = [
            ("1", "📚 Harvest Metadata", f"{harvested} examples collected"),
            ("2", "🔬 Quick A/B Test", "Test specific setting"),
            ("3", "👁️ Optician Mode", "Binary search for optimal values"),
            ("4", "📊 View Recommendations", "From harvested data"),
            ("5", "🤖 Generate LLM Prompt", "For prompt optimization"),
            ("6", "✨ Apply Optimized Settings", "Update workflow"),
            ("7", "🎯 Set Style Goal", self.style_goal[:20] if self.style_goal else "Not set"),
            ("T", "Change Workflow", ""),
            ("Q", "Quit", ""),
        ]

        for key, label, hint in menu_items:
            hint_str = styled(f" ({hint})", Style.DIM) if hint else ""
            print(f"    {styled(key, Style.CYAN, Style.BOLD)}  {label}{hint_str}")

    def harvest_metadata_menu(self):
        """Harvest metadata from files."""
        print_box("📚 Harvest Metadata", [
            "Extract settings from existing generations.",
            "Supports: PNG files (with embedded metadata), JSON workflows",
        ], Style.BLUE, icon="")

        print(f"\n  {styled('Options:', Style.DIM)}")
        print(f"    {styled('1', Style.CYAN)} Harvest from directory")
        print(f"    {styled('2', Style.CYAN)} Harvest from single file")
        print(f"    {styled('B', Style.CYAN)} Back")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Choice: ").strip().lower()

        if choice == '1':
            directory = input(f"  {styled('▶', Style.CYAN)} Directory path: ").strip()
            if os.path.isdir(directory):
                print(f"\n  {styled('⏳', Style.YELLOW)} Scanning directory...")
                results = self.harvester.harvest_from_directory(directory)
                print(f"  {styled('✓', Style.GREEN)} Found {len(results)} files with metadata")

                if results:
                    self._show_harvest_summary(results)
            else:
                print(f"  {styled('✗', Style.RED)} Directory not found")

        elif choice == '2':
            filepath = input(f"  {styled('▶', Style.CYAN)} File path: ").strip()
            if filepath.endswith('.png'):
                result = self.harvester.harvest_from_png(filepath)
            elif filepath.endswith('.json'):
                result = self.harvester.harvest_from_workflow(filepath)
            else:
                print(f"  {styled('✗', Style.RED)} Unsupported file type")
                return

            if result:
                self.harvester.harvested_settings.append(result)
                print(f"  {styled('✓', Style.GREEN)} Extracted metadata:")
                self._show_settings(result)
            else:
                print(f"  {styled('⚠', Style.YELLOW)} No metadata found")

    def _show_harvest_summary(self, results: List[Dict]):
        """Show summary of harvested results."""
        best_practices = self.harvester.get_best_practices()

        print_box("Harvest Summary", [], Style.GREEN, icon="📊")

        for key in ["cfg", "steps", "denoise"]:
            data = best_practices.get(key, {})
            if data.get("recommended"):
                print(f"  {key}: avg={data['recommended']:.2f} (range: {data.get('min', '?')}-{data.get('max', '?')})")

        for key in ["sampler", "scheduler"]:
            data = best_practices.get(key, {})
            if data.get("recommended"):
                print(f"  {key}: most common = {data['recommended']}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def _show_settings(self, settings: Dict):
        """Display settings nicely."""
        for key, value in settings.items():
            if value is not None and key not in ["source", "source_type"]:
                if isinstance(value, list):
                    if value:
                        print(f"    {key}: {', '.join(str(v) for v in value[:3])}")
                elif isinstance(value, str) and len(value) > 50:
                    print(f"    {key}: {value[:50]}...")
                else:
                    print(f"    {key}: {value}")

    def quick_ab_test_menu(self):
        """Quick A/B test for a specific setting."""
        if not self.tester:
            print(f"  {styled('⚠', Style.YELLOW)} Load a workflow first")
            return

        print_box("🔬 Quick A/B Test", [
            "Compare two values for a setting.",
        ], Style.MAGENTA, icon="")

        settings = [
            ("cfg", "CFG Scale", 1.0, 15.0),
            ("steps", "Steps", 10, 50),
            ("denoise", "Denoise", 0.3, 1.0),
        ]

        print(f"\n  {styled('Select setting:', Style.DIM)}")
        for i, (key, name, low, high) in enumerate(settings):
            print(f"    {styled(str(i+1), Style.CYAN)} {name} ({low}-{high})")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Setting #: ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(settings):
                key, name, low, high = settings[idx]

                val_a = input(f"  {styled('▶', Style.CYAN)} Value A: ").strip()
                val_b = input(f"  {styled('▶', Style.CYAN)} Value B: ").strip()

                val_a = float(val_a) if '.' in val_a else int(val_a)
                val_b = float(val_b) if '.' in val_b else int(val_b)

                self.tester.run_ab_test(key, val_a, val_b)
        except (ValueError, IndexError):
            print(f"  {styled('Invalid input', Style.YELLOW)}")

    def optician_mode_menu(self):
        """Optician mode - binary search for optimal values."""
        if not self.tester:
            print(f"  {styled('⚠', Style.YELLOW)} Load a workflow first")
            return

        print_box("👁️ Optician Mode", [
            "'Better A... or B?' binary search to find optimal values.",
            "Works like an eye exam!",
        ], Style.MAGENTA, icon="")

        print(f"\n  {styled('Select what to optimize:', Style.DIM)}")
        print(f"    {styled('1', Style.CYAN)} CFG Scale (continuous)")
        print(f"    {styled('2', Style.CYAN)} Steps (continuous)")
        print(f"    {styled('3', Style.CYAN)} Denoise (continuous)")
        print(f"    {styled('4', Style.CYAN)} Sampler (tournament)")
        print(f"    {styled('5', Style.CYAN)} Scheduler (tournament)")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Choice: ").strip()

        if choice == '1':
            result = self.tester.binary_search_optimal("cfg", 1.0, 15.0, precision=0.5)
            self.optimized_settings["cfg"] = result
        elif choice == '2':
            result = self.tester.binary_search_optimal("steps", 10, 50, precision=5)
            self.optimized_settings["steps"] = int(result)
        elif choice == '3':
            result = self.tester.binary_search_optimal("denoise", 0.3, 1.0, precision=0.05)
            self.optimized_settings["denoise"] = result
        elif choice == '4':
            result = self.tester.test_discrete_options("sampler", DEFAULT_RANGES["sampler"])
            self.optimized_settings["sampler"] = result
        elif choice == '5':
            result = self.tester.test_discrete_options("scheduler", DEFAULT_RANGES["scheduler"])
            self.optimized_settings["scheduler"] = result

    def view_recommendations(self):
        """View recommendations from harvested data."""
        best_practices = self.harvester.get_best_practices()

        if not best_practices or not any(v.get("recommended") for v in best_practices.values() if isinstance(v, dict)):
            print(f"\n  {styled('⚠', Style.YELLOW)} No recommendations yet. Harvest some metadata first!")
            return

        print_box("📊 Recommendations", [
            "Based on harvested metadata from example generations.",
        ], Style.GREEN, icon="")

        print(f"\n  {styled('From harvested examples:', Style.BOLD)}")

        for key in ["cfg", "steps", "denoise"]:
            data = best_practices.get(key, {})
            if data.get("recommended"):
                rec = data['recommended']
                min_v = data.get('min', '?')
                max_v = data.get('max', '?')
                print(f"    {key}: {styled(f'{rec:.2f}', Style.CYAN)} (range: {min_v}-{max_v})")

        for key in ["sampler", "scheduler"]:
            data = best_practices.get(key, {})
            if data.get("recommended"):
                print(f"    {key}: {styled(data['recommended'], Style.CYAN)}")
                if data.get("distribution"):
                    dist = data["distribution"]
                    print(f"           {dict(sorted(dist.items(), key=lambda x: -x[1])[:3])}")

        # Show optimized settings from A/B tests
        if self.optimized_settings:
            print(f"\n  {styled('From A/B testing:', Style.BOLD)}")
            for key, value in self.optimized_settings.items():
                print(f"    {key}: {styled(str(value), Style.GREEN)}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def generate_llm_prompt_menu(self):
        """Generate LLM prompt for non-visual optimization."""
        print_box("🤖 Generate LLM Prompt", [
            "Get help from an LLM for settings that can't be A/B tested.",
        ], Style.BLUE, icon="")

        print(f"\n  {styled('Options:', Style.DIM)}")
        print(f"    {styled('1', Style.CYAN)} Prompt optimization (positive/negative)")
        print(f"    {styled('2', Style.CYAN)} Settings analysis (based on test results)")

        choice = input(f"\n  {styled('▶', Style.CYAN)} Choice: ").strip()

        if choice == '1':
            current_pos = input(f"  {styled('▶', Style.CYAN)} Current positive prompt: ").strip()
            current_neg = input(f"  {styled('▶', Style.CYAN)} Current negative prompt: ").strip()
            model_name = input(f"  {styled('▶', Style.CYAN)} Model name: ").strip()

            prompt = SettingsLLMPromptGenerator.generate_prompt_optimization_request(
                current_pos,
                current_neg,
                model_name,
                [],
                self.style_goal,
                self.harvester.harvested_settings[:5]
            )
        elif choice == '2':
            model_type = input(f"  {styled('▶', Style.CYAN)} Model type (SD15/SDXL/Flux): ").strip()

            prompt = SettingsLLMPromptGenerator.generate_settings_analysis_request(
                self.optimized_settings,
                self.tester.test_history if self.tester else [],
                model_type,
                self.style_goal
            )
        else:
            return

        # Display and copy
        print()
        print(styled("─" * 70, Style.GRAY))
        print(styled("COPY THE FOLLOWING PROMPT:", Style.BOLD, Style.GREEN))
        print(styled("─" * 70, Style.GRAY))
        print()
        print(prompt)
        print()
        print(styled("─" * 70, Style.GRAY))

        if Clipboard.copy(prompt):
            print(f"\n  {styled('📋 Copied to clipboard!', Style.GREEN)}")

        input(f"\n  {styled('Press Enter to continue...', Style.DIM)}")

    def apply_settings(self):
        """Apply optimized settings to workflow."""
        if not self.optimized_settings:
            print(f"\n  {styled('⚠', Style.YELLOW)} No optimized settings yet. Run some tests first!")
            return

        print_box("✨ Apply Settings", [
            "Apply your optimized settings to the workflow.",
        ], Style.GREEN, icon="")

        print(f"\n  {styled('Settings to apply:', Style.BOLD)}")
        for key, value in self.optimized_settings.items():
            print(f"    {key}: {value}")

        confirm = input(f"\n  {styled('▶', Style.CYAN)} Apply these settings? (y/n): ").strip().lower()

        if confirm == 'y':
            # Create optimized workflow
            optimized = copy.deepcopy(self.tester.workflow_content)

            for key, value in self.optimized_settings.items():
                optimized = self.tester._create_variant(key, value)

            # Save
            output_path = self.workflow_path.replace(".json", "_optimized.json")
            with open(output_path, 'w') as f:
                json.dump(optimized, f, indent=2)

            print(f"  {styled('✓', Style.GREEN)} Saved: {styled(output_path, Style.BOLD)}")

    def set_style_goal(self):
        """Set the style/optimization goal."""
        print_box("🎯 Set Style Goal", [
            "Describe what you're trying to achieve.",
            "This helps with LLM prompt generation.",
        ], Style.BLUE, icon="")

        goal = input(f"\n  {styled('▶', Style.CYAN)} Your goal: ").strip()
        if goal:
            self.style_goal = goal
            print(f"  {styled('✓', Style.GREEN)} Goal set!")

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        app = LoRAOptimizer()
        app.run()
    except KeyboardInterrupt:
        print(f"\n\n  {styled('Interrupted. Goodbye!', Style.DIM)}")
        sys.exit(0)

if __name__ == "__main__":
    main()
