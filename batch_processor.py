"""
Batch Processing Utilities for Performance Lab v2.1
Process multiple workflow variations from CSV files
"""

import csv
import json
import os
from typing import List, Dict, Any


def parse_batch_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Parse a CSV file with batch workflow parameters.

    CSV Format:
    prompt,negative_prompt,steps,cfg,resolution,seed
    "a cat",ugly,30,7.0,1024,12345
    "a dog",blurry,25,6.5,768,67890

    Returns list of dicts with parsed parameters.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    batch_configs = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            config = {}

            # Parse each column
            for key, value in row.items():
                if not key:  # Skip empty columns
                    continue

                key = key.strip().lower()
                value = value.strip()

                # Type conversion
                if key in ['steps', 'seed', 'resolution', 'width', 'height', 'batch_size']:
                    try:
                        config[key] = int(value)
                    except ValueError:
                        config[key] = 0

                elif key in ['cfg', 'cfg_scale', 'denoise', 'strength']:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        config[key] = 0.0

                elif key in ['enabled', 'use_tiling', 'upscale']:
                    config[key] = value.lower() in ['true', '1', 'yes', 'on']

                else:
                    config[key] = value

            config['_row'] = row_num
            batch_configs.append(config)

    return batch_configs


def create_batch_variations(base_params: Dict[str, Any], variations: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Create batch variations from base parameters.

    Example:
        base = {"prompt": "a cat", "steps": 30}
        variations = {"cfg": [6.0, 7.0, 8.0], "resolution": [512, 768, 1024]}
        Returns 9 combinations (3 cfg × 3 resolutions)
    """
    import itertools

    # Get variation keys and their values
    var_keys = list(variations.keys())
    var_values = [variations[k] for k in var_keys]

    # Generate all combinations
    batch_configs = []
    for combo in itertools.product(*var_values):
        config = base_params.copy()
        for key, value in zip(var_keys, combo):
            config[key] = value
        batch_configs.append(config)

    return batch_configs


def save_batch_results(results: List[Dict[str, Any]], output_path: str):
    """Save batch processing results to CSV."""
    if not results:
        return

    # Get all unique keys
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    fieldnames = sorted(all_keys)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def generate_batch_csv_template(output_path: str, model_type: str = "image"):
    """Generate a template CSV for batch processing."""

    templates = {
        "image": {
            "headers": ["prompt", "negative_prompt", "steps", "cfg", "resolution", "seed", "sampler"],
            "example": [
                "a cat in a garden",
                "ugly, blurry, low quality",
                "30",
                "7.0",
                "1024",
                "12345",
                "dpmpp_2m"
            ]
        },
        "video": {
            "headers": ["prompt", "negative_prompt", "frames", "fps", "motion_scale", "seed"],
            "example": [
                "a cat walking in a garden",
                "static, frozen, low quality",
                "60",
                "24",
                "127",
                "12345"
            ]
        },
        "llm": {
            "headers": ["system_prompt", "user_prompt", "temperature", "max_tokens", "top_p"],
            "example": [
                "You are a helpful assistant",
                "Explain quantum computing",
                "0.7",
                "500",
                "0.9"
            ]
        }
    }

    template = templates.get(model_type, templates["image"])

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(template["headers"])
        writer.writerow(template["example"])


class BatchQueue:
    """Manage a queue of batch jobs."""

    def __init__(self):
        self.queue = []
        self.completed = []
        self.failed = []

    def add(self, config: Dict[str, Any]):
        """Add a configuration to the queue."""
        self.queue.append(config)

    def add_many(self, configs: List[Dict[str, Any]]):
        """Add multiple configurations."""
        self.queue.extend(configs)

    def get_next(self) -> Dict[str, Any]:
        """Get the next item from the queue."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def mark_completed(self, config: Dict[str, Any], result: Dict[str, Any]):
        """Mark a job as completed."""
        self.completed.append({**config, **result})

    def mark_failed(self, config: Dict[str, Any], error: str):
        """Mark a job as failed."""
        self.failed.append({**config, "error": error})

    def get_status(self) -> Dict[str, int]:
        """Get queue status."""
        return {
            "queued": len(self.queue),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "total": len(self.queue) + len(self.completed) + len(self.failed)
        }

    def export_results(self, output_path: str):
        """Export all results to CSV."""
        save_batch_results(self.completed + self.failed, output_path)
