"""
GPU Benchmark Database for Performance Lab v2.1
Provides performance hints based on GPU capabilities
"""

GPU_BENCHMARKS = {
    # NVIDIA RTX 40-series
    "RTX 4090": {
        "vram_gb": 24,
        "flux_steps_per_sec": 1.2,
        "sdxl_steps_per_sec": 2.5,
        "sd15_steps_per_sec": 4.0,
        "video_frames_per_sec": 0.8,
        "tier": "flagship",
        "recommended_batch": 4,
        "max_resolution": 2048,
    },
    "RTX 4080": {
        "vram_gb": 16,
        "flux_steps_per_sec": 0.9,
        "sdxl_steps_per_sec": 2.0,
        "sd15_steps_per_sec": 3.5,
        "video_frames_per_sec": 0.6,
        "tier": "high",
        "recommended_batch": 3,
        "max_resolution": 1536,
    },
    "RTX 4070 Ti": {
        "vram_gb": 12,
        "flux_steps_per_sec": 0.7,
        "sdxl_steps_per_sec": 1.6,
        "sd15_steps_per_sec": 3.0,
        "video_frames_per_sec": 0.4,
        "tier": "high",
        "recommended_batch": 2,
        "max_resolution": 1536,
    },
    "RTX 4060 Ti": {
        "vram_gb": 16,
        "flux_steps_per_sec": 0.5,
        "sdxl_steps_per_sec": 1.2,
        "sd15_steps_per_sec": 2.2,
        "video_frames_per_sec": 0.3,
        "tier": "mid",
        "recommended_batch": 2,
        "max_resolution": 1024,
    },

    # NVIDIA RTX 30-series
    "RTX 3090": {
        "vram_gb": 24,
        "flux_steps_per_sec": 0.8,
        "sdxl_steps_per_sec": 1.8,
        "sd15_steps_per_sec": 3.2,
        "video_frames_per_sec": 0.5,
        "tier": "high",
        "recommended_batch": 3,
        "max_resolution": 1536,
    },
    "RTX 3080": {
        "vram_gb": 10,
        "flux_steps_per_sec": 0.6,
        "sdxl_steps_per_sec": 1.4,
        "sd15_steps_per_sec": 2.8,
        "video_frames_per_sec": 0.4,
        "tier": "mid",
        "recommended_batch": 2,
        "max_resolution": 1024,
    },
    "RTX 3060": {
        "vram_gb": 12,
        "flux_steps_per_sec": 0.4,
        "sdxl_steps_per_sec": 1.0,
        "sd15_steps_per_sec": 2.0,
        "video_frames_per_sec": 0.2,
        "tier": "mid",
        "recommended_batch": 1,
        "max_resolution": 768,
    },

    # NVIDIA Professional
    "A100": {
        "vram_gb": 40,
        "flux_steps_per_sec": 1.5,
        "sdxl_steps_per_sec": 3.2,
        "sd15_steps_per_sec": 5.0,
        "video_frames_per_sec": 1.2,
        "tier": "datacenter",
        "recommended_batch": 8,
        "max_resolution": 2048,
    },
    "A5000": {
        "vram_gb": 24,
        "flux_steps_per_sec": 0.9,
        "sdxl_steps_per_sec": 2.0,
        "sd15_steps_per_sec": 3.5,
        "video_frames_per_sec": 0.6,
        "tier": "professional",
        "recommended_batch": 4,
        "max_resolution": 1536,
    },
    "H100": {
        "vram_gb": 80,
        "flux_steps_per_sec": 2.5,
        "sdxl_steps_per_sec": 5.0,
        "sd15_steps_per_sec": 8.0,
        "video_frames_per_sec": 2.0,
        "tier": "datacenter",
        "recommended_batch": 16,
        "max_resolution": 4096,
    },

    # AMD
    "RX 7900 XTX": {
        "vram_gb": 24,
        "flux_steps_per_sec": 0.7,
        "sdxl_steps_per_sec": 1.5,
        "sd15_steps_per_sec": 2.8,
        "video_frames_per_sec": 0.4,
        "tier": "high",
        "recommended_batch": 3,
        "max_resolution": 1536,
    },
}


def get_gpu_benchmark(gpu_name: str) -> dict:
    """Get benchmark data for a GPU, with fuzzy matching."""
    # Direct match
    if gpu_name in GPU_BENCHMARKS:
        return GPU_BENCHMARKS[gpu_name]

    # Fuzzy match
    gpu_upper = gpu_name.upper()
    for benchmark_name, data in GPU_BENCHMARKS.items():
        if benchmark_name.upper() in gpu_upper or gpu_upper in benchmark_name.upper():
            return data

    # Default fallback
    return {
        "vram_gb": 0,
        "flux_steps_per_sec": 0.3,
        "sdxl_steps_per_sec": 0.8,
        "sd15_steps_per_sec": 1.5,
        "video_frames_per_sec": 0.1,
        "tier": "unknown",
        "recommended_batch": 1,
        "max_resolution": 512,
    }


def estimate_generation_time(gpu_name: str, model_type: str, steps: int, resolution: int, batch_size: int = 1) -> float:
    """Estimate generation time in seconds."""
    benchmark = get_gpu_benchmark(gpu_name)

    # Get steps per second for model type
    speed_key = f"{model_type}_steps_per_sec"
    steps_per_sec = benchmark.get(speed_key, 1.0)

    # Calculate base time
    base_time = steps / steps_per_sec if steps_per_sec > 0 else steps

    # Resolution multiplier (quadratic scaling)
    base_res = 512 if "sd15" in model_type else 1024
    res_multiplier = (resolution / base_res) ** 2

    # Batch multiplier (near-linear)
    batch_multiplier = batch_size * 0.9  # Slight efficiency gain

    return base_time * res_multiplier * batch_multiplier


def compare_gpus(current_gpu: str, steps: int, model_type: str = "sdxl") -> list:
    """Compare current GPU to alternatives."""
    current_bench = get_gpu_benchmark(current_gpu)
    current_speed = current_bench.get(f"{model_type}_steps_per_sec", 1.0)
    current_time = steps / current_speed if current_speed > 0 else steps

    comparisons = []
    for gpu_name, bench in GPU_BENCHMARKS.items():
        if gpu_name == current_gpu:
            continue

        other_speed = bench.get(f"{model_type}_steps_per_sec", 1.0)
        other_time = steps / other_speed if other_speed > 0 else steps

        speedup = (current_time - other_time) / current_time * 100 if current_time > 0 else 0

        comparisons.append({
            "gpu": gpu_name,
            "tier": bench["tier"],
            "vram": bench["vram_gb"],
            "time": other_time,
            "speedup_percent": speedup,
        })

    # Sort by speedup
    comparisons.sort(key=lambda x: x["speedup_percent"], reverse=True)
    return comparisons[:5]  # Top 5
