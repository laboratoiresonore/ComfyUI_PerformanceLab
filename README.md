# ComfyUI Performance Lab v0.2 - Model Tuner Edition

**Iterative Workflow Optimization with Smart Model Detection & One-Click Tuning**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║          ⚡ COMFYUI PERFORMANCE LAB v0.2 - MODEL TUNER EDITION ⚡            ║
║       Auto-Detect Models • Smart Optimization • LoRA Tuning • More!         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

![Performance Lab](https://img.shields.io/badge/ComfyUI-Performance%20Lab-blue) ![Python 3.7+](https://img.shields.io/badge/Python-3.7+-green) ![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen) ![Version](https://img.shields.io/badge/Version-0.2.0-orange)

## What's New in v0.2

| Feature | Description |
|---------|-------------|
| 🎛️ **Model Tuner** | Auto-detect SD1.5, SDXL, Flux, SD3 and apply optimal settings |
| 🔍 **Smart Model Detection** | Automatically identifies model type from workflow |
| 📊 **Model-Specific Presets** | Optimal settings for each model type |
| 🎨 **LoRA Strength Tuning** | Recommendations and A/B testing for LoRA values |
| 🎯 **Sampler Recommendations** | Best samplers/schedulers per model and use case |
| 📦 **One-Step Installer** | `python install.py` - works anywhere |

### Previous Features (v0.1)

| Feature | Description |
|---------|-------------|
| ⚡ **Quick Actions** | One-key optimizations (bypass upscalers, cap resolution, etc.) |
| 📊 **Benchmark Mode** | Multiple runs for reliable metrics with statistics |
| 🧠 **Smart Suggestions** | AI-free workflow analysis & recommendations |
| ⚙️ **Presets System** | 8GB VRAM, Speed Test, Quality presets |
| 📈 **Progress Dashboard** | Visual history of all optimizations |
| 🔄 **Workflow Diff** | See exactly what changed |
| 📋 **Multi-Platform Clipboard** | Windows, macOS, Linux support |
| 💾 **Config Persistence** | Save settings between sessions |
| 🔧 **Built-in Mods** | Common optimizations included |

## Overview

Performance Lab creates a **human-in-the-loop optimization cycle** where you:

1. **Apply** a mod or quick action to your ComfyUI workflow
2. **Test** it by running ComfyUI (monitors automatically)
3. **Review** rich metrics (timing, VRAM, errors)
4. **Decide** to keep or revert changes
5. **Repeat** until your workflow is optimized

You can use the built-in smart suggestions for instant optimizations, or generate prompts for external LLMs (Claude, GPT-4, Gemini, Llama) for deeper analysis.

## Installation

### One-Step Install (Recommended)

```bash
# Clone the repo
git clone https://github.com/laboratoiresonore/ComfyUI_PerformanceLab.git
cd ComfyUI_PerformanceLab

# Run the installer
python install.py
```

The installer will:
- Auto-detect your ComfyUI installation
- Set up all necessary files
- Create a launcher script
- Offer to run Performance Lab immediately

### Manual Install

```bash
# Clone or copy the files to your ComfyUI directory
cd /path/to/ComfyUI
git clone https://github.com/laboratoiresonore/ComfyUI_PerformanceLab.git Workflowmods

# Run directly
python Workflowmods/performance_lab.py
```

No dependencies required! Uses only Python standard library.

## Directory Structure

```
ComfyUI_PerformanceLab/
├── performance_lab.py              # Main application (v0.2)
├── model_tuner.py                  # Model detection & optimization
├── install.py                      # One-step installer
├── lora_optimizer.py               # LoRA settings optimizer
├── mod_manager.py                  # Simple mod manager
├── mods/                           # Your mod collection
│   ├── vram_optimizer.py           # Reduce VRAM usage
│   ├── bypass_upscalers.py         # Skip upscaling
│   ├── mute_group.py               # Mute node groups
│   └── unwrap_list.py              # Unwrap list nodes
├── performance_lab_config.json     # Auto-saved configuration
└── README.md
```

## Quick Start

```bash
python performance_lab.py
```

1. Enter your workflow JSON path when prompted
2. Use **⚡ Quick Actions [2]** for instant optimizations
3. Or use **🧠 Smart Suggestions [5]** for analysis
4. Run **📊 Benchmark Mode [6]** for reliable baselines
5. Generate LLM prompts with **[3]** for deeper optimization

## Main Menu

| Key | Action | Description |
|-----|--------|-------------|
| **1** | Apply a Mod | Select and apply a mod from `mods/` |
| **2** | ⚡ Quick Actions | One-key optimizations |
| **3** | Generate LLM Prompt | Create prompt for Claude/GPT/Gemini |
| **4** | Paste New Mod | Add mod code from LLM response |
| **5** | 🧠 Smart Suggestions | AI-free workflow analysis |
| **6** | 📊 Benchmark Mode | Run multiple times for metrics |
| **7** | 📈 View Dashboard | Session history & trends |
| **8** | ⚙️ Presets | Apply optimization presets |
| **9** | Set Goal | Tell LLMs what you're optimizing |
| **M** | 🎛️ Model Tuner | Auto-detect model & optimize |
| **C** | Test Connection | Verify ComfyUI API access |
| **T** | Change Target | Switch to different workflow |
| **E** | Export Session | Save session to file |
| **Q** | Quit | Exit (saves configuration) |

## Model Tuner

The Model Tuner automatically detects your model type and applies optimal settings:

### Supported Models

| Model | Resolution | Steps | CFG | Best Samplers |
|-------|------------|-------|-----|---------------|
| SD 1.5 | 512x512 | 25 | 7.5 | dpmpp_2m, euler_ancestral |
| SD 2.1 | 768x768 | 30 | 7.0 | dpmpp_2m, euler |
| SDXL | 1024x1024 | 30 | 7.0 | dpmpp_2m_sde, euler_ancestral |
| SDXL Turbo | 512x512 | 4 | 1.0 | euler_ancestral |
| SD3 | 1024x1024 | 28 | 4.5 | euler, dpmpp_2m |
| Flux Dev | 1024x1024 | 28 | 3.5 | euler, ipndm |
| Flux Schnell | 1024x1024 | 4 | 1.0 | euler |
| Stable Cascade | 1024x1024 | 20 | 4.0 | euler |

### Model Tuner Features

- **Auto-Detection**: Identifies model from workflow nodes and settings
- **Optimal Presets**: Speed, Balanced, Quality, Creative, Consistent
- **LoRA Recommendations**: Suggested strengths per model type
- **Sampler Guide**: Best sampler/scheduler combos for each use case

### Usage

1. Load your workflow
2. Press **M** for Model Tuner
3. Review detected model and optimal settings
4. Choose an action:
   - Apply optimal settings
   - Create speed variant
   - Create quality variant
   - View all recommendations

## Quick Actions

One-key optimizations that instantly create an experimental file:

| Action | Effect | Impact |
|--------|--------|--------|
| Cap 768px | Reduce all resolutions to 768 | ~60% faster |
| Cap 1024px | Reduce all resolutions to 1024 | ~40% faster |
| Bypass Upscalers | Skip all upscaler nodes | 2-4GB VRAM saved |
| Reduce Steps | Set sampling steps to 20 | Faster iteration |
| Reduce Batch | Set batch size to 1 | VRAM reduction |
| 🚀 Speed Test | All optimizations combined | Max speed |
| 💾 8GB VRAM | Optimized for 8GB GPUs | Fit on 8GB cards |
| ↩️ Revert | Restore original workflow | Undo all changes |

## Smart Suggestions

The lab automatically analyzes your workflow and suggests optimizations:

```
[CRITICAL] Cap resolution from 2048px to 768px for testing
   → Very high resolution - cap to 768 for faster iteration

[HIGH] Bypass upscalers during testing
   → Upscalers are VRAM-heavy and not needed for iteration

[MEDIUM] Reduce steps from 50 to 20 for testing
   → 20 steps often sufficient for testing composition
```

No external LLM needed - these are rule-based suggestions from analyzing your workflow structure!

## Benchmark Mode

Run your workflow multiple times for reliable metrics:

```
═══ Run 1/3 ═══
✓ Run 1: 12.45s | Peak VRAM: 7.82GB

═══ Run 2/3 ═══
✓ Run 2: 12.31s | Peak VRAM: 7.81GB

═══ Run 3/3 ═══
✓ Run 3: 12.52s | Peak VRAM: 7.83GB

📊 BENCHMARK RESULTS
Duration:
  Average: 12.43s
  Min: 12.31s
  Max: 12.52s
  Range: ±0.10s

Peak VRAM:
  Average: 7.82 GB
```

## Presets

Quick-apply optimization profiles:

| Preset | Settings |
|--------|----------|
| 🚀 Speed Test | 512px, 15 steps, no upscale, batch 1 |
| 💾 8GB VRAM | 768px, batch 1, no upscale |
| ⚖️ Balanced | 1024px, 25 steps |
| 🎨 Quality | Original settings |

## LLM Prompt Generation

Generate optimized prompts for external LLMs:

- **Claude** - Detailed context, nuanced reasoning
- **GPT-4** - Structured format, explicit instructions
- **Gemini** - Concise, efficient prompts
- **Llama/Mistral** - Clear examples, explicit format

The prompts include:
- Your optimization goal
- Workflow structure analysis
- Node type distribution
- Modification history
- Latest test results
- Request format for mod code

## Writing Mods

Mods are simple Python files:

```python
# mods/my_optimization.py

description = "Brief description shown in the menu"

def apply(content):
    """
    Args:
        content: Parsed JSON workflow (dict)

    Returns:
        Modified dict if changes made, None otherwise
    """
    nodes = content.get("nodes", [])

    for node in nodes:
        # Your optimization logic here
        pass

    return content  # or None if no changes
```

### Node Mode Values

| Mode | Effect |
|------|--------|
| 0 | Always execute (normal) |
| 1 | Bypass (skip, pass inputs through) |
| 2 | Mute (completely disabled) |
| 4 | Never execute |

### Example Mods

**Cap Resolution:**
```python
description = "Cap all resolutions to 512px"

def apply(content):
    for node in content.get("nodes", []):
        widgets = node.get("widgets_values", [])
        for i, w in enumerate(widgets):
            if isinstance(w, int) and w > 512 and w % 8 == 0:
                widgets[i] = 512
    return content
```

**Reduce Steps:**
```python
description = "Reduce sampling steps to 20"

def apply(content):
    for node in content.get("nodes", []):
        if "sampler" in node.get("type", "").lower():
            widgets = node.get("widgets_values", [])
            for i, w in enumerate(widgets):
                if isinstance(w, int) and 20 < w <= 150:
                    widgets[i] = 20
    return content
```

## Metrics Collected

During generation monitoring:

- **Duration**: Total generation time in seconds
- **Peak VRAM**: Maximum GPU memory used
- **Average VRAM**: Mean VRAM during generation
- **Baseline VRAM**: Memory used before generation
- **Error Details**: Node IDs, types, and messages
- **Success/Failure**: Generation completion status

## Configuration

Settings are saved automatically to `performance_lab_config.json`:

- Last workflow path
- Last optimization goal
- Benchmark run count
- ComfyUI URL
- Custom presets

Edit `COMFY_URL` at the top of `performance_lab.py` to change the default:

```python
COMFY_URL = "http://127.0.0.1:8188"  # ComfyUI API address
```

## The Optimization Loop

```
                    ┌─────────────────────────────┐
                    │   Your ComfyUI Workflow     │
                    └─────────────┬───────────────┘
                                  │
           ┌──────────────────────┴──────────────────────┐
           │                                             │
┌──────────▼──────────┐                    ┌─────────────▼──────────┐
│  Quick Actions (2)  │         OR         │  Smart Suggestions (5) │
│  Instant one-key    │                    │  AI-free analysis      │
│  optimizations      │                    │  of your workflow      │
└──────────┬──────────┘                    └─────────────┬──────────┘
           │                                             │
           └──────────────────────┬──────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Test in ComfyUI           │
                    │   (Monitors automatically)  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Review Metrics            │
                    │   • Duration                │
                    │   • VRAM usage              │
                    │   • Errors                  │
                    │   • Comparison to baseline  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Keep or Revert?           │
                    │   Your choice!              │
                    └─────────────┬───────────────┘
                                  │
                                  └──────────► Repeat!
```

## Tips

1. **Start with Smart Suggestions** - Get instant recommendations without an LLM
2. **Use Benchmark Mode** for baselines - Know your starting point
3. **Quick Actions for testing** - Bypass upscalers, reduce resolution
4. **Set a clear goal** - "Reduce VRAM to 8GB" is better than "optimize"
5. **Export sessions** - Track your progress over time
6. **The dashboard shows trends** - See if optimizations are working

## Requirements

- Python 3.7+
- ComfyUI running with API enabled (default: http://127.0.0.1:8188)
- Terminal with ANSI color support (most modern terminals)

## Troubleshooting

**ComfyUI not detected?**
- Make sure ComfyUI is running
- Check the URL with option [C] Test Connection
- Try changing the URL if ComfyUI is on a different port

**Mod not working?**
- Check syntax with `python -m py_compile yourmod.py`
- Ensure `apply()` returns the modified dict
- Return `None` if no changes were made

**Clipboard not working?**
- Install `xclip` on Linux: `sudo apt install xclip`
- Or manually copy the generated prompt

## Version History

- **v0.2.0** - Model Tuner Edition
  - Model Tuner with auto-detection (SD1.5, SDXL, Flux, SD3, etc.)
  - Model-specific optimization presets
  - LoRA strength recommendations
  - Sampler/scheduler recommendations per model
  - One-step installer (install.py)
  - Standalone model_tuner.py module

- **v0.1.0** - Ultimate Edition
  - Quick Actions menu
  - Benchmark Mode
  - Smart Suggestions
  - Presets System
  - Progress Dashboard
  - Workflow Diff
  - Multi-Platform Clipboard
  - Configuration Persistence
  - Built-in Mods Library

- **v2.0** - Original release
  - Basic mod system
  - LLM prompt generation
  - Session history

## License

MIT - Use freely, modify as needed.

---

Made with ⚡ for the ComfyUI community
