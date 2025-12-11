# ComfyUI Performance Lab v0.1 - Ultimate Edition

**Iterative Workflow Optimization with LLM-Assisted Analysis & Smart Features**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║             ⚡ COMFYUI PERFORMANCE LAB v0.1 - ULTIMATE EDITION ⚡             ║
║     Iterative Workflow Optimization with LLM-Assisted Analysis & More!       ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

![Performance Lab](https://img.shields.io/badge/ComfyUI-Performance%20Lab-blue) ![Python 3.7+](https://img.shields.io/badge/Python-3.7+-green) ![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen) ![Version](https://img.shields.io/badge/Version-0.1.0-orange)

## What's New in v0.1

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

```bash
# Clone or copy the files
mkdir comfyui-performance-lab
cd comfyui-performance-lab

# No dependencies required! Uses only Python standard library.
python performance_lab.py
```

## Directory Structure

```
comfyui-performance-lab/
├── performance_lab.py              # Main application (v0.1)
├── performance_lab_backup.py       # Backup of v2.0
├── mod_manager.py                  # Simple mod manager (v1.0)
├── mods/                           # Your mod collection
│   ├── vram_optimizer.py           # Reduce VRAM usage
│   ├── bypass_upscalers.py         # Skip upscaling
│   └── reduce_steps.py             # Lower step count
├── performance_lab_config.json     # Auto-saved configuration
├── session_*.json                  # Exported sessions
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
| **C** | Test Connection | Verify ComfyUI API access |
| **T** | Change Target | Switch to different workflow |
| **E** | Export Session | Save session to file |
| **Q** | Quit | Exit (saves configuration) |

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
