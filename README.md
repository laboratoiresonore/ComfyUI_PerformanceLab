# ComfyUI Performance Lab

> **One-Line Install** (ComfyUI Manager with security set to "weak"):
> ```
> https://github.com/laboratoiresonore/ComfyUI_PerformanceLab
> ```
> Paste this URL in ComfyUI Manager → Install via Git URL

---

**Iterative Workflow Optimization with Smart Model Detection & One-Click Tuning**

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚡ COMFYUI PERFORMANCE LAB ⚡                              ║
║       Auto-Detect Models • Smart Optimization • LoRA Tuning • More!         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

![Performance Lab](https://img.shields.io/badge/ComfyUI-Performance%20Lab-blue) ![Python 3.7+](https://img.shields.io/badge/Python-3.7+-green) ![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen) ![Version](https://img.shields.io/badge/Version-0.4.0-orange)

## What's New in v0.4 - Multi-Machine Distributed Optimization

| Feature | Description |
|---------|-------------|
| 🌐 **Distributed Menu** | New [D] menu for multi-machine AI pipeline optimization |
| 🔌 **Network Service Nodes** | ComfyUI_NetworkServices node pack with 70+ service presets |
| 🦙 **KoboldLLM Node** | Connect to Kobold instances for LLM on remote machines |
| 🖼️ **RemoteComfyUI Node** | Execute SD/Flux/video workflows on remote ComfyUI servers |
| 🎛️ **LocalGenerator Node** | Universal REST API node for any AI service (STT, TTS, Embeddings, etc.) |
| 💓 **Health Check Nodes** | Monitor endpoint availability and measure latencies |
| 🗺️ **Machine Profiles** | Register GPU/CPU specs for each machine in your network |
| 📊 **Bottleneck Detection** | Identify which machine is slowing your pipeline |
| ⚡ **Parallel Analysis** | Find nodes that can run simultaneously |
| 🤖 **Distributed LLM Prompts** | Generate prompts that include machine specs for better recommendations |

### Supported Services (70+ presets)

**Image/Video**: ComfyUI, Automatic1111, Forge, InvokeAI, Fooocus, SwarmUI, Kohya, AnimateDiff, SVD, Mochi, CogVideo, Hunyuan, LTX-Video

**LLM**: KoboldCpp, Ollama, llama.cpp, Text-Gen-WebUI, vLLM, LMDeploy, TGI, LocalAI, Jan, LM Studio, GPT4All, ExLlama, TabbyAPI, Aphrodite

**STT**: Whisper, Faster-Whisper, whisper.cpp, WhisperX, NeMo ASR, Vosk

**TTS**: Coqui TTS, XTTS, AllTalk, Silero, Piper, Bark, Tortoise, StyleTTS2, OpenVoice, Fish Speech

**Embeddings**: Text Embeddings Inference, Sentence Transformers, Infinity, FastEmbed

**And more**: LLaVA, CogVLM, Moondream, AudioCraft, MusicGen, Real-ESRGAN, GFPGAN, SAM, Florence-2...

### Previous Features (v0.3) - LLM Enhancement

| Feature | Description |
|---------|-------------|
| 🤖 **LLM Enhancer** | Advanced AI context generation for better LLM assistance |
| 📋 **Node Catalog Export** | Query installed ComfyUI nodes to include in prompts |
| 💻 **System Specs** | Include GPU/VRAM/CPU info for context-aware optimization |
| 🎯 **Goal-Based Prompts** | Templates for Debug, Speed, Quality, VRAM, Explain |
| ✅ **Mod Validation** | Validate LLM-generated workflow mods before applying |
| 📜 **Conversation Memory** | Remember context across optimization sessions |
| 📚 **Knowledge Base** | Common issues and solutions library |
| 🗺️ **Workflow Graph** | ASCII/Mermaid visualization of workflow structure |
| 📝 **Error History** | Track and export errors for debugging |

### Previous Features (v0.2)

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

Performance Lab creates an **iterative optimization cycle** with minimal file clutter:

### The Workflow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  1. LOAD      Load any ComfyUI workflow                                │
│  2. COLLECT   Run Performance Lab data collection (benchmark baseline)  │
│  3. GENERATE  Generate LLM prompt with workflow + metrics              │
│  4. SUBMIT    Copy-paste prompt to Claude/Gemini/GPT-4                 │
│  5. RECEIVE   Get formatted mod code from LLM                          │
│  6. PASTE     Paste mod directly into Performance Lab                  │
│  7. TEST      Run optimized workflow in ComfyUI (auto-monitors)        │
│  8. REVIEW    See metrics: timing, VRAM, errors, comparison            │
│  9. DECIDE    Accept (overwrites original) or Reject (discards)        │
│  10. REPEAT   Continue until fully optimized                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Principles

- **Minimal file clutter**: Accept overwrites the original, Reject discards changes
- **Smart protection**: Never overwrites incompatible configs (e.g., SD → Flux)
- **Model detection**: Auto-detects when model family changes
- **Use "Save As" only when needed**: Explicit saves for version branching

### Safety Features

- **Fingerprint detection**: Tracks model family, resolution, features
- **Incompatible overwrite protection**: Blocks SD1.5 → SDXL → Flux overwrites
- **Warnings**: Alerts for significant changes before applying

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
├── performance_lab.py              # Main application (v0.3)
├── llm_enhancer.py                 # LLM context generation & validation
├── model_tuner.py                  # Model detection & optimization
├── workflow_utils.py               # Fingerprinting & beautification
├── install.py                      # One-step installer
├── knowledge_base.json             # Common issues & solutions (auto-generated)
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
| **L** | 🤖 LLM Enhancer | Advanced AI context generation |
| **M** | 🎛️ Model Tuner | Auto-detect model & optimize |
| **B** | 🎨 Beautify | Organize & clean up workflow |
| **S** | 💾 Save As | Save to new file (branch version) |
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

## Workflow Beautifier

Press **B** to organize and clean up your workflow layout:

| Mode | Description |
|------|-------------|
| 📐 Organize by Category | Group nodes by function (input, model, sampling, output) |
| 📏 Align to Grid | Snap all nodes to 50px grid |
| ➡️ Flow Left-to-Right | Arrange nodes in L→R processing flow |
| ⬇️ Flow Top-to-Down | Arrange nodes in T→D processing flow |
| 📦 Compact Layout | Minimize space (0.7x spacing) |
| 📭 Expand Layout | Add breathing room (1.4x spacing) |
| 🎨 Color Code Nodes | Color nodes by function |
| 📁 Create Groups | Add visual group boxes around categories |

### Node Categories

Nodes are automatically categorized:

- **Input**: Loaders, images, masks
- **Model**: Checkpoints, UNETs, VAEs, CLIPs, LoRAs
- **Conditioning**: Prompts, text encoders
- **Sampling**: KSamplers, schedulers
- **ControlNet**: Preprocessors, ControlNet apply
- **IPAdapter**: Face/style adapters
- **Upscale**: ESRGAN, Ultimate Upscale
- **Output**: Save, preview nodes

## LLM Enhancer

Press **L** to access advanced AI context generation for better LLM assistance:

### Features

| Option | Description |
|--------|-------------|
| **📋 Generate Full Context** | Comprehensive prompt with all available context |
| **🔧 Debug Workflow** | Generate debug-focused prompt for fixing errors |
| **⚡ Optimize Speed** | Generate speed optimization prompt |
| **🎨 Improve Quality** | Generate quality improvement prompt |
| **💾 Reduce VRAM** | Generate VRAM optimization prompt |
| **📖 Explain Workflow** | Generate explanation request prompt |
| **✅ Validate Response** | Validate and parse LLM mod response before applying |
| **🔍 Node Catalog** | Browse installed ComfyUI nodes |
| **💻 System Specs** | Show current hardware context |
| **📚 Knowledge Base** | Browse and search common solutions |
| **📜 History** | View past LLM interactions |
| **🗺️ Workflow Graph** | ASCII/Mermaid visualization |

### Goal-Based Prompt Templates

Each goal generates a specialized prompt:

- **Debug**: Includes error logs, node validation, fix format
- **Speed**: Focuses on step reduction, samplers, caching
- **Quality**: Upscaling, refinement passes, CFG tuning
- **VRAM**: Tiled VAE, fp16/fp8, memory-efficient techniques
- **Explain**: Educational breakdown of workflow components

### Node Catalog

Queries ComfyUI's `/object_info` API to get all installed nodes:

```
# Available ComfyUI Nodes
Total nodes: 847
Categories: 42

## conditioning (15 nodes)
  - CLIPTextEncode [conditioning] - in(clip: CLIP, text: STRING) -> out(CONDITIONING)
  - ConditioningCombine [conditioning] - in(conditioning_1: CONDITIONING) -> out(CONDITIONING)
  ...
```

### System Specs

Automatically detects and includes:

- OS and version
- CPU name and cores
- RAM amount
- GPU name and VRAM
- CUDA version
- PyTorch version
- Optimization notes based on hardware

### Mod Validation

Validates LLM-generated mods before applying:

- **JSON syntax check**: Fixes common formatting issues
- **Node validation**: Verifies all nodes exist in catalog
- **Link validation**: Checks connections are valid
- **Breaking change detection**: Warns about removed nodes or type changes
- **Auto-fix**: Attempts to repair common issues

### Conversation Memory

Persists context across sessions (SQLite database):

- Tracks optimization attempts per workflow
- Records applied/success status
- Shows success rates and patterns
- Provides history context in prompts

### Knowledge Base

Built-in solutions for common issues:

- **VRAM**: OOM errors, tiling, precision
- **Quality**: Black images, artifacts, CFG tuning
- **Speed**: Slow generation, CUDA issues
- **Models**: Flux CFG, LoRA strength, sampler selection
- **Nodes**: Missing nodes, compatibility

Entries are automatically matched to your workflow.

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

- **v0.3.0**
  - LLM Enhancer module for advanced AI context generation
  - Node Catalog Export - queries ComfyUI /object_info API
  - System Specs collector - GPU, VRAM, CPU info for context
  - Goal-based prompt templates (Debug, Speed, Quality, VRAM, Explain)
  - Mod Validation Layer - validates LLM responses before applying
  - Conversation Memory - persists context across sessions
  - Knowledge Base - common issues and solutions library
  - Workflow Graph Export - ASCII and Mermaid visualizations
  - Error History tracking and export

- **v0.2.1**
  - Iterative workflow (accept=overwrite, reject=discard)
  - Workflow fingerprinting & incompatible overwrite protection
  - Beautification menu (organize, align, color code)
  - Save As option for explicit version branching

- **v0.2.0**
  - Model Tuner with auto-detection (SD1.5, SDXL, Flux, SD3, etc.)
  - Model-specific optimization presets
  - LoRA strength recommendations
  - Sampler/scheduler recommendations per model
  - One-step installer (install.py)

- **v0.1.0**
  - Quick Actions menu
  - Benchmark Mode
  - Smart Suggestions
  - Presets System
  - Progress Dashboard
  - Multi-Platform Clipboard
  - Configuration Persistence
  - Built-in Mods Library

## License

MIT - Use freely, modify as needed.

---

Made with ⚡ for the ComfyUI community
