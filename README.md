# ComfyUI Performance Lab

**Make any ComfyUI workflow faster, use less VRAM, or produce better quality - with help from AI.**

![Performance Lab](https://img.shields.io/badge/ComfyUI-Performance%20Lab-blue) ![Python 3.7+](https://img.shields.io/badge/Python-3.7+-green) ![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen) ![Version](https://img.shields.io/badge/Version-0.4.0-orange)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚡ COMFYUI PERFORMANCE LAB v0.4 ⚡                         ║
║           Load → Test → Get AI Suggestions → Accept/Reject → Repeat          ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## What Does It Do?

Performance Lab helps you optimize ANY ComfyUI workflow - from a simple SD generator to a complex multi-machine network of AI services. It creates a **simple loop**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. LOAD      Load any ComfyUI workflow                                     │
│  2. TEST      Run it and collect performance metrics (time, VRAM)           │
│  3. ASK AI    Generate a prompt and paste it to Claude/GPT/Gemini           │
│  4. GET MOD   Copy the AI's suggested improvement                           │
│  5. PASTE     Paste the mod into Performance Lab                            │
│  6. TEST      Run the modified workflow                                     │
│  7. DECIDE    Better? Accept. Worse? Reject.                                │
│  8. REPEAT    Keep optimizing until you're happy!                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**That's it.** Performance Lab handles all the complexity - you just copy-paste between it and your favorite LLM.

---

## Installation

### Method 1: ComfyUI Manager (Easiest)

If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed:

1. Open ComfyUI in your browser
2. Click **Manager** → **Install via Git URL**
3. Paste: `https://github.com/laboratoiresonore/ComfyUI_PerformanceLab`
4. Click **OK** and restart ComfyUI

Then run Performance Lab from your terminal:
```bash
cd ComfyUI/custom_nodes/ComfyUI_PerformanceLab
python performance_lab.py
```

### Method 2: Git Clone (Recommended)

Open a terminal in your ComfyUI folder:

```bash
# Navigate to your ComfyUI installation
cd /path/to/ComfyUI

# Clone Performance Lab
git clone https://github.com/laboratoiresonore/ComfyUI_PerformanceLab.git custom_nodes/ComfyUI_PerformanceLab

# Run the installer (sets everything up)
cd custom_nodes/ComfyUI_PerformanceLab
python install.py

# Start Performance Lab
python performance_lab.py
```

### Method 3: Download ZIP

1. Download from GitHub: **Code → Download ZIP**
2. Extract to `ComfyUI/custom_nodes/ComfyUI_PerformanceLab/`
3. Open terminal in that folder and run: `python performance_lab.py`

**No additional dependencies required!** Works with Python's standard library.

---

## Quick Start (5 Minutes)

1. **Start ComfyUI** (must be running for metrics collection)

2. **Run Performance Lab**:
   ```bash
   python performance_lab.py
   ```

3. **Load your workflow** when prompted (enter the path to your `.json` file)

4. **Try Quick Actions [2]** - Instant one-click optimizations:
   - Cap resolution to 768px (~60% faster testing)
   - Bypass upscalers (saves 2-4GB VRAM)
   - Speed Test preset (all optimizations at once)

5. **Get AI Help [3]** - Generate a prompt, paste it to Claude/ChatGPT/Gemini, copy the response back

6. **Accept or Reject** - If it's better, keep it. If not, discard it.

---

## Main Menu

```
⚡ MAIN MENU
Target: my_workflow.json
ComfyUI: ● Connected

  1  Apply a Mod (from mods folder)
  2  ⚡ Quick Actions (instant one-click optimizations)
  3  Generate LLM Prompt (for Claude/GPT/Gemini)
  4  Paste New Mod (from AI response)
  5  🧠 Smart Suggestions (AI-free analysis)
  6  📊 Benchmark Mode (run 3x for reliable metrics)
  7  📈 View Dashboard (see all your optimizations)
  8  ⚙️  Presets (8GB VRAM, Speed Test, etc.)
  9  Set Goal (tell the AI what you want)
  L  🤖 LLM Enhancer (advanced AI context)
  M  🎛️  Model Tuner (auto-detect SD/SDXL/Flux)
  D  🌐 Distributed (multi-machine optimization)
  Q  Quit
```

---

## What Makes It Different?

### Simple Iterative Loop
Most optimization tools give you one suggestion. Performance Lab lets you **iterate** - make a change, test it, keep or discard, repeat. This is how real optimization works.

### Works with Any LLM
Use Claude, ChatGPT, GPT-4, Gemini, Llama, Mistral - whatever you have access to. Performance Lab generates optimized prompts for each.

### Minimal File Clutter
- **Accept** = overwrites your original workflow
- **Reject** = discards the changes completely
- No `_v1`, `_v2`, `_final_FINAL` files everywhere

### Smart Protection
Performance Lab knows when changes are risky:
- Won't overwrite an SD 1.5 workflow with SDXL settings
- Warns you about significant changes
- Tracks what model family you're using

---

## Quick Actions (No AI Needed)

Press **[2]** for instant optimizations:

| Action | What It Does | Expected Improvement |
|--------|-------------|---------------------|
| Cap 768px | Reduce all resolutions to 768 | ~60% faster |
| Cap 1024px | Reduce all resolutions to 1024 | ~40% faster |
| Bypass Upscalers | Skip all upscaler nodes | 2-4GB VRAM saved |
| Reduce Steps | Set sampling steps to 20 | Faster iteration |
| 🚀 Speed Test | All optimizations combined | Maximum speed |
| 💾 8GB VRAM | Optimized for 8GB GPUs | Fits on 8GB cards |

---

## AI-Assisted Optimization

Press **[3]** to generate a prompt for your favorite LLM:

1. Performance Lab creates a detailed prompt with:
   - Your workflow structure
   - Current performance metrics
   - Your optimization goal
   - System specs (GPU, VRAM, etc.)

2. Copy the prompt to Claude/ChatGPT/Gemini

3. The AI responds with optimized workflow code

4. Press **[4]** and paste the response

5. Test it, then Accept or Reject

### Supported Goals

Tell the AI what you want with **[9] Set Goal**:

- **Speed** - "Make this run in under 10 seconds"
- **VRAM** - "Make this work on 8GB VRAM"
- **Quality** - "Improve image quality without changing speed much"
- **Debug** - "Fix why this keeps giving black images"

---

## Multi-Machine / Network Workflows (v0.4)

Press **[D]** for distributed workflow optimization.

If your workflow uses remote services (Kobold LLM, remote ComfyUI, TTS, STT), Performance Lab can:

- **Health check** all your endpoints
- **Measure latency** to each service
- **Find bottlenecks** (which machine is slow?)
- **Suggest parallelization** (what can run simultaneously?)
- **Generate smart prompts** with your machine specs so the AI can suggest things like "move STT to machine 2" or "lower context size on the LLM server"

### Included Network Nodes

The `ComfyUI_NetworkServices` custom node pack includes:

| Node | What It Does |
|------|-------------|
| **KoboldLLM** | Connect to Kobold instances for text generation |
| **RemoteComfyUI** | Run workflows on remote ComfyUI servers |
| **LocalGenerator** | Universal REST API node (70+ presets) |
| **EndpointHealthCheck** | Monitor service availability |

### Supported Services (70+ presets)

**Image/Video**: ComfyUI, Automatic1111, Forge, InvokeAI, Fooocus, AnimateDiff, SVD, Mochi, CogVideo...

**LLM**: KoboldCpp, Ollama, llama.cpp, Text-Gen-WebUI, vLLM, LMDeploy, TGI, LocalAI...

**STT**: Whisper, Faster-Whisper, whisper.cpp, WhisperX, NeMo ASR, Vosk...

**TTS**: Coqui TTS, XTTS, AllTalk, Silero, Piper, Bark, Tortoise, StyleTTS2...

**And more**: Embeddings, Vision models, Audio generation, Upscaling...

---

## Model Tuner

Press **[M]** to auto-detect your model and apply optimal settings.

| Model | Resolution | Steps | CFG | Best Samplers |
|-------|------------|-------|-----|---------------|
| SD 1.5 | 512x512 | 25 | 7.5 | dpmpp_2m, euler_ancestral |
| SDXL | 1024x1024 | 30 | 7.0 | dpmpp_2m_sde, euler_ancestral |
| SD3 | 1024x1024 | 28 | 4.5 | euler, dpmpp_2m |
| Flux Dev | 1024x1024 | 28 | 3.5 | euler, ipndm |
| Flux Schnell | 1024x1024 | 4 | 1.0 | euler |

---

## LLM Enhancer

Press **[L]** for advanced AI context generation:

- **Node Catalog** - Shows all your installed ComfyUI nodes
- **System Specs** - Includes your GPU, VRAM, CPU info
- **Knowledge Base** - Common issues and solutions
- **Mod Validation** - Validates AI responses before applying
- **Conversation Memory** - Remembers context across sessions

---

## Writing Custom Mods

Create a Python file in the `mods/` folder:

```python
# mods/my_optimization.py

description = "What this mod does (shown in menu)"

def apply(content):
    """
    content: The workflow as a Python dict
    Returns: Modified dict, or None if no changes
    """
    for node in content.get("nodes", []):
        # Your optimization logic here
        pass
    return content
```

---

## Troubleshooting

**ComfyUI not detected?**
- Make sure ComfyUI is running before starting Performance Lab
- Check connection with **[C]** Test Connection
- Default URL is `http://127.0.0.1:8188`

**Clipboard not working on Linux?**
- Install xclip: `sudo apt install xclip`
- Or manually copy/paste the prompts

**Mod not working?**
- Test syntax: `python -m py_compile mods/yourmod.py`
- Make sure `apply()` returns the modified content

---

## File Structure

```
ComfyUI_PerformanceLab/
├── performance_lab.py       # Main application
├── llm_enhancer.py          # AI context generation
├── model_tuner.py           # Model detection & tuning
├── workflow_utils.py        # Workflow analysis
├── distributed_optimizer.py # Multi-machine support
├── services_config.py       # Network services config
├── logging_config.py        # Logging setup
├── install.py               # One-step installer
├── mods/                    # Your mod collection
│   ├── vram_optimizer.py
│   ├── bypass_upscalers.py
│   └── ...
├── custom_nodes/            # ComfyUI network nodes
│   └── ComfyUI_NetworkServices/
└── tests/                   # Test suite
```

---

## Version History

**v0.4.0** - Multi-Machine Distributed Optimization
- New [D] Distributed menu for multi-machine pipelines
- ComfyUI_NetworkServices node pack (70+ service presets)
- KoboldLLM, RemoteComfyUI, LocalGenerator, HealthCheck nodes
- Machine profiling and bottleneck detection
- Services configuration file support
- Improved error handling and logging
- Test suite with pytest

**v0.3.0** - LLM Enhancer
- Advanced AI context generation
- Node Catalog, System Specs, Knowledge Base
- Mod Validation, Conversation Memory

**v0.2.0** - Model Tuner
- Auto-detection for SD1.5, SDXL, Flux, SD3
- Model-specific presets and LoRA recommendations

**v0.1.0** - Initial Release
- Quick Actions, Benchmark Mode, Smart Suggestions
- Presets, Dashboard, Multi-platform clipboard

---

## License

MIT - Use freely, modify as needed.

---

Made with ⚡ for the ComfyUI community
