# Performance Lab Example Workflows

This directory contains example ComfyUI workflows demonstrating the Performance Lab nodes.

## Available Workflows

### ✅ `performance_lab_v2.json` (CURRENT - v2.0)

**Status:** ✅ Fully compatible with v2.0

**Description:** Complete demonstration of all 11 v2.0 nodes including:
- Basic performance monitoring (Timer, Report, VRAM Monitor)
- Auto GPU detection and suggestions (AutoFix)
- LLM-powered optimization (Optimizer, A/B Test)
- Preference learning (Feedback)
- Network discovery and LiteLLM setup (NetworkSetup)

**Use this workflow to:**
1. Learn how to use all v2.0 features
2. Copy individual node groups for your own workflows
3. Test Performance Lab functionality

**Quick Start:**
```
1. Load this workflow in ComfyUI
2. Connect your actual workflow between the Timer and Report nodes
3. Connect AutoFix outputs to your KSampler/EmptyLatent
4. Run to get performance metrics and GPU-aware suggestions
```

---

### ❌ `performance_lab_v0.9_legacy.json` (DEPRECATED)

**Status:** ❌ NOT compatible with v2.0 (LEGACY)

**Description:** Old v0.9 workflow using nodes that were removed in v2.0:
- `PerfLab_OptimizationLoop` → Use `PerfLab_Optimizer` + `PerfLab_ABTest`
- `PerfLab_OneClickOptimize` → Use `PerfLab_AutoFix`
- `PerfLab_AutoDetectGPU` → Use `PerfLab_AutoFix`
- `PerfLab_QuickStart` → See README.md and workflow descriptions
- `PerfLab_SmartPrompt` → Use `PerfLab_Optimizer`

**Migration Guide:**
If you have old workflows, replace nodes as follows:

| Old Node (v0.9) | New Node (v2.0) | Notes |
|-----------------|-----------------|-------|
| OptimizationLoop | LLM Optimizer | More flexible, works with any LLM |
| OneClickOptimize | AutoFix | Automatic GPU detection built-in |
| AutoDetectGPU | AutoFix | Combined into one node |
| QuickStart | (removed) | Use node descriptions instead |
| SmartPrompt | LLM Optimizer | More powerful prompting |

---

## Node Categories

### Core Utility Nodes (6)
- ⏱️ **Start Timer** - Place at workflow start
- 📊 **Performance Report** - Place at workflow end
- 💾 **VRAM Monitor** - Check memory anywhere
- 📝 **Show Text** - Display outputs
- 📐 **Cap Resolution** - Limit size for testing
- 📊 **Compare Results** - Before/after comparison

### LLM-Powered Nodes (5)
- 🪄 **AutoFix** - Drop anywhere for auto GPU detection
- 🧠 **LLM Optimizer** - KoboldCPP/Ollama integration
- 🔬 **A/B Test** - Compare configurations
- 👍 **Record Preference** - Train the system
- 🌐 **Network Setup** - Multi-machine discovery

---

## Usage Tips

### Basic Monitoring (No LLM)
```
[Timer] → [Your Workflow] → [Report] → [ShowText]
```

### Auto GPU Suggestions (No LLM)
```
[AutoFix] → Connect outputs to:
  - suggested_steps → KSampler steps
  - suggested_resolution → EmptyLatent width/height
  - suggested_cfg → KSampler cfg
```

### LLM-Guided Optimization
```
[Timer] → [Workflow] → [Report] → [LLM Optimizer] → [ShowText]
                                         ↓
Enter issue: "Images too dark"
Get: Specific parameter fixes
```

### A/B Testing
```
[Report A] ─┬─→ [Compare Results] → See % improvement
[Report B] ─┘

[Config A] ─┬─→ [A/B Test] → Get theoretical speedup
[Config B] ─┘
```

---

## LLM Setup

Performance Lab works with:
- **KoboldCPP** (default: http://127.0.0.1:5001)
- **Ollama** (http://127.0.0.1:11434)
- **LiteLLM** (for load balancing multiple backends)

### Quick LLM Setup

**Option 1: KoboldCPP** (Recommended)
```bash
# Download and run KoboldCPP
./koboldcpp --model your_model.gguf --port 5001
```

**Option 2: Ollama**
```bash
ollama serve
ollama pull llama3.2
```

**Option 3: LiteLLM** (Multi-machine)
```bash
# Use Network Setup node to discover services
# Copy generated config to ~/.litellm/config.yaml
litellm --config ~/.litellm/config.yaml
# Then point LLM Optimizer to http://localhost:4000
```

---

## Creating Your Own Workflow

1. **Start Simple:**
   - Add Timer at start
   - Add Report at end
   - Run to get baseline metrics

2. **Add Auto-Suggestions:**
   - Drop AutoFix node anywhere
   - Connect outputs to your sampler settings
   - Get GPU-aware recommendations

3. **Optimize with LLM:**
   - Connect Report to LLM Optimizer
   - Describe your issue
   - Apply suggested changes

4. **Compare Results:**
   - Use Compare Results for before/after
   - Use A/B Test for configuration testing
   - Record Preference to train the system

---

## Version History

- **v2.0.0** - Current version (11 focused nodes)
- **v0.9.0** - Legacy version (deprecated, 31 nodes)

For more info, see the main [README.md](../README.md)
