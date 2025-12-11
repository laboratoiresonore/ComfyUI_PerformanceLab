# ComfyUI Manager Registry Submission

To make this node pack visible in ComfyUI Manager's "Install Custom Nodes" search, submit a PR to:
https://github.com/ltdrdata/ComfyUI-Manager

Add this entry to `custom-node-list.json`:

```json
{
    "author": "Laboratoire Sonore",
    "title": "Performance Lab",
    "reference": "https://github.com/laboratoiresonore/ComfyUI_PerformanceLab",
    "files": [
        "https://github.com/laboratoiresonore/ComfyUI_PerformanceLab"
    ],
    "install_type": "git-clone",
    "description": "Make any ComfyUI workflow faster, use less VRAM, or produce better quality. 30 nodes for performance monitoring, optimization, analysis, and AI-assisted workflow tuning. Includes automated LLM optimization with Ollama support."
}
```

## Alternative: Install via Git URL

Users can still install manually without registry submission:

1. Open ComfyUI Manager
2. Click "Install via Git URL"
3. Paste: `https://github.com/laboratoiresonore/ComfyUI_PerformanceLab`
4. Restart ComfyUI

## Verifying Installation

After installation, you should see in the console:
```
[Performance Lab] v0.8.0 loaded - 30 nodes available
```

And 30 nodes in the "Performance Lab" category in the node menu.
