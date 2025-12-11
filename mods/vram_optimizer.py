"""
VRAM Optimizer for 5060 Ti / 8GB Cards
Reduces batch sizes, enables VAE tiling, and caps resolution.
"""

description = "VRAM optimization: batch=1, VAE tiling, resolution cap at 832px"

# Configuration
SAFE_RESOLUTION = 832
VAE_TILE_SIZE = [256, 64, 8, 4]  # [tile_width, tile_height, overlap_width, overlap_height]
BATCH_SIZE = 1

def apply(content):
    """
    Apply VRAM optimizations to a ComfyUI workflow.
    
    Args:
        content: Parsed JSON workflow (dict)
    
    Returns:
        Modified workflow dict, or None if no changes made
    """
    if not isinstance(content, dict):
        print("   -> Expected dict (JSON workflow)")
        return None
    
    nodes = content.get("nodes", [])
    if not nodes:
        print("   -> No nodes found in workflow")
        return None
    
    changes_made = 0
    
    for node in nodes:
        node_type = node.get("type", "").lower()
        widgets = node.get("widgets_values", [])
        
        # 1. Cap resolution sliders
        if "resolution" in node_type or "size" in node_type:
            for i, w in enumerate(widgets):
                if isinstance(w, int) and w > SAFE_RESOLUTION and w % 8 == 0:
                    print(f"   -> Node {node.get('id')}: Capping resolution {w} → {SAFE_RESOLUTION}")
                    widgets[i] = SAFE_RESOLUTION
                    changes_made += 1
        
        # 2. Enable VAE tiling
        if "vae" in node_type and "tile" in node_type:
            if len(widgets) >= 4:
                print(f"   -> Node {node.get('id')}: Setting VAE tile size")
                node["widgets_values"] = VAE_TILE_SIZE
                changes_made += 1
        
        # 3. Reduce batch sizes to 1
        if "batch" in node_type or "sampler" in node_type:
            for i, w in enumerate(widgets):
                if isinstance(w, int) and w > BATCH_SIZE and w < 100:  # Likely a batch size
                    print(f"   -> Node {node.get('id')}: Reducing batch {w} → {BATCH_SIZE}")
                    widgets[i] = BATCH_SIZE
                    changes_made += 1
        
        # 4. Wan-specific nodes (common in video workflows)
        if "wan" in node_type:
            # Last widget is often batch size
            if widgets and isinstance(widgets[-1], int) and widgets[-1] > 1:
                print(f"   -> Node {node.get('id')}: Wan batch {widgets[-1]} → {BATCH_SIZE}")
                widgets[-1] = BATCH_SIZE
                changes_made += 1
    
    if changes_made > 0:
        print(f"   -> Total: {changes_made} optimizations applied")
        return content
    else:
        print("   -> No VRAM optimizations applicable")
        return None
