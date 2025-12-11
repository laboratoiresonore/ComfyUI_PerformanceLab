"""
Bypass Upscalers
Bypasses all upscaling nodes to speed up testing.
"""

description = "Bypasses upscaling nodes (ESRGAN, Ultimate SD, etc.) for faster testing"

# Node types to bypass (case-insensitive partial match)
BYPASS_PATTERNS = [
    "upscale",
    "esrgan", 
    "real-esrgan",
    "ultim",
    "4x",
    "2x",
    "supir",
]

def apply(content):
    """
    Bypass upscaling nodes for faster iteration.
    
    ComfyUI node modes:
        0 = Always execute
        1 = Bypass (skip)
        2 = Mute (disabled)
    
    Args:
        content: Parsed JSON workflow (dict)
    
    Returns:
        Modified workflow with upscalers bypassed
    """
    if not isinstance(content, dict):
        return None
    
    nodes = content.get("nodes", [])
    bypassed_count = 0
    
    for node in nodes:
        node_type = node.get("type", "").lower()
        
        # Check if this node matches any bypass pattern
        for pattern in BYPASS_PATTERNS:
            if pattern in node_type:
                current_mode = node.get("mode", 0)
                if current_mode != 1:  # Not already bypassed
                    node["mode"] = 1  # Bypass
                    print(f"   -> Bypassing node {node.get('id')}: {node.get('type')}")
                    bypassed_count += 1
                break
    
    if bypassed_count > 0:
        print(f"   -> Total: {bypassed_count} upscaling nodes bypassed")
        return content
    else:
        print("   -> No upscaling nodes found to bypass")
        return None
