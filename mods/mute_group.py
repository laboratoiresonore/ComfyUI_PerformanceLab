"""
Group Muter
Mutes all nodes within a specified group by name.
"""

description = "Mutes all nodes within a group (interactive: asks for group name)"

def apply(content):
    """
    Mute all nodes within a specific group.
    
    ComfyUI node modes:
        0 = Always execute
        1 = Bypass (skip)
        2 = Mute (disabled)
        4 = Never execute
    
    Args:
        content: Parsed JSON workflow (dict)
    
    Returns:
        Modified workflow with group nodes muted
    """
    if not isinstance(content, dict):
        return None
    
    groups = content.get("groups", [])
    nodes = content.get("nodes", [])
    
    if not groups:
        print("   -> No groups found in this workflow")
        return None
    
    # Show available groups
    print("\n   Available groups:")
    for i, group in enumerate(groups):
        title = group.get("title", f"Untitled ({i})")
        print(f"      {i + 1}. {title}")
    
    # Ask which group to mute
    try:
        choice = input("\n   Enter group # to mute (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return None
        
        idx = int(choice) - 1
        if not (0 <= idx < len(groups)):
            print("   -> Invalid group number")
            return None
    except (ValueError, EOFError):
        return None
    
    target_group = groups[idx]
    group_title = target_group.get("title", "Untitled")
    
    # Get group bounding box
    bounding = target_group.get("bounding", [0, 0, 0, 0])
    if len(bounding) < 4:
        print("   -> Group has invalid bounding box")
        return None
    
    gx, gy, gw, gh = bounding
    
    # Mute nodes within the group
    muted_count = 0
    for node in nodes:
        pos = node.get("pos", [0, 0])
        if len(pos) >= 2:
            nx, ny = pos[0], pos[1]
            # Check if node is within group bounds
            if (gx <= nx <= gx + gw) and (gy <= ny <= gy + gh):
                node["mode"] = 2  # Muted
                muted_count += 1
    
    if muted_count > 0:
        print(f"   -> Muted {muted_count} nodes in group '{group_title}'")
        return content
    else:
        print(f"   -> No nodes found in group '{group_title}'")
        return None
