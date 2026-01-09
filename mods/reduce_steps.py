description = "Reduce sampling steps to 20 for faster iteration"

def apply(content):
    nodes = content.get("nodes", [])
    changed = False

    for node in nodes:
        node_type = node.get("type", "").lower()
        if "sampler" in node_type:
            widgets = node.get("widgets_values", [])
            for i, w in enumerate(widgets):
                if isinstance(w, int) and 20 < w <= 150:
                    widgets[i] = 20
                    changed = True

    return content if changed else None
