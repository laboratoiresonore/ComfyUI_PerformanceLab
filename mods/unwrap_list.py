"""
Workflow List Unwrapper
Fixes workflows exported as arrays instead of objects.
"""

description = "Unwraps JSON list into single workflow object (fixes export format)"

def apply(content):
    """
    If the workflow was exported as a list, extract the first item.
    
    Args:
        content: Either a list containing workflow(s) or a workflow dict
    
    Returns:
        The workflow dict, or None if no changes needed
    """
    if isinstance(content, list):
        if len(content) > 0:
            print(f"   -> Detected list with {len(content)} item(s). Extracting first.")
            return content[0]
        else:
            print("   -> List is empty, cannot unwrap")
            return None
    
    print("   -> Content is already a dict, no unwrapping needed")
    return None
