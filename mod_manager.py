import os
import importlib.util
import json
import time
import urllib.request
import urllib.error
import sys

MODS_DIR = "mods"
COMFY_URL = "http://127.0.0.1:8188"

# ---------------------------------------------------------
# COMFYUI API HELPERS
# ---------------------------------------------------------
def api_request(endpoint):
    """Generic helper for ComfyUI API requests."""
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/{endpoint}") as response:
            return json.loads(response.read().decode())
    except Exception:
        return None

def wait_for_generation_metrics():
    """
    Monitors ComfyUI to capture performance metrics.
    Returns: (success, duration_seconds)
    """
    print(f"\n📡 Connecting to ComfyUI at {COMFY_URL}...")
    
    # 1. Check initial state
    initial_history = api_request("history")
    if initial_history is None:
        print("⚠️  ComfyUI not found. Is it running? (Skipping auto-monitor)")
        return False, 0

    initial_ids = set(initial_history.keys())
    
    print("⏳ WAITING: Load the '_experimental' file and click 'Queue Prompt'...")
    print("   (Press Ctrl+C to abort waiting)")

    start_time = None
    is_running = False
    
    try:
        while True:
            time.sleep(0.5) # Fast poll
            
            # Check Queue Status
            queue_data = api_request("queue")
            history_data = api_request("history")
            
            if not queue_data or not history_data:
                continue

            pending_count = queue_data.get('queue_pending', [])
            running_count = queue_data.get('queue_running', [])
            
            # A. DETECT START
            if not is_running and (len(pending_count) > 0 or len(running_count) > 0):
                print("🚀 Generation STARTED! Tracking time...")
                start_time = time.time()
                is_running = True

            # B. DETECT FINISH
            current_ids = set(history_data.keys())
            new_ids = current_ids - initial_ids
            
            if is_running and len(new_ids) > 0:
                # Generation finished
                end_time = time.time()
                duration = end_time - start_time
                
                # Check for errors in the new run
                latest_id = list(new_ids)[0]
                run_data = history_data[latest_id]
                status_str = "Unknown"
                if 'status' in run_data:
                    status_str = run_data['status'].get('status_str', 'success')
                
                if status_str == 'error':
                    print("\n❌ Generation FAILED (ComfyUI Error).")
                    return False, duration
                else:
                    return True, duration

    except KeyboardInterrupt:
        print("\n🛑 Waiting aborted by user.")
        return None, 0

# ---------------------------------------------------------
# CORE FILE OPS
# ---------------------------------------------------------
def load_mod(mod_name):
    mod_path = os.path.join(MODS_DIR, mod_name)
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def get_target_file():
    target = input("\nEnter the filename to target (e.g., 'workflow.json'): ").strip()
    if not os.path.exists(target):
        print(f"❌ Error: File '{target}' does not exist.")
        return None
    return target

def create_experimental_path(filepath):
    base, ext = os.path.splitext(filepath)
    return f"{base}_experimental{ext}"

def read_file_smart(filepath):
    if filepath.lower().endswith('.json'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f), "json"
        except: return None, "error"
    else:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read(), "text"
        except: return None, "error"

def write_file_smart(filepath, content, mode):
    try:
        if mode == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        return True
    except Exception as e:
        print(f"❌ Write Error: {e}")
        return False

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main():
    while True:
        # Simple menu refresh
        if not os.path.exists(MODS_DIR): os.makedirs(MODS_DIR)
        mods = [f for f in os.listdir(MODS_DIR) if f.endswith('.py') and f != '__init__.py']

        print("\n=== 🧪  PERFORMANCE LAB MOD MANAGER ===")
        for i, mod_file in enumerate(mods):
            print(f"{i + 1} | {mod_file}")
        
        choice = input("\nEnter mod # (or 'q' to quit): ").strip()
        if choice.lower() == 'q': break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(mods):
                mod_name = mods[idx]
                mod = load_mod(mod_name)
                
                # Show mod description if available
                if hasattr(mod, 'description'):
                    print(f"ℹ️  Info: {mod.description}")

                target_file = get_target_file()
                if not target_file: continue

                content, mode = read_file_smart(target_file)
                if mode == "error": continue

                # APPLY LOGIC
                print(f"⚙️  Applying mod logic...")
                try:
                    new_content = mod.apply(content)
                except Exception as e:
                    print(f"❌ Mod Crashed: {e}")
                    continue

                if new_content is None or new_content == content:
                    print("⚠️  Mod made no changes. Aborting.")
                    continue

                # SAVE EXPERIMENTAL
                exp_file = create_experimental_path(target_file)
                if write_file_smart(exp_file, new_content, mode):
                    print(f"\n📄 Experimental file created: {exp_file}")
                    
                    # PERFORMANCE WAIT LOOP
                    success, duration = wait_for_generation_metrics()
                    
                    if success is None: # User Aborted
                        if os.path.exists(exp_file): os.remove(exp_file)
                        continue

                    print("\n" + "="*40)
                    if success:
                        print(f"✨ GENERATION SUCCESSFUL")
                        print(f"⏱️  Total Time: {duration:.2f} seconds")
                    else:
                        print(f"💥 GENERATION FAILED (Check ComfyUI Console)")
                    print("="*40)

                    # DECISION
                    while True:
                        confirm = input(f"Keep changes to '{target_file}'? (Y/N): ").strip().lower()
                        if confirm == 'y':
                            if os.path.exists(target_file): os.remove(target_file)
                            os.rename(exp_file, target_file)
                            print(f"✅ UPDATED: {target_file} is now running the modded version.")
                            break
                        elif confirm == 'n':
                            if os.path.exists(exp_file): os.remove(exp_file)
                            print(f"🗑️  REVERTED: Experimental file deleted.")
                            break
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
