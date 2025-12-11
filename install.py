#!/usr/bin/env python3
"""
Performance Lab - One-Step Installer

Run this script from anywhere to install Performance Lab:
    python install.py

Or with curl (if hosted):
    curl -sSL https://raw.githubusercontent.com/laboratoiresonore/ComfyUI_PerformanceLab/main/install.py | python3
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

REPO_URL = "https://github.com/laboratoiresonore/ComfyUI_PerformanceLab.git"
REPO_NAME = "ComfyUI_PerformanceLab"

# Common ComfyUI installation paths
COMFY_PATHS = [
    # Linux
    Path.home() / "AI" / "ComfyUI",
    Path.home() / "ComfyUI",
    Path("/opt/ComfyUI"),
    Path("/usr/local/ComfyUI"),
    # Windows
    Path("C:/ComfyUI"),
    Path("C:/AI/ComfyUI"),
    Path.home() / "Documents" / "ComfyUI",
    # macOS
    Path.home() / "Applications" / "ComfyUI",
    # Current directory
    Path.cwd(),
    Path.cwd().parent,
]

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════════

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════════╗
║         ⚡ PERFORMANCE LAB - ONE-STEP INSTALLER ⚡               ║
╚══════════════════════════════════════════════════════════════════╝{C.RESET}
""")

def success(msg): print(f"{C.GREEN}✓ {msg}{C.RESET}")
def error(msg): print(f"{C.RED}✗ {msg}{C.RESET}")
def info(msg): print(f"{C.BLUE}ℹ {msg}{C.RESET}")
def warn(msg): print(f"{C.YELLOW}⚠ {msg}{C.RESET}")

# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_comfyui() -> Path | None:
    """Find ComfyUI installation."""
    # Check environment variable
    if env_path := os.environ.get("COMFYUI_PATH"):
        p = Path(env_path)
        if (p / "main.py").exists() or (p / "comfy").exists():
            return p

    # Search common paths
    for path in COMFY_PATHS:
        if path.exists():
            # Check if it's ComfyUI root
            if (path / "main.py").exists() or (path / "comfy").exists():
                return path
            # Check if it contains ComfyUI
            for subdir in path.iterdir():
                if subdir.is_dir() and ((subdir / "main.py").exists() or (subdir / "comfy").exists()):
                    return subdir

    return None

def has_git() -> bool:
    """Check if git is available."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def is_already_installed(comfy_path: Path) -> Path | None:
    """Check if Performance Lab is already installed."""
    custom_nodes = comfy_path / "custom_nodes" / REPO_NAME
    if custom_nodes.exists():
        return custom_nodes

    # Also check Workflowmods location
    workflowmods = comfy_path / "Workflowmods"
    if workflowmods.exists() and (workflowmods / "performance_lab.py").exists():
        return workflowmods

    return None

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

def install_via_git(dest: Path) -> bool:
    """Install via git clone."""
    try:
        subprocess.run(
            ["git", "clone", REPO_URL, str(dest)],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        error(f"Git clone failed: {e.stderr.decode()}")
        return False

def install_via_download(dest: Path) -> bool:
    """Install via direct download (fallback)."""
    import urllib.request
    import zipfile
    import tempfile

    zip_url = f"{REPO_URL.replace('.git', '')}/archive/refs/heads/main.zip"

    try:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            info(f"Downloading from {zip_url}...")
            urllib.request.urlretrieve(zip_url, tmp.name)

            with zipfile.ZipFile(tmp.name, 'r') as zf:
                # Extract to temp dir first
                with tempfile.TemporaryDirectory() as tmpdir:
                    zf.extractall(tmpdir)
                    # Find extracted folder (usually repo-branch)
                    extracted = list(Path(tmpdir).iterdir())[0]
                    shutil.move(str(extracted), str(dest))

            os.unlink(tmp.name)
        return True
    except Exception as e:
        error(f"Download failed: {e}")
        return False

def create_launcher(comfy_path: Path, install_path: Path):
    """Create convenient launcher script."""
    launcher = comfy_path / "performance_lab.sh" if platform.system() != "Windows" else comfy_path / "performance_lab.bat"

    if platform.system() == "Windows":
        content = f'@echo off\npython "{install_path / "performance_lab.py"}" %*\n'
    else:
        content = f'#!/bin/bash\npython3 "{install_path / "performance_lab.py"}" "$@"\n'

    launcher.write_text(content)

    if platform.system() != "Windows":
        launcher.chmod(0o755)

    return launcher

def setup_alias():
    """Suggest shell alias."""
    shell = os.environ.get("SHELL", "")

    if "zsh" in shell:
        rc_file = "~/.zshrc"
    elif "bash" in shell:
        rc_file = "~/.bashrc"
    else:
        return None

    return rc_file

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    banner()

    # Step 1: Find ComfyUI
    info("Looking for ComfyUI installation...")
    comfy_path = find_comfyui()

    if not comfy_path:
        error("Could not find ComfyUI installation!")
        print(f"\n{C.YELLOW}Please set COMFYUI_PATH environment variable:{C.RESET}")
        print(f"  export COMFYUI_PATH=/path/to/ComfyUI")
        print(f"  python install.py")
        sys.exit(1)

    success(f"Found ComfyUI at: {comfy_path}")

    # Step 2: Check if already installed
    existing = is_already_installed(comfy_path)
    if existing:
        warn(f"Performance Lab already installed at: {existing}")
        response = input(f"\n{C.YELLOW}Update existing installation? [y/N]: {C.RESET}").strip().lower()
        if response != 'y':
            info("Installation cancelled.")
            sys.exit(0)

        # Update via git pull if it's a git repo
        if (existing / ".git").exists():
            info("Updating via git pull...")
            try:
                subprocess.run(["git", "-C", str(existing), "pull"], check=True)
                success("Updated successfully!")
                sys.exit(0)
            except subprocess.CalledProcessError:
                warn("Git pull failed, will reinstall...")

        shutil.rmtree(existing)

    # Step 3: Determine install location
    # Prefer Workflowmods if it exists, otherwise custom_nodes
    if (comfy_path / "Workflowmods").exists():
        install_path = comfy_path / "Workflowmods"
        info(f"Installing to existing Workflowmods: {install_path}")
        # Don't clone into existing dir, just ensure files are there
        if not (install_path / "performance_lab.py").exists():
            error("Workflowmods exists but missing performance_lab.py")
            sys.exit(1)
        success("Performance Lab already in Workflowmods!")
    else:
        install_path = comfy_path / "custom_nodes" / REPO_NAME
        install_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 4: Install
        info(f"Installing to: {install_path}")

        if has_git():
            info("Installing via git...")
            if not install_via_git(install_path):
                warn("Git failed, trying direct download...")
                if not install_via_download(install_path):
                    sys.exit(1)
        else:
            info("Git not found, downloading directly...")
            if not install_via_download(install_path):
                sys.exit(1)

    success("Installation complete!")

    # Step 5: Create launcher
    launcher = create_launcher(comfy_path, install_path)
    success(f"Created launcher: {launcher}")

    # Step 6: Print usage
    print(f"""
{C.CYAN}{'═' * 66}{C.RESET}

{C.GREEN}{C.BOLD}Installation successful!{C.RESET}

{C.BOLD}To run Performance Lab:{C.RESET}
  cd {install_path}
  python performance_lab.py

{C.BOLD}Or use the launcher:{C.RESET}
  {launcher}

{C.BOLD}Quick tip:{C.RESET} Add an alias to your shell:
  alias perflab='python {install_path / "performance_lab.py"}'

{C.CYAN}{'═' * 66}{C.RESET}
""")

    # Ask to run now
    response = input(f"{C.YELLOW}Run Performance Lab now? [Y/n]: {C.RESET}").strip().lower()
    if response != 'n':
        os.chdir(install_path)
        os.execvp(sys.executable, [sys.executable, "performance_lab.py"])

if __name__ == "__main__":
    main()
