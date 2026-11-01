"""Run the GLM judge (and optionally the renderer) on a rented vast.ai GPU: SSH in, start them, forward their ports.

    python vast.py key       # public key to add at vast.ai (Account > SSH Keys); created if missing
    python vast.py ssh       # interactive shell on the instance
    python vast.py setup     # upload scripts/serve_glm.sh and start GLM in the background
    python vast.py render    # upload render-service-js and start RENDER_INSTANCES copies (scripts/serve_render.sh)
    python vast.py logs      # follow the GLM log (Ctrl+C stops following, not GLM)
    python vast.py status    # GPUs, vLLM process, /v1/models answer, healthy renderers
    python vast.py tunnel    # forward GLM_PORT (and the renderer ports) here; keep it open while judging
    python vast.py stop      # stop vLLM and the renderers on the instance
    python vast.py run CMD   # run one shell command on the instance and return its exit code

Every command except `ssh` runs in batch mode: it fails instead of waiting for a password or host-key prompt, so
an agent without a terminal (e.g. a Claude Code session) can drive the instance with `run`, `status` and `tunnel`.

Settings come from .env (see .env.example). VAST_SSH is the instance's SSH command as vast.ai shows it,
e.g. "ssh -p 41234 root@ssh4.vast.ai -L 8080:localhost:8080", or the `vastai ssh-url` form
"ssh://root@ssh4.vast.ai:41234". Only OpenSSH (built into Windows 10/11) is needed.
"""

from __future__ import annotations

import io
import os
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

from arena.env import load_env

ROOT = Path(__file__).resolve().parent
REMOTE_SCRIPT = "serve_glm.sh"
REMOTE_LOG = "glm.log"
RENDER_FILES = ("src", "package.json", "package-lock.json")
STOP_RENDER = "pkill -9 -f '^node src/server.js'; pkill -9 -f \"^$HOME/.cache/puppeteer/chrome\""


def target() -> tuple[str, str, str]:
    """(user, host, port) of the instance from VAST_SSH."""
    spec = os.environ.get("VAST_SSH", "").strip()
    url = re.fullmatch(r"ssh://(?:([^@]+)@)?([^:/]+)(?::(\d+))?/?", spec)
    if url:
        return url.group(1) or "root", url.group(2), url.group(3) or "22"
    port = re.search(r"-p\s+(\d+)", spec)
    login = re.search(r"(?:^|\s)([\w.-]+)@([\w.-]+)", spec)
    if not login:
        sys.exit('Set VAST_SSH in .env to the instance SSH command, e.g. "ssh -p 41234 root@ssh4.vast.ai"')
    return login.group(1), login.group(2), port.group(1) if port else "22"


def key_path() -> Path:
    return Path(os.path.expanduser(os.environ.get("VAST_KEY") or "~/.ssh/id_ed25519"))


def ssh_options(port_flag: str, batch: bool = True) -> list[str]:
    _, _, port = target()
    options = [port_flag, port, "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
               "-o", "ConnectTimeout=15"]
    if batch:
        options += ["-o", "BatchMode=yes"]
    if key_path().is_file():
        options += ["-i", str(key_path())]
    return options


def ssh(*remote: str, tty: bool = False, extra: list[str] | None = None) -> int:
    user, host, _ = target()
    command = ["ssh", *ssh_options("-p", batch=not tty), *(extra or []), *(["-t"] if tty else []),
               f"{user}@{host}", *remote]
    return subprocess.call(command)


def remote_env() -> str:
    values = {
        "GLM_API_KEY": os.environ.get("GLM_API_KEY", "EMPTY"),
        "PORT": os.environ.get("GLM_PORT", "8002"),
        "MAX_MODEL_LEN": os.environ.get("GLM_MAX_MODEL_LEN", "32768"),
    }
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in values.items())


def render_ports() -> list[int]:
    base = int(os.environ.get("RENDER_BASE_PORT", "8101"))
    return [base + i for i in range(int(os.environ.get("RENDER_INSTANCES", "8")))]


def cmd_key() -> int:
    private = key_path()
    public = private.with_name(private.name + ".pub")
    if not private.is_file():
        private.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "duel-vast", "-f", str(private)])
    print(f"Add this public key at https://cloud.vast.ai/manage-keys/ (before creating the instance):\n")
    print(public.read_text().strip())
    return 0


def cmd_setup() -> int:
    user, host, _ = target()
    script = ROOT / "scripts" / REMOTE_SCRIPT
    if subprocess.call(["scp", *ssh_options("-P"), str(script), f"{user}@{host}:{REMOTE_SCRIPT}"]):
        return 1
    start = (
        f"pkill -f '[v]llm serve' 2>/dev/null; sed -i 's/\\r$//' {REMOTE_SCRIPT}; "
        f"{remote_env()} nohup setsid bash {REMOTE_SCRIPT} > {REMOTE_LOG} 2>&1 < /dev/null & "
        f"sleep 2; tail -n 5 {REMOTE_LOG}"
    )
    code = ssh(start)
    print("\nGLM is starting in the background. The first run builds vLLM and downloads ~20 GB (10-20 min).")
    print("Follow it with:  python vast.py logs   — ready when it prints 'Application startup complete'.")
    return code


def cmd_render() -> int:
    service = ROOT / (os.environ.get("RENDER_SERVICE_DIR") or "../AI-3D-Generation/render-service-js")
    if not (service / "src" / "server.js").is_file():
        sys.exit(f"render-service-js not found at {service}; set RENDER_SERVICE_DIR in .env")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name in RENDER_FILES:
            tar.add(service / name, arcname=name)
        tar.add(ROOT / "scripts" / "serve_render.sh", arcname="serve_render.sh")
    user, host, _ = target()
    upload = ["ssh", *ssh_options("-p"), f"{user}@{host}", "mkdir -p render && tar xzf - -C render"]
    if subprocess.run(upload, input=archive.getvalue()).returncode:
        return 1
    ports = render_ports()
    if not ports:
        sys.exit("RENDER_INSTANCES must be at least 1")
    code = ssh(f"sed -i 's/\\r$//' render/serve_render.sh; "
               f"RENDER_INSTANCES={len(ports)} RENDER_BASE_PORT={ports[0]} bash render/serve_render.sh")
    urls = ",".join(f"http://127.0.0.1:{p}" for p in ports)
    print(f"\nWith `python vast.py tunnel` open, use these renderers (RENDERER_URL in .env, or --renderer-url):\n{urls}")
    return code


def cmd_status() -> int:
    port, key = os.environ.get("GLM_PORT", "8002"), os.environ.get("GLM_API_KEY", "EMPTY")
    ports = render_ports()
    check = (
        "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; "
        "echo '--- vllm:'; pgrep -af '[v]llm serve' | head -2 || echo 'not running'; "
        f"echo '--- /v1/models:'; curl -s -m 5 -H 'Authorization: Bearer {key}' http://127.0.0.1:{port}/v1/models "
        "|| echo 'not answering yet'; echo; "
        f"echo '--- log:'; tail -n 5 {REMOTE_LOG} 2>/dev/null; "
        f"ok=0; for p in {' '.join(map(str, ports))}; do "
        "curl -s -m 3 http://127.0.0.1:$p/health | grep -q ok && ok=$((ok+1)); done; "
        f"echo \"--- renderers: $ok/{len(ports)} healthy\""
    )
    return ssh(check)


def cmd_tunnel() -> int:
    port, ports = os.environ.get("GLM_PORT", "8002"), render_ports()
    forwards = [arg for p in (port, *ports) for arg in ("-L", f"{p}:127.0.0.1:{p}")]
    print(f"GLM at http://127.0.0.1:{port}/v1 while this runs (GLM_BASE_URL in .env)", end="")
    print(f", renderers on ports {ports[0]}-{ports[-1]}" if ports else "", end=". Ctrl+C closes the tunnel.\n")
    return ssh(extra=["-N", "-o", "ExitOnForwardFailure=yes", *forwards])


COMMANDS = {
    "key": cmd_key,
    "ssh": lambda: ssh(tty=True),
    "setup": cmd_setup,
    "render": cmd_render,
    "logs": lambda: ssh(f"tail -n 100 -f {REMOTE_LOG}"),
    "status": cmd_status,
    "tunnel": cmd_tunnel,
    "stop": lambda: ssh(f"pkill -f '[v]llm serve' && echo 'vllm stopped' || echo 'vllm was not running'; {STOP_RENDER}; "
                        "echo 'renderers stopped'"),
}


def main() -> int:
    load_env(ROOT / ".env")
    if len(sys.argv) > 2 and sys.argv[1] == "run":
        return ssh(" ".join(sys.argv[2:]))
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 2
    return COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
