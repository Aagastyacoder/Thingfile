#!/usr/bin/env python3
"""Start a fresh ~/BlindSight checkout as an HTTP gateway for the ESP32.

First run installs prerequisites and downloads an ONNX model; internet, free
storage, and (if system packages are missing) sudo permission are required.
Later runs reuse the installed environment and model. Ctrl+C stops the server.

Usage:
  python3 ~/blindsight_pi_start.py [path/to/BlindSight]
  python3 ~/blindsight_pi_start.py --install-service [path/to/BlindSight]
  python3 ~/blindsight_pi_start.py --setup-exhibition

The service starts at boot. View its terminal output with:
  journalctl -u blindsight-gateway -f
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import uuid
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import pwd
import urllib.error
import urllib.request
import venv

MODEL_URL = ("https://huggingface.co/SpotLab/YOLOv8Detection/resolve/"
             "3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx")
MODEL_SHA256 = "dd48a79dd7fec8ca25fde4eca742ff7bca23b27e2e903eb23bc1d9f83a459bd2"
PACKAGES = ["Flask>=3.0,<4", "Pillow>=10", "numpy>=1.24", "onnxruntime>=1.16",
            "requests>=2.31", "python-dotenv>=1.0"]


def say(message: str) -> None:
    print(message, flush=True)


def find_repo(value: str | None) -> Path:
    candidates = [Path(value).expanduser()] if value else [
        Path.home() / name for name in ("BlindSight", "Blindsight", "blindsight")
    ]
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "edge_server" / "server.py").is_file() and (root / "edge_server" / "vision_ai.py").is_file():
            return root
    raise RuntimeError("BlindSight repo not found. Clone it into ~/BlindSight, or pass its path as an argument.")


def install_system_packages(packages: list[str]) -> None:
    if shutil.which("apt-get") is None:
        raise RuntimeError("Automatic system setup needs Raspberry Pi OS/Debian (apt-get not found).")
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    if prefix and not shutil.which("sudo"):
        raise RuntimeError("sudo is needed to install " + ", ".join(packages))
    say("Installing system packages: " + ", ".join(packages) + " (sudo may ask for your password)")
    subprocess.run(prefix + ["apt-get", "update"], check=True)
    subprocess.run(prefix + ["apt-get", "install", "-y"] + packages, check=True)


def prepare_environment(repo: Path) -> Path:
    if not shutil.which("espeak-ng"):
        install_system_packages(["espeak-ng"])
    env_dir = repo / ".venv-esp-gateway"
    python = env_dir / "bin" / "python"
    if not python.is_file():
        say("Creating a private Python environment (first run)...")
        try:
            venv.EnvBuilder(with_pip=True).create(env_dir)
        except (subprocess.CalledProcessError, OSError, RuntimeError):
            say("Python venv support is missing; trying to install it...")
            install_system_packages(["python3-venv"])
            shutil.rmtree(env_dir, ignore_errors=True)
            venv.EnvBuilder(with_pip=True).create(env_dir)
    check = "import flask, PIL, numpy, onnxruntime, requests, dotenv"
    if subprocess.run([str(python), "-c", check], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        say("Installing server dependencies (first run; this can take a while)...")
        subprocess.run([str(python), "-m", "pip", "install", *PACKAGES], check=True)
    return python


def prepare_model(repo: Path) -> Path | None:
    models = repo / "edge_server" / "models"
    for name in ("yolov8n.onnx", "yolov8n_int8.onnx"):
        path = models / name
        if path.is_file() and path.stat().st_size > 0:
            say(f"Using existing vision model: {path}")
            return path
    models.mkdir(parents=True, exist_ok=True)
    target = models / "yolov8n.onnx"
    temporary = models / "yolov8n.onnx.download"
    say("No model found. Downloading a pinned 640x640 YOLOv8n ONNX model (about 13 MB)...")
    say("Source: SpotLab/YOLOv8Detection on Hugging Face; SHA-256 will be checked.")
    try:
        digest = hashlib.sha256()
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "BlindSight-Pi-Setup/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Model checksum did not match; refusing the download")
        temporary.replace(target)
        say("Vision model downloaded and checksum verified.")
        return target
    except (OSError, ValueError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        say(f"WARNING: model download failed: {exc}")
        say("Server will start, but local vision will NOT work until a valid model is installed.")
        say(f"Put a compatible YOLOv8n ONNX file in {target}, then restart this script.")
        return None


def ipv4_addresses() -> list[str]:
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True)
        found = []
        for item in result.stdout.split():
            try:
                socket.inet_pton(socket.AF_INET, item)
                if not item.startswith("127.") and item not in found:
                    found.append(item)
            except OSError:
                pass
        return found
    except (OSError, subprocess.CalledProcessError):
        return []


def show_addresses() -> None:
    addresses = ipv4_addresses()
    if not addresses:
        say("[Network] No LAN IPv4 address yet. Connect the Pi to Wi-Fi or Ethernet.")
    else:
        say("[Network] Current Pi IPv4 address(es): " + ", ".join(addresses))
        for ip in addresses:
            say(f"          ESP URL: http://{ip}:5000    Health: http://{ip}:5000/health")
        if len(addresses) > 1:
            say("          Use the address on the network shared with the ESP32.")


def serve(repo: Path, model: str) -> None:
    """Run inside the private venv; leave the checked-out source files unchanged."""
    sys.path.insert(0, str(repo / "edge_server"))
    import server
    from flask import has_request_context, request

    # Repo .env files are loaded on import and may contain blank VISION_COMMAND.
    # Explicitly choose this venv's interpreter for the vision subprocess.
    from dataclasses import replace
    script = repo / "edge_server" / "vision_ai.py"
    settings = replace(server.load_settings(), vision_script=script,
                       vision_command=[sys.executable, str(script)])
    if model:
        os.environ["BLINDSIGHT_MODEL_PATH"] = model
    else:
        os.environ.pop("BLINDSIGHT_MODEL_PATH", None)

    # Current firmware's playWavStream requires WAV, while the upstream server
    # prefers MP3 if edge-tts is present. Restrict only /analyse to espeak WAV.
    original = server.VisionService.synthesize_reply

    def esp_compatible_reply(self, text, voice, out_stem):
        if has_request_context() and request.path == "/analyse":
            wav = out_stem.with_suffix(".wav")
            try:
                # The /audio route checks for MP3 first; remove stale MP3 for
                # the same image hash before returning the new WAV.
                out_stem.with_suffix(".mp3").unlink(missing_ok=True)
                self.speak(text, wav)
                return wav
            except (OSError, subprocess.SubprocessError):
                return None
        return original(self, text, voice, out_stem)

    server.VisionService.synthesize_reply = esp_compatible_reply
    app = server.create_app(settings)

    @app.before_request
    def record_first_analysis_request():
        # The main firmware sends POST /analyse. This records a request, not
        # authenticated ESP identity or proof of successful analysis.
        flag = os.environ.get("BLINDSIGHT_FIRST_REQUEST_FILE")
        if flag and request.method == "POST" and request.path == "/analyse":
            try:
                Path(flag).write_text((request.remote_addr or "unknown")[:100], encoding="utf-8")
            except OSError as exc:
                say(f"Could not record incoming request: {exc}")

    # ESP firmware parses compact JSON and expects plain HTTP on port 5000.
    app.json.compact = True
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)


def check_port_available() -> None:
    # Fail early rather than claim readiness if another process owns the port.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("0.0.0.0", 5000))
    except OSError as exc:
        raise RuntimeError(f"Cannot use port 5000 ({exc}). Stop the other server first.") from exc
    finally:
        probe.close()


def run(repo: Path) -> int:
    check_port_available()
    say(f"BlindSight repository: {repo}")
    say("Preparing the Pi. The first run needs internet and may take several minutes.")
    show_addresses()
    python = prepare_environment(repo)
    model = prepare_model(repo)
    command = [str(python), "-u", str(Path(__file__).resolve()), "--serve", str(repo), str(model or "")]
    say("Starting HTTP server on port 5000...")
    event_dir = tempfile.TemporaryDirectory(prefix="blindsight-request-")
    flag = Path(event_dir.name) / "first-request"
    child_env = os.environ.copy()
    child_env["BLINDSIGHT_FIRST_REQUEST_FILE"] = str(flag)
    try:
        proc = subprocess.Popen(command, env=child_env)
    except BaseException:
        event_dir.cleanup()
        raise
    try:
        ready = False
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited during startup (exit code {proc.returncode}).")
            try:
                with urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=1):
                    ready = True
            except urllib.error.HTTPError as exc:
                # /health returns 503 when degraded, but the server is up.
                if exc.code == 503:
                    ready = True
                    say("[Health] Server is up but reports degraded status. Check /health for details.")
            except (OSError, urllib.error.URLError):
                time.sleep(0.5)
            if ready:
                break
        if not ready:
            raise RuntimeError("Server did not respond to /health in time. Check the output above.")
        say("\nServer is running and waiting for ESP32 HTTP requests.")
        say("Watch the server logs below for POST /analyse. Press Ctrl+C to stop.\n")
        show_addresses()
        next_ip = time.monotonic() + 10
        received = False
        while proc.poll() is None:
            time.sleep(1)
            if not received and flag.is_file():
                try:
                    source = flag.read_text(encoding="utf-8").strip() or "unknown"
                except OSError:
                    source = "unknown"
                say(f"[Connection] First POST /analyse received from {source}.")
                say("[Connection] Repeated IP display stopped. Check the HTTP status to confirm analysis succeeded.")
                received = True
            if not received and time.monotonic() >= next_ip:
                show_addresses()
                next_ip = time.monotonic() + 10
        say(f"Server exited (code {proc.returncode}).")
        return proc.returncode or 0
    except KeyboardInterrupt:
        say("\nStopping BlindSight...")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        event_dir.cleanup()


def systemd_quote(value: str) -> str:
    """Quote a single systemd argument and disable specifier expansion."""
    if "\n" in value or "\r" in value or "\x00" in value:
        raise RuntimeError("Paths containing line breaks cannot be used in the systemd service.")
    return '"' + value.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"') + '"'


def install_service(repo: Path) -> int:
    if os.geteuid() == 0:
        raise RuntimeError("Run --install-service as your normal Pi user, without sudo; the script will ask for sudo when needed.")
    if not shutil.which("systemctl") or not shutil.which("sudo"):
        raise RuntimeError("Automatic boot setup requires systemd, systemctl and sudo.")
    check_port_available()
    account = pwd.getpwuid(os.getuid())
    script = Path(__file__).resolve()
    interpreter = Path(sys.executable).resolve()
    say(f"Setting up BlindSight for user {account.pw_name}...")
    say("The first run needs internet and may take several minutes.")
    show_addresses()
    prepare_environment(repo)
    model = prepare_model(repo)
    if model is None:
        say("WARNING: Installing startup service without a model. Vision will not work until a model is available.")

    unit = f'''[Unit]
Description=BlindSight ESP32 HTTP gateway
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User={account.pw_name}
WorkingDirectory={str(repo).replace('%', '%%')}
Environment=PYTHONUNBUFFERED=1
ExecStart={systemd_quote(str(interpreter))} -u {systemd_quote(str(script))} {systemd_quote(str(repo))}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="blindsight-", suffix=".service", delete=False) as file:
            temporary = file.name
            file.write(unit)
        say("Installing and enabling the boot service (sudo may ask for your password)...")
        subprocess.run(["sudo", "install", "-m", "644", temporary,
                        "/etc/systemd/system/blindsight-gateway.service"], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "--now", "blindsight-gateway.service"], check=True)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    say("Boot service enabled and started. It starts automatically even without logging in.")
    say("View the IP and server logs: journalctl -u blindsight-gateway -f")
    say("Check status: systemctl status blindsight-gateway --no-pager")
    say("Do not run the script manually at the same time: both would use port 5000.")
    return 0


def setup_exhibition() -> int:
    """Install two Wi-Fi profiles and a desktop log window without replacing the boot unit."""
    if os.geteuid() == 0:
        raise RuntimeError("Run --setup-exhibition as your regular Pi user, not with sudo.")
    if not shutil.which("nmcli") or not shutil.which("sudo"):
        raise RuntimeError("Wi-Fi setup requires NetworkManager (nmcli) and sudo. No Wi-Fi changes were made.")
    result = subprocess.run(["nmcli", "-t", "-f", "RUNNING", "general"], capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != "running":
        raise RuntimeError("NetworkManager is not running; this Wi-Fi setup cannot configure your system.")
    target_dir = Path("/etc/NetworkManager/system-connections")
    if not target_dir.is_dir():
        raise RuntimeError("NetworkManager connection directory not found; no Wi-Fi changes were made.")
    paths = [target_dir / "blindsight-gateway-primary.nmconnection",
             target_dir / "blindsight-gateway-fallback.nmconnection"]
    say("Checking existing NetworkManager profiles (sudo may ask for your password)...")
    subprocess.run(["sudo", "-v"], check=True)
    for path in paths:
        check = subprocess.run(["sudo", "test", "-e", str(path)], check=False)
        if check.returncode == 0:
            raise RuntimeError("BlindSight Wi-Fi profiles already exist. Not overwriting them; ask for help changing them.")
        if check.returncode != 1:
            raise RuntimeError("Could not safely check existing NetworkManager profiles; no changes made.")
    say('Setting up Wi-Fi for the case-sensitive SSID "BlindSight".')
    say("Enter each hotspot password. Nothing will appear while you type.")
    passwords = [getpass.getpass("First password: "), getpass.getpass("Fallback password: ")]
    for password in passwords:
        if not (8 <= len(password) <= 63) or any(c in password for c in "\r\n\\"):
            raise RuntimeError("Passwords must be 8–63 characters without newlines or backslashes.")
    if passwords[0] == passwords[1]:
        raise RuntimeError("The two passwords are identical; no profiles were installed.")
    say("Installing two protected NetworkManager profiles (sudo may ask for your password)...")
    for index, (path, password) in enumerate(zip(paths, passwords)):
        profile = f'''[connection]
id=BlindSight-Gateway-{'primary' if index == 0 else 'fallback'}
uuid={uuid.uuid4()}
type=wifi
autoconnect=true
autoconnect-priority={100 if index == 0 else 90}
autoconnect-retries=1

[wifi]
mode=infrastructure
ssid=BlindSight

[wifi-security]
key-mgmt=wpa-psk
psk={password}

[ipv4]
method=auto

[ipv6]
method=auto
'''
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="blindsight-wifi-", delete=False) as f:
                temporary = f.name
                f.write(profile)
            subprocess.run(["sudo", "install", "-m", "600", temporary, str(path)], check=True)
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
    subprocess.run(["sudo", "nmcli", "connection", "reload"], check=True)
    say("Wi-Fi profiles installed. NetworkManager chooses between them when connecting; test both hotspot passwords.")
    say("If another Wi-Fi network is already connected, the Pi may stay on it until reconnect or reboot.")

    if not shutil.which("lxterminal"):
        install_system_packages(["lxterminal"])
    desktop_dir = Path.home() / ".config" / "autostart"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    desktop_file = desktop_dir / "blindsight-monitor.desktop"
    desktop_file.write_text('''[Desktop Entry]
Type=Application
Name=BlindSight Monitor
Exec=lxterminal --title=BlindSight -e journalctl -u blindsight-gateway -n 40 -f
Terminal=false
X-GNOME-Autostart-enabled=true
''', encoding="utf-8")
    say(f"Desktop log window configured: {desktop_file}")
    say("The window opens only after graphical login; use Raspberry Pi Desktop Autologin if desired.")
    say("The server service starts separately at boot, even without graphical login.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", help="repo path (default: ~/BlindSight)")
    parser.add_argument("extra", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--install-service", action="store_true", help="install dependencies and start automatically at boot")
    parser.add_argument("--setup-exhibition", action="store_true", help="set up hotspot profiles and desktop monitor")
    args = parser.parse_args()
    try:
        if args.setup_exhibition:
            return setup_exhibition()
        repo = find_repo(args.repo)
        if args.serve:
            serve(repo, args.extra or "")
            return 0
        if args.install_service:
            return install_service(repo)
        return run(repo)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        say(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
