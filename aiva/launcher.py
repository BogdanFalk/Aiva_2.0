"""Launcher: optionally boots OBS, makes sure the HOUSE is running, then runs one sister.

    python launcher.py [--no-obs] [--character aiva|saga]

The house (house.py) owns the microphone, Deepgram, wake words, hotkeys and the floor; each sister
is her own process that connects to it. If the house isn't up yet it is started in its own console
window and we wait for it to publish aiva/house_state.json before the sister boots.
"""
import asyncio
import os
import socket
import subprocess
import sys
import time

import psutil


def is_process_running(process_name):
    """Check if a process is running"""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and process_name in proc.info['name']:
            return True
    return False


def popen_unelevated(cmd, cwd=None, extra_path=None):
    """Launch an exe forcing it to run WITHOUT elevation.

    obs64.exe carries a RUNASADMIN compatibility flag on this machine, which
    makes plain CreateProcess fail with WinError 740 (and would otherwise
    demand a UAC click on every boot). Setting __COMPAT_LAYER=RunAsInvoker
    overrides the flag for this launch only — OBS doesn't need admin rights
    for anything Aiva does.
    """
    env = os.environ.copy()
    env["__COMPAT_LAYER"] = "RunAsInvoker"
    if extra_path:
        env["PATH"] = extra_path + os.pathsep + env["PATH"]
    return subprocess.Popen(cmd, env=env, cwd=cwd)


def launch_obs():
    """Launch OBS if it's not already running"""
    if not is_process_running('obs64'):
        print("Launching OBS...")
        obs_dir = r"C:\Program Files\obs-studio\bin\64bit"
        obs_path = os.path.join(obs_dir, "obs64.exe")

        if os.path.exists(obs_path):
            # OBS wants to be started from its own bin directory
            popen_unelevated([obs_path], cwd=obs_dir, extra_path=obs_dir)
            return True

        print("Could not find OBS. Please make sure it's installed.")
        return False
    return True


def wait_for_process(process_name, max_wait=60):
    """Wait for a process to be ready"""
    print(f"Waiting for {process_name} to be ready...")
    start_time = time.time()

    while not is_process_running(process_name):
        if time.time() - start_time > max_wait:
            print(f"Timed out waiting for {process_name} to start.")
            return False
        time.sleep(1)

    # Give the process a moment to fully initialize
    time.sleep(5)
    return True


# --- the house ---------------------------------------------------------------------------------

def _house_alive(here):
    """True if aiva/house_state.json names a live pid whose port answers."""
    try:
        from modules.house_proto import read_state
    except Exception:
        return False
    st = read_state()
    if not st or not st.get("pid") or not st.get("port"):
        return False
    if not psutil.pid_exists(int(st["pid"])):
        return False
    try:
        with socket.create_connection(("127.0.0.1", int(st["port"])), timeout=0.5):
            return True
    except OSError:
        return False


def ensure_house(here, max_wait=40):
    """Start house.py in its own console window if it isn't running, and wait for it."""
    if _house_alive(here):
        print("House is up.")
        return True
    print("Starting the house...")
    env = os.environ.copy()
    env.setdefault("AIVA_WAKE_WORD", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    bat = os.path.join(os.path.dirname(here), "start_house.bat")
    if os.path.isfile(bat):
        # the .bat owns the console title ("AivaHouse") and the exit-42 restart loop — without it a
        # voice "restart the house" would just end the house for good
        subprocess.Popen(["cmd", "/c", bat], cwd=os.path.dirname(bat), env=env,
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen([sys.executable, "house.py"], cwd=here, env=env,
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    start = time.time()
    while time.time() - start < max_wait:
        if _house_alive(here):
            print("House is up.")
            return True
        time.sleep(1)
    print("The house did not come up in time; she will keep trying to connect to it.")
    return False


def character_from_argv(argv):
    if "--character" in argv:
        i = argv.index("--character")
        if i + 1 < len(argv):
            return argv[i + 1].strip().lower()
    return "aiva"


async def main():
    """Main launcher function"""
    print("Aiva Launcher starting...")

    # (The avatar renderer is started/adopted by each sister herself in main.py.)

    # Launch OBS (optional — skip with --no-obs, and never a hard failure)
    if "--no-obs" not in sys.argv:
        if launch_obs():
            if not wait_for_process('obs64'):
                print("OBS did not start properly; continuing without it.")
        else:
            print("Could not launch OBS; continuing without it.")

    slug = character_from_argv(sys.argv)
    print(f"Starting {slug.capitalize()}...")
    try:
        # Get the directory where the executable is located
        if getattr(sys, 'frozen', False):
            # If running as a compiled executable
            application_path = os.path.dirname(sys.executable)
            # Add the application path to Python path
            sys.path.append(application_path)
        else:
            # If running as a script
            application_path = os.path.dirname(os.path.abspath(__file__))

        # Change to the application directory
        os.chdir(application_path)
        if application_path not in sys.path:
            sys.path.insert(0, application_path)

        ensure_house(application_path)

        # Import main after setting up the path
        from main import main as aiva_main

        # Run the sister
        await aiva_main()
        print(f"{slug.capitalize()} is running!")

    except Exception as e:
        print(f"Error starting {slug.capitalize()}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
