import subprocess
import time
import os
import sys
import psutil
import asyncio

def is_process_running(process_name):
    """Check if a process is running"""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and process_name in proc.info['name']:
            return True
    return False

def popen_unelevated(cmd, cwd=None, extra_path=None):
    """Launch an exe forcing it to run WITHOUT elevation.

    VTube Studio.exe and obs64.exe carry a RUNASADMIN compatibility flag on
    this machine, which makes plain CreateProcess fail with WinError 740
    (and would otherwise demand a UAC click on every boot). Setting
    __COMPAT_LAYER=RunAsInvoker overrides the flag for this launch only —
    neither app needs admin rights for anything Aiva does.
    """
    env = os.environ.copy()
    env["__COMPAT_LAYER"] = "RunAsInvoker"
    if extra_path:
        env["PATH"] = extra_path + os.pathsep + env["PATH"]
    return subprocess.Popen(cmd, env=env, cwd=cwd)

def launch_vtube_studio():
    """Launch VTube Studio if it's not already running"""
    if not is_process_running('VTube Studio'):
        print("Launching VTube Studio...")
        vtube_path = r"C:\Program Files (x86)\Steam\steamapps\common\VTube Studio\VTube Studio.exe"

        if os.path.exists(vtube_path):
            popen_unelevated([vtube_path, "-nosteam"])
            return True

        print("Could not find VTube Studio. Please make sure it's installed.")
        return False
    return True

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

async def main():
    """Main launcher function"""
    print("Aiva Launcher starting...")
    
    # Launch VTube Studio
    if not launch_vtube_studio():
        print("Failed to launch VTube Studio. Exiting...")
        return
    
    # Wait for VTube Studio
    if not wait_for_process('VTube Studio'):
        print("VTube Studio did not start properly. Exiting...")
        return
    
    # (The desktop overlay is started and stopped by Aiva herself in main.py.)

    # Launch OBS (optional — skip with --no-obs, and never a hard failure)
    if "--no-obs" not in sys.argv:
        if launch_obs():
            if not wait_for_process('obs64'):
                print("OBS did not start properly; continuing without it.")
        else:
            print("Could not launch OBS; continuing without it.")
    
    # Launch Aiva
    print("Starting Aiva...")
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
        
        # Import main after setting up the path
        from main import main as aiva_main
        
        # Run Aiva
        await aiva_main()
        print("Aiva is running!")
        
    except Exception as e:
        print(f"Error starting Aiva: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 