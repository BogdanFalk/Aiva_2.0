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

def launch_vtube_studio():
    """Launch VTube Studio if it's not already running"""
    if not is_process_running('VTube Studio'):
        print("Launching VTube Studio...")
        vtube_path = r"C:\Program Files (x86)\Steam\steamapps\common\VTube Studio\VTube Studio.exe"
        
        if os.path.exists(vtube_path):
            # Launch with -nosteam parameter
            subprocess.Popen([vtube_path, "-nosteam"])
            return True
        
        print("Could not find VTube Studio. Please make sure it's installed.")
        return False
    return True

def launch_obs():
    """Launch OBS if it's not already running"""
    if not is_process_running('obs64'):
        print("Launching OBS...")
        obs_dir = r"C:\Program Files\obs-studio\bin\64bit"
        obs_exe = "obs64.exe"
        obs_path = os.path.join(obs_dir, obs_exe)
        
        if os.path.exists(obs_path):
            # Set up environment variables
            env = os.environ.copy()
            # Add OBS directory to PATH
            env['PATH'] = obs_dir + os.pathsep + env['PATH']
            
            # Try to launch OBS with elevated privileges if needed
            try:
                # First try normal launch
                subprocess.Popen([obs_path], env=env)
            except PermissionError:
                print("Trying to launch OBS with elevation...")
                try:
                    # Try with elevation
                    subprocess.Popen(['runas', '/user:Administrator', obs_path], env=env)
                except Exception as e:
                    print(f"Failed to launch OBS with elevation: {str(e)}")
                    return False
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
    
    # Launch OBS
    if not launch_obs():
        print("Failed to launch OBS. Exiting...")
        return
    
    # Wait for OBS
    if not wait_for_process('obs64'):
        print("OBS did not start properly. Exiting...")
        return
    
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