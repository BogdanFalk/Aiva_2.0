import os
import subprocess
import winreg
import json

def load_app_paths():
    """Load application paths from the configuration file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), 'app_paths.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print("[DEBUG] Error parsing app_paths.json")
        return {}

def get_installed_apps():
    """Get list of installed applications from Windows registry"""
    apps = []
    # Common registry paths for installed applications
    registry_paths = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
    ]
    
    for path in registry_paths:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                for i in range(0, winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            try:
                                app_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                if app_name and install_location:
                                    apps.append({
                                        "name": app_name.lower(),
                                        "path": install_location
                                    })
                            except WindowsError:
                                continue
                    except WindowsError:
                        continue
        except WindowsError:
            continue
    
    return apps

def find_app_path(app_name):
    """Find the path of an installed application"""
    apps = get_installed_apps()
    app_name = app_name.lower()
    
    # First try exact match
    for app in apps:
        if app["name"] == app_name:
            return app["path"]
    
    # Then try partial match
    for app in apps:
        if app_name in app["name"]:
            return app["path"]
    
    return None

def launch_app(app_name):
    """Launch an application by name"""
    try:
        app_name = app_name.lower()
        
        # First check the app_paths.json configuration
        app_paths = load_app_paths()
        if app_name in app_paths:
            exe_path = app_paths[app_name]
            print(f"[DEBUG] Found {app_name} in app_paths.json: {exe_path}")
            if os.path.exists(exe_path):
                # Try to run with elevation if needed
                try:
                    subprocess.Popen([exe_path])
                except PermissionError:
                    print(f"[DEBUG] Trying to run {exe_path} with elevation")
                    subprocess.Popen(['runas', '/user:Administrator', exe_path])
                return ""
            else:
                print(f"[DEBUG] Path from app_paths.json doesn't exist: {exe_path}")
        
        # Try to find the app in installed applications
        app_path = find_app_path(app_name)
        if app_path:
            # Try to find the executable in the installation directory
            for root, dirs, files in os.walk(app_path):
                for file in files:
                    if file.endswith('.exe') and not file.lower().endswith('uninstall.exe'):
                        exe_path = os.path.join(root, file)
                        print(f"[DEBUG] Found executable at: {exe_path}")
                        try:
                            subprocess.Popen([exe_path])
                        except PermissionError:
                            print(f"[DEBUG] Trying to run {exe_path} with elevation")
                            subprocess.Popen(['runas', '/user:Administrator', exe_path])
                        return ""
        
        # If not found in installed apps, try common locations
        common_paths = [
            os.path.join(os.environ['ProgramFiles'], app_name),
            os.path.join(os.environ['ProgramFiles(x86)'], app_name),
            os.path.join(os.environ['LOCALAPPDATA'], app_name),
            os.path.join(os.environ['APPDATA'], app_name)
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.exe') and not file.lower().endswith('uninstall.exe'):
                            exe_path = os.path.join(root, file)
                            print(f"[DEBUG] Found executable at: {exe_path}")
                            try:
                                subprocess.Popen([exe_path])
                            except PermissionError:
                                print(f"[DEBUG] Trying to run {exe_path} with elevation")
                                subprocess.Popen(['runas', '/user:Administrator', exe_path])
                            return ""
        
        # If still not found, try using the Windows shell command
        print(f"[DEBUG] Trying to launch {app_name} using Windows shell")
        subprocess.Popen(f'start "" "{app_name}"', shell=True)
        return ""
    except Exception as e:
        return f"Error: {str(e)}" 