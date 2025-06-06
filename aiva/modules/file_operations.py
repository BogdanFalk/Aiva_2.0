import os
from pathlib import Path

def resolve_path(file_path):
    print(f"Resolving path: {file_path}")

    """Resolve special paths like ~ and common locations"""
    if file_path is None:
        raise ValueError("File path cannot be None")
    
    # Get the current user's home directory
    home_dir = os.path.expanduser("~")
    
    # Log the home directory
    print(f"Home directory: {home_dir}")

    # Handle common locations for Windows
    if "desktop" in file_path.lower():
        desktop = os.path.join(home_dir, "Desktop")
        print(f"Desktop directory: {desktop}")
        file_path = file_path.replace("desktop", desktop, 1)
        print(f"File path after replacing desktop: {file_path}")
    elif "documents" in file_path.lower():
        documents = os.path.join(home_dir, "Documents")
        file_path = file_path.replace("documents", documents, 1)
    elif "downloads" in file_path.lower():
        downloads = os.path.join(home_dir, "Downloads")
        file_path = file_path.replace("downloads", downloads, 1)
    
    # Convert forward slashes to backslashes for Windows
    file_path = file_path.replace('/', '\\')
    
    # Log the resolved path
    print(f"Resolved path: {file_path}")

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    return file_path

def create_file(file_path, content):
    """Create a new file with specified content"""
    file_path = resolve_path(file_path)
    with open(file_path, 'w') as f:
        f.write(content)
    return ""

def open_file(file_path):
    """Open an existing file"""
    file_path = resolve_path(file_path)
    os.startfile(file_path)
    return ""

def list_directory(path='.'):
    """List files in a directory"""
    path = resolve_path(path)
    files = os.listdir(path)
    return f"Here are the files: {', '.join(files)}" 