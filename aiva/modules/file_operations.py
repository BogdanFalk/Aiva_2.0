import os


def resolve_path(file_path):
    """Resolve special paths like ~ and common locations.

    Guardrail: resolved paths must stay inside the user's home directory —
    the LLM only gets to touch the user's own files.
    """
    if file_path is None:
        raise ValueError("File path cannot be None")

    home_dir = os.path.expanduser("~")

    # Handle common locations for Windows
    lowered = file_path.lower()
    if lowered.startswith("desktop"):
        file_path = os.path.join(home_dir, "Desktop", file_path[len("desktop"):].lstrip("\\/"))
    elif lowered.startswith("documents"):
        file_path = os.path.join(home_dir, "Documents", file_path[len("documents"):].lstrip("\\/"))
    elif lowered.startswith("downloads"):
        file_path = os.path.join(home_dir, "Downloads", file_path[len("downloads"):].lstrip("\\/"))

    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path.replace("/", "\\"))

    if not file_path.lower().startswith(home_dir.lower()):
        raise ValueError(f"Refusing to touch a path outside the user profile: {file_path}")

    return file_path


def create_file(file_path, content=""):
    """Create a new file with specified content."""
    file_path = resolve_path(file_path)
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


def open_file(file_path):
    """Open an existing file with its default application."""
    file_path = resolve_path(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")
    os.startfile(file_path)
    return file_path


def list_directory(path="."):
    """List files in a directory."""
    path = resolve_path(path)
    return os.listdir(path)
