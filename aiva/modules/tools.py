"""Native tool-calling surface for Aiva.

Replaces the old prose-JSON action protocol: each tool is a real function
schema registered with the LLM service, so responses can never truncate or
malform an action.

run_command is deliberately two-step: it only STAGES a command; nothing
executes until the user verbally confirms and the model calls
confirm_pending_command. This gate is non-bypassable.
"""

import subprocess

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from modules import file_operations, utilities
from modules.app_launcher import launch_app as _launch_app_impl

# --- pending shell-command gate ------------------------------------------

_pending_command: str | None = None


# --- tool handlers ---------------------------------------------------------

async def launch_app(params: FunctionCallParams):
    app_name = params.arguments.get("app_name", "")
    result = _launch_app_impl(app_name)
    if result:  # non-empty means "Error: ..."
        await params.result_callback({"success": False, "detail": result})
    else:
        await params.result_callback({"success": True, "launched": app_name})


async def create_file(params: FunctionCallParams):
    try:
        path = file_operations.create_file(
            params.arguments.get("file_path"),
            params.arguments.get("content", ""),
        )
        await params.result_callback({"success": True, "created": path})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def open_file(params: FunctionCallParams):
    try:
        path = file_operations.open_file(params.arguments.get("file_path"))
        await params.result_callback({"success": True, "opened": path})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def list_directory(params: FunctionCallParams):
    try:
        files = file_operations.list_directory(params.arguments.get("path", "."))
        await params.result_callback({"success": True, "entries": files[:100]})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def run_command(params: FunctionCallParams):
    global _pending_command
    command = params.arguments.get("command", "").strip()
    if not command:
        await params.result_callback({"success": False, "error": "no command provided"})
        return
    _pending_command = command
    print(f"[COMMAND STAGED, AWAITING CONFIRMATION] {command}")
    await params.result_callback({
        "status": "pending_confirmation",
        "command": command,
        "instruction": "Tell the user what this command does and ask them to confirm. "
                       "Only after they answer, call confirm_pending_command.",
    })


async def confirm_pending_command(params: FunctionCallParams):
    global _pending_command
    command = _pending_command
    _pending_command = None

    if command is None:
        await params.result_callback({"success": False, "error": "no command is pending"})
        return

    if not params.arguments.get("confirmed", False):
        print(f"[COMMAND DISCARDED] {command}")
        await params.result_callback({"success": True, "status": "discarded", "command": command})
        return

    print(f"[COMMAND EXECUTING] {command}")
    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        await params.result_callback({
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-1000:],
            "stderr": completed.stderr[-1000:],
        })
    except subprocess.TimeoutExpired:
        await params.result_callback({"success": False, "error": "command timed out after 30s"})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def get_weather(params: FunctionCallParams):
    result = utilities.get_weather(
        params.arguments.get("city"),
        params.arguments.get("day", "today"),
    )
    await params.result_callback(result)


async def get_datetime(params: FunctionCallParams):
    await params.result_callback({
        "date": utilities.get_current_date(),
        "spoken_time": utilities.get_current_time(),
    })


def make_vtube_handlers(vtube):
    """VTube handlers close over the shared VTubeStudio instance."""

    async def vtube_expression(params: FunctionCallParams):
        ok = await vtube.trigger_expression(params.arguments.get("expression", ""))
        await params.result_callback({"success": ok})

    async def vtube_move(params: FunctionCallParams):
        ok = await vtube.move_model(
            x=params.arguments.get("x", 0),
            y=params.arguments.get("y", 0),
            rotation=params.arguments.get("rotation", 0),
            size=params.arguments.get("size", 1.0),
        )
        await params.result_callback({"success": ok})

    return vtube_expression, vtube_move


# --- schemas ---------------------------------------------------------------

TOOL_SCHEMAS = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="launch_app",
        description="Launch an application on the user's Windows PC by name, e.g. 'steam', 'notepad', 'chrome'.",
        properties={"app_name": {"type": "string", "description": "Name of the application"}},
        required=["app_name"],
    ),
    FunctionSchema(
        name="create_file",
        description="Create a text file. Paths may start with 'desktop', 'documents' or 'downloads', e.g. 'desktop\\notes.txt'.",
        properties={
            "file_path": {"type": "string", "description": "Where to create the file"},
            "content": {"type": "string", "description": "Text content of the file"},
        },
        required=["file_path"],
    ),
    FunctionSchema(
        name="open_file",
        description="Open an existing file with its default application.",
        properties={"file_path": {"type": "string", "description": "Path of the file to open"}},
        required=["file_path"],
    ),
    FunctionSchema(
        name="list_directory",
        description="List the files in a directory.",
        properties={"path": {"type": "string", "description": "Directory path, e.g. 'desktop'"}},
        required=[],
    ),
    FunctionSchema(
        name="run_command",
        description="Stage a Windows shell command for execution. It does NOT run yet: the user must confirm first.",
        properties={"command": {"type": "string", "description": "The shell command to stage"}},
        required=["command"],
    ),
    FunctionSchema(
        name="confirm_pending_command",
        description="Execute or discard the staged shell command, according to the user's verbal answer.",
        properties={"confirmed": {"type": "boolean", "description": "true if the user said yes"}},
        required=["confirmed"],
    ),
    FunctionSchema(
        name="get_weather",
        description="Get current weather or tomorrow's forecast for a city.",
        properties={
            "city": {"type": "string", "description": "City name; omit for the user's default city"},
            "day": {"type": "string", "enum": ["today", "tomorrow"]},
        },
        required=[],
    ),
    FunctionSchema(
        name="get_datetime",
        description="Get the current date and time.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="vtube_expression",
        description="Trigger one of your avatar's expressions, e.g. 'happy', 'surprised', 'angry'.",
        properties={"expression": {"type": "string", "description": "Expression name"}},
        required=["expression"],
    ),
    FunctionSchema(
        name="vtube_move",
        description="Move your avatar model on screen.",
        properties={
            "x": {"type": "number", "description": "Horizontal position, roughly -1 to 1"},
            "y": {"type": "number", "description": "Vertical position, roughly -1 to 1"},
            "rotation": {"type": "number", "description": "Rotation in degrees"},
            "size": {"type": "number", "description": "Model size/zoom"},
        },
        required=[],
    ),
])


def register_tools(llm, vtube):
    """Register every tool handler on the LLM service."""
    vtube_expression, vtube_move = make_vtube_handlers(vtube)

    llm.register_function("launch_app", launch_app)
    llm.register_function("create_file", create_file)
    llm.register_function("open_file", open_file)
    llm.register_function("list_directory", list_directory)
    llm.register_function("run_command", run_command)
    llm.register_function("confirm_pending_command", confirm_pending_command)
    llm.register_function("get_weather", get_weather)
    llm.register_function("get_datetime", get_datetime)
    llm.register_function("vtube_expression", vtube_expression)
    llm.register_function("vtube_move", vtube_move)
