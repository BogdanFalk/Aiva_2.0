import json
import asyncio
from modules.file_operations import create_file, open_file, list_directory
from modules.app_launcher import launch_app
from modules.voice import VoiceAssistant
from modules.ai_handler import AIHandler
from modules.vtube_studio import VTubeStudio
from modules.utilities import get_current_time, get_current_date, get_weather
import msvcrt  # For Windows key detection

async def perform_action(action_type, **kwargs):
    """Perform system actions based on the action type and parameters"""
    try:
        if action_type == "create_file":
            return create_file(kwargs.get('file_path'), kwargs.get('content', ''))
        
        elif action_type == "open_file":
            return open_file(kwargs.get('file_path'))
        
        elif action_type == "run_command":
            command = kwargs.get('command')
            if not command:
                return "Error: No command provided"
            import subprocess
            subprocess.run(command, shell=True, capture_output=True, text=True)
            return ""
        
        elif action_type == "list_directory":
            return list_directory(kwargs.get('path', '.'))
        
        elif action_type == "launch_app" or action_type == "launch_application":
            return launch_app(kwargs.get('app_name') or kwargs.get('name'))
        
        elif action_type == "vtube_expression":
            await vtube.trigger_expression(kwargs.get('expression'))
            return ""
        
        elif action_type == "vtube_move" or action_type == "vtube_move_model" or action_type == "vtube_move_model_position" or action_type == "move_model":
            print("Moving model to", kwargs.get('x', 0), kwargs.get('y', 0), kwargs.get('rotation', 0), kwargs.get('size', 2))
            await vtube.move_model(
                x=kwargs.get('x', 0),
                y=kwargs.get('y', 0),
                rotation=kwargs.get('rotation', 0),
                size=kwargs.get('size', 2)
            )
            return ""
        
        elif action_type == "get_time":
            return get_current_time()
        
        elif action_type == "get_date":
            return get_current_date()
        
        elif action_type == "get_weather":
            return get_weather(kwargs.get('city'), kwargs.get('day', 'today'))
        
        else:
            return f"Unknown action type: {action_type}"
    
    except Exception as e:
        return f"Error: {str(e)}"

async def main():
    global vtube
    voice = VoiceAssistant()
    ai = AIHandler()
    vtube = VTubeStudio()
    
    # Connect to VTube Studio
    await vtube.connect()
    
    print("Welcome to Aiva! You can speak or type your questions.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Press Enter to start voice input, or type your question directly.")
    print("Press Escape at any time to quit.")
    
    while True:
        # Check for Escape key before getting input
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # Escape key
                voice.speak("Goodbye!")
                await vtube.close()
                break
        
        user_input = ""
        
        if user_input.lower() in ["exit", "quit"]:
            voice.speak("Goodbye! Have a great day!")
            await vtube.close()
            break
            
        if user_input == "":
            print("Listening... (Press Escape to quit)")
            user_input = voice.get_voice_input()
            if user_input is None:
                continue
                
        reply = ai.get_response(user_input)
        
        # Check if the reply is in JSON format (contains an action)
        try:
            response_data = json.loads(reply)
            print("\n[DEBUG] Parsed response:", response_data)
            if "action" in response_data:
                print(f"[DEBUG] Performing action: {response_data['action']['type']} with params: {response_data['action']['params']}")
                # Perform the action
                action_result = await perform_action(
                    response_data["action"]["type"],
                    **response_data["action"]["params"]
                )
                print(f"[DEBUG] Action result: {action_result}")
                
                # For time, date, and weather, combine the AI's personality with the actual result
                if response_data["action"]["type"] in ["get_time", "get_date", "get_weather"]:
                    # Extract the actual data from the action result
                    if response_data["action"]["type"] == "get_weather":
                        # For weather, we want to keep the temperature and description
                        temp = action_result.split("temperature in")[1].split("°C")[0].strip()
                        desc = action_result.split("with")[1].strip().rstrip(".")
                        # Combine with AI's personality response
                        combined_response = f"{response_data['response']} {temp}°C with {desc}. Perfect weather for some cookies and anime! 🍪🌸"
                    elif response_data["action"]["type"] == "get_time":
                        # For time, we want to keep the natural time format
                        time_part = action_result.split(" in")[0]
                        period = action_result.split(" in")[1].strip()
                        # Combine with AI's personality response
                        combined_response = f"{response_data['response']} {time_part} {period}! Time for a cookie break, don't you think? 🍪"
                    elif response_data["action"]["type"] == "get_date":
                        # For date, we want to keep the full date
                        combined_response = f"{response_data['response']} {action_result}! Another beautiful day to learn new things! 🌸"
                    
                    voice.speak(combined_response)
                else:
                    # For other actions, use the AI's natural response
                    voice.speak(response_data["response"])
            else:
                # If not an action, speak the entire response
                voice.speak(reply)
        except json.JSONDecodeError:
            print("[DEBUG] Not a JSON response, treating as normal text")
            # If not JSON, it's a normal response
            voice.speak(reply)
        
        # Check for Escape key after each action
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # Escape key
                voice.speak("Goodbye!")
                await vtube.close()
                break

if __name__ == "__main__":
    asyncio.run(main()) 