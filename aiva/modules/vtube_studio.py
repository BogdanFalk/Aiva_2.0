import pyvts
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

class VTubeStudio:
    def __init__(self):
        self.vts = pyvts.vts()
        self.connected = False

    async def connect(self):
        """Connect to VTube Studio"""
        try:
            # Connect to VTube Studio
            await self.vts.connect()
            
            # Request authentication token
            await self.vts.request_authenticate_token()
            
            # Authenticate with the token
            auth_status = await self.vts.request_authenticate()
            
            if auth_status:
                self.connected = True
                print("Connected and authenticated with VTube Studio!")
            else:
                print("Failed to authenticate with VTube Studio")
                print("Please check VTube Studio for authentication request")
                
        except Exception as e:
            print(f"Error connecting to VTube Studio: {e}")
            print("Make sure VTube Studio is running and WebSocket API is enabled")

    async def trigger_expression(self, expression_name):
        """Trigger a specific expression"""
        try:
            await self.vts.trigger_expression(expression_name)
            return True
        except Exception as e:
            print(f"Error triggering expression: {e}")
            return False

    async def move_model(self, x=0, y=0, rotation=0, size=1.0):
        """Move the model to a specific position"""
        try:
            # Create the request message for model position
            request_msg = self.vts.vts_request.requestMoveModel(
                x=x,
                y=y,
                rot=rotation,
                size=size,
                relative=False,
                move_time=0.2
            )
            # Send the request
            await self.vts.request(request_msg)
            return True
        except Exception as e:
            print(f"Error moving model: {e}")
            return False

    async def set_parameter(self, parameter_name, value):
        """Set a specific parameter value"""
        try:
            await self.vts.set_parameter(parameter_name, value)
            return True
        except Exception as e:
            print(f"Error setting parameter: {e}")
            return False

    async def close(self):
        """Close the connection"""
        if self.connected:
            await self.vts.close()
            self.connected = False 