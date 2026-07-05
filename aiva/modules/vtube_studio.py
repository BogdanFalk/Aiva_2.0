import pyvts


class VTubeStudio:
    def __init__(self):
        self.vts = pyvts.vts()
        self.connected = False

    async def connect(self):
        """Connect to VTube Studio"""
        try:
            await self.vts.connect()
            await self.vts.request_authenticate_token()
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

    async def ensure_connected(self):
        """Reconnect once if the connection was never made or has dropped."""
        if not self.connected:
            # Fresh client: a failed/closed pyvts session can't be reused
            self.vts = pyvts.vts()
            await self.connect()
        return self.connected

    async def get_hotkeys(self):
        """List the current model's hotkeys (expressions, props, animations)."""
        try:
            if not await self.ensure_connected():
                return []
            response = await self.vts.request(self.vts.vts_request.requestHotKeyList())
            self._hotkeys = response["data"].get("availableHotkeys", [])
            return self._hotkeys
        except Exception as e:
            print(f"Error listing hotkeys: {e}")
            self.connected = False
            return []

    async def trigger_hotkey(self, name):
        """Trigger a hotkey by (fuzzy) name. Most are toggles."""
        try:
            if not await self.ensure_connected():
                return False
            hotkeys = getattr(self, "_hotkeys", None) or await self.get_hotkeys()
            wanted = name.strip().casefold()
            match = next((h for h in hotkeys if h["name"].casefold() == wanted), None) \
                or next((h for h in hotkeys if wanted in h["name"].casefold()), None)
            if not match:
                print(f"No hotkey matching {name!r}")
                return False
            await self.vts.request(
                self.vts.vts_request.requestTriggerHotKey(match["hotkeyID"])
            )
            return True
        except Exception as e:
            print(f"Error triggering hotkey: {e}")
            self.connected = False
            return False

    async def move_model(self, x=0, y=0, rotation=0, size=0):
        """Move the model to a specific position"""
        try:
            if not await self.ensure_connected():
                return False
            request_msg = self.vts.vts_request.requestMoveModel(
                x=x,
                y=y,
                rot=rotation,
                size=size,
                relative=False,
                move_time=0.2
            )
            await self.vts.request(request_msg)
            return True
        except Exception as e:
            print(f"Error moving model: {e}")
            self.connected = False
            return False

    async def set_parameter(self, parameter_name, value):
        """Set a specific parameter value"""
        try:
            if not await self.ensure_connected():
                return False
            await self.vts.set_parameter(parameter_name, value)
            return True
        except Exception as e:
            print(f"Error setting parameter: {e}")
            self.connected = False
            return False

    async def close(self):
        """Close the connection"""
        if self.connected:
            try:
                await self.vts.close()
            except Exception:
                pass
            self.connected = False
