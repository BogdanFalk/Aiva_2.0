@echo off
title AivaConsole
rem Boots VTube Studio and Aiva in ambient wake-word mode (OBS skipped;
rem remove --no-obs below if you want OBS back in the chain).
cd /d "%~dp0aiva"
set AIVA_WAKE_WORD=1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
"%~dp0.venv\Scripts\python.exe" launcher.py --no-obs
echo.
echo Aiva exited.
pause
