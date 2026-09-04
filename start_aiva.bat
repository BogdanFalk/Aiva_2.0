@echo off
title AivaConsole
rem Boots Aiva in ambient wake-word mode; she brings up her own avatar
rem renderer (OBS skipped; remove --no-obs below if you want OBS back).
cd /d "%~dp0aiva"
set AIVA_WAKE_WORD=1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set AIVA_RESTART_LOOP=1

:launch
"%~dp0.venv\Scripts\python.exe" launcher.py --no-obs
if %errorlevel% equ 42 (
    echo Restarting Aiva...
    goto launch
)
echo.
echo Aiva exited.
pause
