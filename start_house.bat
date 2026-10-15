@echo off
title AivaHouse
rem The house: the one microphone, Deepgram, wake words, hotkeys, the floor arbiter and the
rem conversation log. Start it first; each sister (start_aiva.bat / start_saga.bat) connects to it
rem and can be started/stopped on her own while the house keeps running.
cd /d "%~dp0aiva"
set AIVA_WAKE_WORD=1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set AIVA_RESTART_LOOP=1

:launch
"%~dp0.venv\Scripts\python.exe" house.py
if %errorlevel% equ 42 (
    echo Restarting the house...
    goto launch
)
echo.
echo The house exited.
pause
