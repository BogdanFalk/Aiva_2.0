@echo off
title SagaConsole
rem Boots Saga (her own process, console, db and avatar). The launcher starts the house first if
rem it isn't up. Stop this window and Aiva keeps going; start it again and Saga reconnects.
cd /d "%~dp0aiva"
set AIVA_WAKE_WORD=1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set AIVA_RESTART_LOOP=1

:launch
"%~dp0.venv\Scripts\python.exe" launcher.py --no-obs --character saga
if %errorlevel% equ 42 (
    echo Restarting Saga...
    goto launch
)
echo.
echo Saga exited.
pause
