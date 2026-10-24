@echo off
title SafeFall - Monitoring Station
echo ========================================================
echo   Starting SafeFall AI Fall Detection Monitoring Station
echo ========================================================
cd /d "%~dp0\windows_monitor"
python src/main.py
pause
