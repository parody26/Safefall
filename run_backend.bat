@echo off
title SafeFall - Shared Backend Server
echo ========================================================
echo   Starting SafeFall Shared Backend Server (Port 8080)
echo ========================================================
cd /d "%~dp0\safefall"
python backend/server.py %*
pause
