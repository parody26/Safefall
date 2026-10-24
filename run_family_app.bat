@echo off
title SafeFall - Family Safety Application
echo ========================================================
echo   Starting SafeFall Family Safety & Emergency Response
echo ========================================================
cd /d "%~dp0\safefall"
python family_app/main.py %*
pause
