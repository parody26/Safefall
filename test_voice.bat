@echo off
title SafeFall - Live Voice & Microphone Test
echo ========================================================
echo   SafeFall Interactive Microphone & Speech Verification
echo ========================================================
cd /d "%~dp0\safefall\windows_monitor"
python test_live_voice.py
pause
