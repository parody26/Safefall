# SafeFall Windows Setup & Run Guide

## 1. Prerequisites
- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Python**: Python 3.10–3.14 (Python 3.14 verified)
- **Hardware**: Integrated webcam or external USB camera; microphone; speakers.

## 2. Windows Monitoring Application

### 2.1 Directory Navigation
```powershell
cd c:\Users\katar\OneDrive\Desktop\SYNTAX\safefall\windows_monitor
```

### 2.2 Install Dependencies
```powershell
pip install -r requirements.txt
```
*(Dependencies include `opencv-python`, `numpy`, `SpeechRecognition`, `pyttsx3`, `requests`, `PySide6`)*

### 2.3 Running the Application
To run with live camera:
```powershell
python src/main.py
```

To run in automated test / simulation mode with synthetic postures:
```powershell
python src/main.py --simulation
```

To run with a prerecorded video sequence:
```powershell
python src/main.py --video-source ../test_data/fall_sample.mp4
```

## 3. Windows Family Safety Application
The family application runs natively on Windows via Flutter desktop or the local client.
```powershell
cd ../flutter_apps
flutter run -d windows
```
