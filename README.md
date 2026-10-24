# SafeFall — AI Fall Detection & Family Emergency Response System

SafeFall is a real-time, privacy-first computer vision and acoustic fall detection system with cloud synchronization and a modern Family Emergency Web Dashboard.

```
                      +---------------------------------------+
                      |       Hosted Cloud Backend            |
                      |   (Vercel Serverless / Render API)    |
                      +-------------------+-------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
        +---------------------------+           +---------------------------+
        |   Family Web Dashboard    |           |    Local Windows Edge     |
        |  (Mobile & Desktop Web)   |           |    AI Camera Monitor      |
        |  - Instant Push Alerts    |           |  - YOLOv8 Pose (17 pts)   |
        |  - Evidentiary Keyframe   |           |  - Biomechanical GRU      |
        |  - Voice Verification     |           |  - Acoustic Fall Alarm    |
        |  - 1-Click Acknowledge    |           |  - Two-Way Elder Speech   |
        +---------------------------+           +---------------------------+
```

---

## 🚀 Cloud Hosting (Vercel & Render)

SafeFall includes a complete, zero-dependency Cloud API and responsive Web Dashboard that can be deployed online in seconds.

### Option A: Deploy to Vercel (Recommended)

1. **Install Vercel CLI** (or push to GitHub and import into Vercel Dashboard):
   ```bash
   npm install -g vercel
   ```
2. **Deploy directly from the project root**:
   ```bash
   vercel
   ```
   Or for production:
   ```bash
   vercel --prod
   ```
3. Once deployed, Vercel gives you a live HTTPS URL (e.g., `https://safefall-app.vercel.app`).
   - Open that URL in any web browser (on your phone, tablet, or PC) to view the **Family Safety Web Dashboard**.
   - Your API endpoints are automatically available under `/api/v1/...` (e.g., `https://safefall-app.vercel.app/api/v1`).

### Option B: Deploy to Render / Railway / Docker

SafeFall also includes [`render.yaml`](./render.yaml), [`Procfile`](./Procfile), and [`Dockerfile`](./Dockerfile):
- **Render**: Connect your GitHub repository and select "Blueprint" using `render.yaml`, or create a Web Service pointing to `python safefall/backend/server.py 10000`.
- **Railway / Fly.io / Docker**:
  ```bash
  docker build -t safefall .
  docker run -p 8080:8080 safefall
  ```

---

## 📡 Connecting the Windows AI Monitor to Hosted Cloud

Once your backend is hosted online, point your local Windows AI Camera Monitor to your cloud URL:

```powershell
# In PowerShell:
$env:SAFEFALL_API_URL="https://your-safefall-app.vercel.app/api/v1"
.\run_monitor.bat
```
Or via CLI argument:
```powershell
python safefall/windows_monitor/src/main.py --api-url "https://your-safefall-app.vercel.app/api/v1"
```

When a fall is detected, the Windows station:
1. Emits the local acoustic beep-beep warning alarm.
2. Interacts with the fallen individual ("SafeFall detected a fall. Are you okay?").
3. Instantly vaults the evidentiary camera keyframe and voice response to your live hosted Vercel/Render dashboard!
4. Caregivers anywhere in the world receive the alert, hear the emergency chime, see the snapshot, and acknowledge or resolve it.

---

## 💻 Running Locally (Offline Mode)

You can run everything on your local machine with zero external cloud dependencies:

1. **Start Backend & Web Dashboard**:
   Double click `run_backend.bat` or run:
   ```powershell
   python safefall/backend/server.py 8080
   ```
   Open **`http://localhost:8080`** in any browser to see the Family Safety Dashboard.

2. **Start Windows Camera AI Monitor**:
   Double click `run_monitor.bat` or run:
   ```powershell
   python safefall/windows_monitor/src/main.py
   ```

3. **(Optional) Run Desktop Tkinter Family App**:
   Double click `run_family_app.bat` or run:
   ```powershell
   python safefall/family_app/main.py
   ```

4. **Test Live Voice Recognition**:
   Double click `test_voice.bat` or run:
   ```powershell
   python safefall/windows_monitor/test_live_voice.py
   ```

---

## 📁 Repository Structure & Cleanup Summary

The repository has been cleaned and optimized for cloud hosting:
- **Bulky 16.6 GB Raw Training Videos Removed**: The neural network features and weights are safely preserved in:
  - Extracted kinematic dataset: [`safefall/windows_monitor/data/le2i_extracted_dataset.npz`](./safefall/windows_monitor/data/le2i_extracted_dataset.npz) (1.1 MB)
  - Trained model weights: [`safefall/windows_monitor/models/fall_vs_lying_gru.pt`](./safefall/windows_monitor/models/fall_vs_lying_gru.pt) (133 KB)
  - Single lightweight YOLO model: [`safefall/windows_monitor/yolov8n-pose.pt`](./safefall/windows_monitor/yolov8n-pose.pt) (6.8 MB)
- **Total repository size**: Reduced from **16.6 GB** down to **8.2 MB**!
- **Serverless API**: [`api/index.py`](./api/index.py) runs on Vercel Python Serverless.
- **Web Dashboard**: [`public/index.html`](./public/index.html) responsive dark-mode emergency center.
- **Unit Tests**: Full test suite (`python -m unittest discover -s safefall/windows_monitor/tests -t safefall/windows_monitor`).
