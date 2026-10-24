# Fall Detection & ADL Datasets Research: SAFER-Activities, Le2i, and URFD

## 1. Executive Summary
To rigorously validate SafeFall against real-world accidental falls while suppressing false alarms from everyday Activities of Daily Living (ADLs) — such as intentional lying down on a bed, resting/sleeping, controlled sitting, and picking up items from the floor — we investigated three primary benchmark datasets: **SAFER-Activities**, **Le2i**, and the **UR Fall Detection Dataset (URFD)**.

---

## 2. Dataset Deep Dives

### 2.1 SAFER-Activities Dataset
- **Official Portal**: [https://safer-activities.github.io/](https://safer-activities.github.io/)
- **Publication**: European Conference on Computer Vision (ECCV 2026), Springer.
- **Authors**: Diwas Lamsal (KU Leuven), Pramod Wickramatilake, Jednipat Moonrinta, Mongkol Ekpanyapong, Matthew N. Dailey (Asian Institute of Technology).
- **Scale**:
  - Over 66 hours of untrimmed multi-camera video footage.
  - 85,310 annotated action instances across 30 distinct action classes.
  - 5,406 fall instances, accompanied by routine daily living activities.
  - Dedicated subset covering wheelchair users and assistive mobility transfers.
- **Data Modalities & Formats**:
  - `annotations/*.csv`: Frame-level annotations specifying exact start and end timestamps (`Action`, `Start Time`, `End Time`).
  - `pose_bboxes/`: Extracted 2D and 3D human pose keypoints derived using YOLOv8, ViTPose, and MotionAGFormer (stored in standard PYSKL / MMAction2 formats).
  - `extracted_features/`: Video backbone embeddings (CLIP, VideoMAE, DINOv3).
- **Licensing Conditions**:
  - **License**: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.
  - **Permissions**: Free to share and adapt for academic research and evaluation.
  - **Constraints**: **Non-Commercial**: Cannot be directly monetized or bundled into proprietary commercial products without explicit dual-licensing permissions from the copyright holders. **ShareAlike**: Derivatives must retain the same license. **Attribution**: Formal citation required.

### 2.2 Le2i Fall Detection Dataset
- **Origin**: *Laboratoire Electronique, Informatique et Image* (Le2i), University of Burgundy.
- **Scale**: 220–250 video sequences across 4 domestic and institutional environments: Home, Coffee Room, Office, Lecture Room.
- **Sensors**: Standard RGB video at 320x240 resolution, 25 FPS.
- **Annotations**: Frame-level ground truth text files specifying:
  - Pre-fall activity frames.
  - The exact start and end frame numbers of the fall transit/collapse phase.
  - Post-fall resting / recovery frames.
- **ADLs Covered**: Walking, standing, bending to tie shoes or pick up an object, crouching, sitting down into chairs, intentional slow lying down on rugs.
- **Licensing**: Academic research use upon request/citation.

### 2.3 UR Fall Detection Dataset (URFD)
- **Origin**: University of Rzeszow (Michal Kępski, Bogdan Kwolek).
- **Scale**: 70 video recordings (30 falls, 40 activities of daily living).
- **Sensors**: Dual Microsoft Kinect cameras (Camera 0 at floor height 0.8m, Camera 1 ceiling-mounted at 2.5m), synchronized with tri-axial inertial accelerometers.
- **Labels**: Categorized into `Fall` and `ADL` with synchronized frame index logs and total acceleration vectors ($SV_{total}$).
- **Licensing**: Open academic research license.

---

## 3. Label Mapping into SafeFall's 7 Explicit Activity States

| Benchmark Action Category (SAFER / Le2i / URFD) | SafeFall Unified State | Detection Engine Behavior |
|---|---|---|
| Walking, standing, slight shifting, gesturing, bending over | `NORMAL_MOVEMENT` | Normal continuous surveillance; no alert. |
| Sitting into chair, sofa, or armchair | `SITTING_DOWN` | Torso remains upright ($\theta < 45^\circ$); downward velocity allowed without triggering fall. |
| Standing up from bed, sofa, or floor | `GETTING_UP` | Negative downward velocity ($v_y < 0$); resets lying counters. |
| Slowly lowering to bed, sofa, or exercise rug | `INTENTIONAL_LYING_DOWN` | Smooth deceleration, low downward acceleration ($|a_y| < 0.22$), prior state is sitting/standing. **Suppresses fall trigger.** |
| Resting on bed, sleeping, motionless on sofa | `LYING_OR_SLEEPING` | Stable lying posture ($\theta > 48^\circ$) with low motion energy ($M < 0.04$). **A prolonged lying posture does NOT count as a fall.** |
| Trip, slip, backward collapse, bed rollout | `SUSPECTED_FALL` | Downward velocity spike + abrupt angular collapse into horizontal + impact. Initiates 3.0s monotonic grace countdown. |
| Landmark occlusion, $<0.35$ confidence, person partly out of frame | `UNCERTAIN` | Suppresses false alarms while logging audit notice; never attributes fall blindly. |

---

## 4. Subject-Wise Leakage Prevention
To ensure reproducible machine learning and feature benchmark evaluation without statistical over-fitting:
1. **No Mixed Splits**: All frames or sequences originating from the same physical subject or same recorded environment must remain strictly isolated within either the Training set or the Testing set (`GroupKFold` / `Leave-One-Group-Out`).
2. **Standard Evaluation Metrics**:
   - **Precision**: $\frac{TP}{TP + FP}$
   - **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$
   - **F1 Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$
   - **False Alarm Rate (FAR / hour)**: Number of false positive alarms generated per monitoring hour of ADLs.
   - **Missed Falls**: Count of ground-truth falls that failed to trigger `SUSPECTED_FALL`.
   - **Detection Latency**: Milliseconds from start of physical fall descent to state machine candidate trigger.
