# SafeFall Detection Specification

## 1. Overview
SafeFall is an AI-powered safety platform designed to monitor elderly or vulnerable individuals in home or assisted-care environments. Its primary goal is to reliably detect falls while actively minimizing false alarms caused by everyday movements (e.g., sitting down, resting on a sofa or bed, bending over to pick up an item, or doing floor exercises).

## 2. Landmark Detection & Geometric Feature Extraction
SafeFall supports industry-standard 33-point pose landmarks (compatible with MediaPipe Pose Landmarker and COCO/YOLO pose keypoints).

### 2.1 Key Landmarks Monitored
- Shoulders: Left Shoulder (11), Right Shoulder (12)
- Hips: Left Hip (23), Right Hip (24)
- Knees: Left Knee (25), Right Knee (26)
- Ankles: Left Ankle (27), Right Ankle (28)
- Head/Nose: Nose (0), Eyes (1, 2), Ears (7, 8)

### 2.2 Extracted Geometric Metrics
For each frame $t$:
1. **Shoulder Midpoint**: $P_{shoulder}(t) = \frac{P_{L\_shoulder} + P_{R\_shoulder}}{2}$
2. **Hip Midpoint**: $P_{hip}(t) = \frac{P_{L\_hip} + P_{R\_hip}}{2}$
3. **Knee Midpoint**: $P_{knee}(t) = \frac{P_{L\_knee} + P_{R\_knee}}{2}$
4. **Ankle Midpoint**: $P_{ankle}(t) = \frac{P_{L\_ankle} + P_{R\_ankle}}{2}$
5. **Torso Vector**: $\vec{V}_{torso} = P_{shoulder} - P_{hip} = (\Delta x, \Delta y)$
6. **Torso Angle with Vertical ($\theta_{torso}$)**:
   $$\theta_{torso} = \arctan2(|\Delta x|, |\Delta y|) \times \frac{180^\circ}{\pi}$$
   - $0^\circ \le \theta_{torso} \le 30^\circ$: Upright (Standing/Sitting)
   - $60^\circ \le \theta_{torso} \le 90^\circ$: Horizontal (Lying)
7. **Aspect Ratio of Bounding Envelope ($AR$)**:
   $$AR = \frac{\text{Width}}{\text{Height}}$$
   - Standing: $AR < 0.6$
   - Sitting: $0.6 \le AR \le 1.1$
   - Lying: $AR > 1.2$
8. **Relative Vertical Position of Hip ($Y_{hip}$)**:
   Normalized vertical coordinate of the hips within the calibrated scene frame (0.0 = top of frame, 1.0 = bottom/floor).
9. **Knee Flexion Angle ($\alpha_{knee}$)**:
   Angle between hip-knee vector and knee-ankle vector.

## 3. Posture Classification Output
SafeFall categorizes postures into four discrete states:
- **STANDING**: Upright torso ($\theta_{torso} < 35^\circ$), vertical bounding box ($AR < 0.8$), hip height elevated.
- **SITTING**: Semi-upright torso ($\theta_{torso} < 45^\circ$), bent knees ($\alpha_{knee} \in [70^\circ, 120^\circ]$), intermediate hip height.
- **LYING**: Torso predominantly horizontal ($\theta_{torso} > 55^\circ$), wide bounding box ($AR > 1.1$), hip height near floor plane.
- **UNKNOWN**: Landmarks occluded, low confidence ($< 0.45$), or ambiguous multi-person presence.

## 4. Multi-Person Policy
For single-individual safety monitoring:
- If $>1$ person is detected with high confidence in the field of view:
  - If a primary tracked bounding box is established, maintain track continuity via IoU tracking.
  - If tracking continuity is broken or identities cross ambiguously, flag scene as **AMBIGUOUS_MULTI_PERSON** and suppress fall escalation until identity resolves, logging an audit notice.
