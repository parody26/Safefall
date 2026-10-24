# SafeFall Test Data & Benchmark Sequences

This directory houses synthetic and benchmark posture sequences for deterministic testing of the SafeFall temporal fall detection engine.

## Included Datasets & Test Fixtures
1. `synthetic_sequences.json`: Timestamped sequences of coordinates covering:
   - Normal standing still (5.0s)
   - Normal sitting on sofa (8.0s)
   - Slow, deliberate lying down on bed (3.5s transition time $\to$ negative for fall)
   - Sudden trip and fall forward (0.5s transition time $\to$ positive for fall)
   - Fall with immediate self-recovery (stood back up at $t=1.8$s $\to$ cancelled before alarm)
   - Sudden fall followed by immobility and silence ($10.0$s timeout $\to$ emergency published)
2. `fall_sample.mp4` (or synthetic stream generator in `CameraSource`): Video stream fixture used for visual and integration testing.
