# Kalman Filter for Face Tracking - Beginner's Guide

## 1. Introduction to Kalman Filter
Imagine you are tracking a moving object (like a face) using a camera. The camera gives you measurements of the position $(x, y, z)$, but these measurements are noisy—they jump around a bit even if the face is still. A Kalman filter is a mathematical tool that helps you "smooth" these measurements to get a better estimate of the true position.

It works in two steps:  
1.  **Predict**: Based on where the object was a moment ago and how it was moving, guess where it is now.
2.  **Update**: Look at the new measurement from the camera. Combine your prediction with the measurement to get a refined estimate.

If you trust your prediction more (e.g., you know the object moves smoothly), you listen less to the noisy measurement. If you trust the measurement more, you listen less to your prediction.

## 2. Task 1.1: Linear Kalman Filter (`linkalman.py`)
We implemented a general `LinKalman` class.
- **State ($x$)**: The variables we want to estimate (e.g., position).
- **Covariance ($P$)**: How uncertain we are about our estimate.
- **Transition Matrix ($A$)**: How the state changes from one step to the next (physics).
- **Measurement Matrix ($H$)**: How the state relates to what we measure.
- **Process Noise ($Q$)**: Uncertainty in our physical model (e.g., wind, bumps, unknown forces).
- **Measurement Noise ($R$)**: Uncertainty in our sensor (camera noise).

## 3. Task 1.2: Face Tracking (`face_kalman.py`)
We used the `LinKalman` class to track a face.

### Simple Model (Constant Position)
- **Assumption**: The face stays in the same place ($x_t = x_{t-1}$).
- **State**: $[x, y, z]$.
- **Why it works**: Even though the face moves, if we assume it stays still but has "process noise" (it can wiggle), the filter will track it but smooth out the jitters.
- **Files**: `face_kalman.py` implements this.

### Playing with Parameters
You can adjust `Q` and `R` in `createKalman()`:

1.  **Measurement Noise ($R$)**:
    *   **What it is**: How much you trust the camera.
    *   **High $R$**: You don't trust the camera. The filter will be very slow to react to changes. The output will be very smooth but "laggy".
    *   **Low $R$**: You trust the camera perfectly. The filter will follow every jitter of the measurement. No smoothing happens.
    *   **Try this**: Increase `R` (e.g., `np.eye(3) * 100`) and move your face quickly. Notice the lag? Decrease `R` (e.g., `np.eye(3) * 1`) and notice the jitter?

2.  **Process Noise ($Q$)**:
    *   **What it is**: How much you expect the face to move unexpectedly.
    *   **High $Q$**: You expect the face to jump around. The filter will rely more on the measurement.
    *   **Low $Q$**: You expect the face to be very steady. The filter will rely more on its prediction (staying still).
    *   **Try this**: Set `Q` very low (e.g., `0.0001`). The filter will "stick" to the old position and be reluctant to move.

## 4. Extension: Velocity Model (`face_kalman_velocity.py`)
We extended the filter to track velocity as well.

- **State**: $[x, y, z, v_x, v_y, v_z]$ (Position and Velocity).
- **Model**: Position changes based on velocity ($x_t = x_{t-1} + v_{x, t-1} \cdot \Delta t$). Velocity stays constant (Constant Velocity model).
- **Improvement**:
    *   **Better Tracking**: The simple model assumes the face wants to stay still. When you move, it thinks the movement is "noise" and tries to pull it back. This causes lag.
    *   **Velocity Model**: It learns that the face is *moving*. It predicts the face will continue moving. This reduces lag significantly when tracking moving objects.
    *   **Prediction**: If the camera loses the face for a few frames, the velocity model can keep predicting where it went!

## 5. Professor's Solution (`face_kalman_professor.py`)
We also included the professor's solution, adapted to run on your machine.
- **Key Difference**: It uses different noise parameters (`R`) for the Z-axis vs X/Y axes, recognizing that depth estimation is noisier.
- **Fixes Applied**:
    - Fixed file paths for calibration data.
    - Added `USE_CALIBRATION` flag (default `False`) to prevent image distortion on non-lab cameras.

## 6. How to Run
Make sure you are in the `Exercise 4` directory:
```powershell
cd "Exercise 4"
```

1.  **Basic Linear Kalman Filter**:
    ```powershell
    python linkalman.py
    ```
2.  **Simple Face Tracking**:
    ```powershell
    python face_kalman.py
    ```
3.  **Velocity Model Extension**:
    ```powershell
    python face_kalman_velocity.py
    ```
4.  **Professor's Solution**:
    ```powershell
    python face_kalman_professor.py
    ```

## 7. Troubleshooting
- **Image Tracking**: If the tracking box is wild or not visible, try adjusting the lighting or moving closer/further.
- **Calibration**: If the image looks warped, ensure `USE_CALIBRATION = False` in the python scripts.

