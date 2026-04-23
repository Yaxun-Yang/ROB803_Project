# Recording Protocol: 比爱心 Gesture

## Environment

- **Location**: Indoor setting with a uniform, solid-color background (e.g., plain wall)
- **Lighting**: Even, diffuse lighting. Avoid strong backlighting (do not stand in front of a window)
- **Space**: At least 2 meters of clear space between the camera and the subject

## Camera Setup

- **Device**: Smartphone camera or webcam
- **Resolution**: 1080p (1920x1080) or 720p (1280x720)
- **Frame rate**: 30 FPS (standard for most smartphone cameras)
- **Orientation**: Landscape (horizontal)
- **Position**: Mounted on a tripod or stable surface at chest height (~1.5m from ground)
- **Distance**: Subject centered in frame, approximately 2 meters from camera
- **Framing**: Full body visible — head to feet, with some margin above the head for raised arms

## Subject

- Face the camera directly (frontal view)
- Wear fitted clothing (loose sleeves or baggy clothing can confuse pose estimation)
- Choose clothing that contrasts with the background color
- Arms initially relaxed at sides

## Gesture Sequence

Total duration: approximately 10-12 seconds.

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Neutral | 2 s | Stand still, arms at sides (calibration reference) |
| 2. Raise arms | 2 s | Raise both arms upward toward head level |
| 3. Form heart | 3 s | Bring fingertips/wrists together above head to form heart shape, hold |
| 4. Lower arms | 2 s | Lower arms back to sides |
| 5. Neutral | 1 s | Stand still (end reference) |

## Output

- **File**: `input/gesture_video.mp4`
- **Codec**: H.264 (default for most smartphones)
- **Naming**: Keep the original recording; do not apply filters or post-processing
