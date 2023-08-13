import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

model_path = "C:\Users\SaiDheerajPeketi\PycharmProjects\PostureCorrectness\pose_landmarker_heavy.task"

# Video
isOpen = False
while not isOpen:
    try:
        cam = cv2.VideoCapture(0)
        isOpen = True
    except Exception as e:
        print("Opening Webcam Failed")
        continue

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cam.read()
        if ret:
            # Pass the frame to the model to do the detection part
            pass
        else:
            break

    cam.release()
    cv2.destroyAllWindows()
