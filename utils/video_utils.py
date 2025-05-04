import os
import cv2
from tqdm import tqdm
import numpy as np

def extractFrames(videoPath, videoOutputDir, fps=2, resize=(224, 224), max_frames=30):
    os.makedirs(videoOutputDir, exist_ok=True)
    video = cv2.VideoCapture(videoPath)
    if not video.isOpened():
        return

    videofps = video.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(videofps // fps)) if videofps > fps else 1

    count = 0
    frameID = 0
    while frameID < max_frames:
        success, frame = video.read()
        if not success:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, resize)
            frameFilename = f"frame_{frameID:04d}.jpg"
            cv2.imwrite(os.path.join(videoOutputDir, frameFilename), frame)
            frameID += 1
        count += 1
    video.release()
