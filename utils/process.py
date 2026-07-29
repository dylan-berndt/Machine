import cv2
import os
from glob import glob
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np


VIDEO_FRAME_SPREAD = 30
register_heif_opener()


def extractFrames(filePath):
    print(f"\rExtracting {filePath}", end="")
    frames = []

    count = 0

    cap = cv2.VideoCapture(filePath)
    while True:
        success, frame = cap.read()
        if not success: break

        if count % VIDEO_FRAME_SPREAD == 0:
            frames.append(frame)

        count += 1

    cap.release()

    return frames


def extractVideo(directory):
    frames = []

    mp4 = glob(os.path.join(directory, "*.mp4"))
    mov = glob(os.path.join(directory, "*.mov"))

    for filePath in (mp4 + mov)[:60]:
        frames.extend(extractFrames(filePath))

    return frames


def openImage(filePath):
    print(f"\rExtracting {filePath}", end="")
    ext = os.path.splitext(filePath)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        img = cv2.imread(filePath)
        return img

    elif ext == ".heic":
        pillowImage = Image.open(filePath)
        array = np.array(pillowImage)
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return bgr


def extractImages(directory):
    frames = []

    jpeg = glob(os.path.join(directory, "*.jpg")) + glob(os.path.join(directory, "*.jpeg"))
    heic = glob(os.path.join(directory, "*.heic"))

    for filePath in (jpeg + heic):
        frames.append(openImage(filePath))

    return frames