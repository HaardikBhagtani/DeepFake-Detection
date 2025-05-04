import shutil
import os
import time
from datetime import datetime
from fastapi.responses import JSONResponse
from model.predict import predict
from utils.video_utils import extractFrames

path = r"test_data/"

def get_video_prediction(videoPath):
    start_time = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    framesPath = os.path.join(path, timestamp)
    os.makedirs(framesPath, exist_ok=True)

    try:

        extractFrames(videoPath, framesPath)
        probability, label = predict(framesPath)

        processing_time = round(time.time() - start_time, 3)
        return {
            "time": processing_time,
            "label": label,
            "probability": probability
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        processing_time = round(time.time() - start_time, 3)
        return {
            "time": processing_time,
            "error": str(e)
        }

    finally:
        if os.path.exists(videoPath):
            os.remove(videoPath)

        if os.path.exists(framesPath):
            shutil.rmtree(framesPath)
