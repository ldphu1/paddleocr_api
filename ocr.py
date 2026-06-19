import torch
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import numpy as np

class paddle_ocr:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        self.ocr = PaddleOCR(
            lang="en",
            text_detection_model_dir="model/det",
            text_recognition_model_dir="model/rec",
        )

        self.yolo = YOLO('model/yolo-pose.pt')

    def get_info(self):
        info = {
            "device": self.device,
            "yolo_model": {
                "framework": "Ultralytics YOLO",
                "model_path": getattr(self.yolo, 'ckpt_path', 'model/yolo-pose.pt'),
                "task": getattr(self.yolo, 'task', 'detect/pose'),
                "names": getattr(self.yolo, 'names', {})
            },
            "paddle_ocr": {
                "framework": "PaddleOCR",
                "language": "en",
                "use_gpu": (self.device == "cuda"),
                "det_algorithm": "DB",
                "rec_algorithm": "SVTR_LCNet"
            }
        }
        return info

    def perspective(self, image, keypoints):
        img_np = np.array(image)
        width = int(max(
            np.linalg.norm(keypoints[1] - keypoints[0]),
            np.linalg.norm(keypoints[2] - keypoints[3])
        ))

        height = int(max(
            np.linalg.norm(keypoints[3] - keypoints[0]),
            np.linalg.norm(keypoints[2] - keypoints[1])
        ))

        pts_dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(keypoints, pts_dst)

        flat_invoice = cv2.warpPerspective(img_np, matrix, (width, height))

        return flat_invoice


    def inference(self, image):
        yolo_result = self.yolo(image)

        for result in yolo_result:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy().astype(np.float32).reshape(-1, 2)

                if len(keypoints) == 4:
                    flat_invoice =  self.perspective(image, keypoints)
                    return self.ocr.ocr(flat_invoice, cls=True)

        raise ValueError("YOLO did not detect any keypoints")
