from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uuid
import time
import uvicorn
import cv2
import numpy as np
import  traceback
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from ocr import paddle_ocr

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading mdoels")
    app_state["model"] = paddle_ocr()
    print("Loading success")
    # app_state.get("model").warmup()
    print("Warm up success")
    yield
    app_state.clear()

app = FastAPI(title="paddleocr", lifespan=lifespan)

@app.get("/health", status_code=200)
async def health_check():
    model = app_state.get("model")

    if not model:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "Models are not initialized yet or loading failed."
            }
        )

    try:
        if hasattr(model, 'yolo') and hasattr(model, 'ocr'):
            return {
                "status": "healthy",
                "device": model.device,
                "message": "System is ready to inference"
            }
        else:
            raise Exception("Model components are missing")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "reason": f"Model health check failed: {str(e)}"
            }
        )


@app.get("/model-info", status_code=200)
async def get_model_info():
    model = app_state.get("model")
    if not model:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "detail": "Models are not initialized yet."
            }
        )

    try:
        model_details = model.get_info()

        return {
            "status": "success",
            "data": model_details
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Failed to retrieve model info: {str(e)}"
            }
        )

async def pre_img(image: UploadFile = File(...)):
    image_bytes = await image.read()

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image is larger than 5MB")

    nparr = np.frombuffer(image_bytes, np.uint8)

    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image_cv

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    req_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        model = app_state.get("model")

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not image.")

        image = await pre_img(file)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        result = await run_in_threadpool(model.inference, image)

        if image is None:
            raise HTTPException(status_code=400, detail="Image not found")

        ocr_lines = []
        if result and result[0]:
            for line in result[0]:
                text_content = line[1][0]
                confidence_score = float(line[1][1])

                ocr_lines.append({
                    "text": text_content,
                    "confidence": round(confidence_score, 2)
                })

        inference_time_ms = round((time.time() - start_time), 2)

        return JSONResponse(
            status_code=200,
            content={
                "request_id": req_id,
                "status": "success",
                "total_lines": len(ocr_lines),
                "predictions": ocr_lines,
                "inference_time_ms": inference_time_ms
            }
        )
    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        error_trace = traceback.format_exc()
        inference_time_ms = round((time.time() - start_time), 2)

        print(f"\n[ERROR] Request_ID: {req_id}")
        print(f"[ERROR] Traceback:\n{error_trace}")

        return JSONResponse(
            status_code=500,
            content={
                "request_id": req_id,
                "status": "error",
                "message": "Error occur when execute image.",
                "error_details": str(e),
                "inference_time_ms": inference_time_ms
            }
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
