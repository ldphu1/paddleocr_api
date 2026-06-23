import requests
import time
import numpy as np

URL = "http://127.0.0.1:8000/inference"
IMG_PATH = "data/IMG_9011_JPG_JPG.rf.iAZVp7OQp2xVEkvl7msQ.jpg"
NUM_REQUESTS = 20

def run():
    print("running...")
    total_latency = []
    prep_latency = []
    infer_latency = []
    post_latency = []

    start_benchmark_time = time.time()

    image_data = None
    with open(IMG_PATH, "rb") as f:
        image_data = f.read()

    for i in range(NUM_REQUESTS):
        files = {"file": (URL, image_data, "image/jpeg")}

        try:
            res = requests.post(url=URL, files=files)
            if res.status_code == 200:
                data = res.json()
                latencies = data.get("latency", {})
                prep_latency.append(latencies.get("preprocess_time", 0))
                infer_latency.append(latencies.get("inference_time", 0))
                post_latency.append(latencies.get("postprocess_time", 0))
                total_latency.append(latencies.get("total", 0))
                print(f"{i + 1}/ {NUM_REQUESTS}: OK")
            else:
                print(f"{i + 1}/ {NUM_REQUESTS}: Failed")
        except Exception as e:
            print(f"ERROR: {e}")

    total_benchmark_time = time.time() - start_benchmark_time
    fps = NUM_REQUESTS / total_benchmark_time


    print("BÁO CÁO KẾT QUẢ BENCHMARK")
    print("="*40)
    print(f"Preprocess (Avg)  : {np.mean(prep_latency):.2f} ms")
    print(f"Inference (Avg)   : {np.mean(infer_latency):.2f} ms")
    print(f"Postprocess (Avg) : {np.mean(post_latency):.2f} ms")
    print("-" * 40)
    print(f"Total Latency (Avg): {np.mean(total_latency):.2f} ms")
    print(f"Latency P95        : {np.percentile(total_latency, 95):.2f} ms")
    print(f"Tốc độ (FPS)       : {fps:.2f} frames/sec")
    print("="*40)

if __name__ == "__main__":
    run()



