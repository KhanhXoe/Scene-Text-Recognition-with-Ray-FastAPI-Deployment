import os
import tempfile
from io import BytesIO

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
import httpx

import numpy as np
from PIL import Image
from ray import serve
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, object_detection_handle):
        self.handle = object_detection_handle

    async def process_image(self, image_data: bytes) -> Response:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(image_data)
                temp_file_path = temp_file.name

            bboxes, classes, names, confs = await self.handle.detect.remote(
                temp_file_path
            )

            image = Image.open(temp_file_path)
            image_array = np.array(image)

            annotator = Annotator(image_array, font="Arial.ttf", pil=False)

            for box, cls, conf in zip(bboxes, classes, confs):
                c = int(cls)
                label = f"{names[c]} {conf:.2f}"
                annotator.box_label(box, label, color=colors(c, True))

            annotated_image = Image.fromarray(annotator.result())
            file_stream = BytesIO()
            annotated_image.save(file_stream, format="jpeg")
            file_stream.seek(0)
            os.unlink(temp_file_path)

            return Response(content=file_stream.getvalue(), media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    @app.get("/detect", response_class=Response)
    async def detect_url(self, image_url: str):
        """Endpoint for processing images from URLs"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                return await self.process_image(response.content)
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")

    @app.post("/detect/upload", response_class=Response)
    async def detect_upload(self, file: UploadFile = File(...)):
        """Endpoint for processing uploaded image files"""
        try:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")

            content = await file.read()
            return await self.process_image(content)

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing uploaded file: {e}"
            )


@serve.deployment(
    ray_actor_options={"num_gpus": 0.5, "num_cpus": 2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class ObjectDetection:
    def __init__(self):
        self.model = YOLO("yolo11s.pt")

    def detect(self, image_path: str):
        try:
            results = self.model(image_path, verbose=False)[0]
            return (
                results.boxes.xyxy.tolist(),
                results.boxes.cls.tolist(),
                results.names,
                results.boxes.conf.tolist(),
            )

        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

entrypoint = APIIngress.bind(ObjectDetection.bind())