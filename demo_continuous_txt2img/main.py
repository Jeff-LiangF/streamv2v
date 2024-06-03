import asyncio
import base64
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

import uvicorn
from config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.wrapper import StreamV2VWrapper

base_model = "stabilityai/sd-turbo"

logger = logging.getLogger("uvicorn")
PROJECT_DIR = Path(__file__).parent.parent


class PredictInputModel(BaseModel):
    """
    The input model for the /predict endpoint.
    """

    prompt: str


class PredictResponseModel(BaseModel):
    """
    The response model for the /predict endpoint.
    """

    base64_image: str


class UpdatePromptResponseModel(BaseModel):
    """
    The response model for the /update_prompt endpoint.
    """

    prompt: str


class Api:
    def __init__(self, config: Config) -> None:
        """
        Initialize the API.

        Parameters
        ----------
        config : Config
            The configuration.
        """
        self.config = config
        self.stream = StreamV2VWrapper(
            model_id_or_path=base_model,
            t_index_list=[0],
            mode='txt2img',
            frame_buffer_size=1,
            warmup=10,
            acceleration=config.acceleration,
            do_add_noise=True,
            output_type="pil",
            use_denoising_batch=False,
            use_cached_attn=True,
            use_feature_injection=True,
            feature_injection_strength=0.8,
            feature_similarity_threshold=0.98,
            cache_interval=1,
            cache_maxframes=4,
            use_tome_cache=False,
            cfg_type="none",
            seed=1,
        )

        self.app = FastAPI()
        self.app.add_api_route(
            "/api/predict",
            self._predict,
            methods=["POST"],
            response_model=PredictResponseModel,
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.mount("/", StaticFiles(directory="./frontend/dist", html=True), name="public")

        self._predict_lock = asyncio.Lock()
        self._update_prompt_lock = asyncio.Lock()

    
    async def _predict(self, inp: PredictInputModel) -> PredictResponseModel:
        """
        Predict an image and return.

        Parameters
        ----------
        inp : PredictInputModel
            The input.

        Returns
        -------
        PredictResponseModel
            The prediction result.
        """
        async with self._predict_lock:
            return PredictResponseModel(base64_image=self._pil_to_base64(self.stream(prompt=inp.prompt)))

    def _pil_to_base64(self, image: Image.Image, format: str = "JPEG") -> bytes:
        """
        Convert a PIL image to base64.

        Parameters
        ----------
        image : Image.Image
            The PIL image.

        format : str
            The image format, by default "JPEG".

        Returns
        -------
        bytes
            The base64 image.
        """
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    def _base64_to_pil(self, base64_image: str) -> Image.Image:
        """
        Convert a base64 image to PIL.

        Parameters
        ----------
        base64_image : str
            The base64 image.

        Returns
        -------
        Image.Image
            The PIL image.
        """
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]
        return Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB")


if __name__ == "__main__":
    from config import Config

    config = Config()

    uvicorn.run(
        Api(config).app,
        host=config.host,
        port=config.port,
        workers=config.workers,
    )
