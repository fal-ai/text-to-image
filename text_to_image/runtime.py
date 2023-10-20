import os
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from fal.toolkit import Image
from fal.toolkit.file import FileRepository
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from pydantic import BaseModel, Field

from text_to_image.loras import (
    determine_auxiliary_features,
    identify_lora_weights,
    stack_loras,
)

CHECKPOINTS_DIR = Path("/data/checkpoints")
LORA_WEIGHTS_DIR = Path("/data/loras")
TEMP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"
)
ONE_MB = 1024**2
CHUNK_SIZE = 32 * ONE_MB
CACHE_PREFIX = ""

SUPPORTED_SCHEDULERS = {
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {}),
    "DPM++ 2M Karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M SDE": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++"},
    ),
    "DPM++ 2M SDE Karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler A": ("EulerAncestralDiscreteScheduler", {}),
}


@dataclass
class Model:
    pipeline: object
    last_cache_hit: float = 0

    def as_base(self) -> object:
        self.last_cache_hit = time.monotonic()

        pipe = self.pipeline
        return pipe


class LoraWeight(BaseModel):
    path: str = Field(
        description="URL or the path to the LoRA weights.",
        examples=[
            "https://civitai.com/api/download/models/135931",
            "https://filebin.net/3chfqasxpqu21y8n/my-custom-lora-v1.safetensors",
        ],
    )
    scale: float = Field(
        default=1.0,
        description="""
            The scale of the LoRA weight. This is used to scale the LoRA weight
            before merging it with the base model.
        """,
        ge=0.0,
        le=1.0,
    )


@dataclass
class GlobalRuntime:
    MAX_CAPACITY: ClassVar[int] = 3

    models: dict[tuple[str, ...], Model] = field(default_factory=dict)
    executor: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)
    repository: str | FileRepository = "fal"

    def __post_init__(self):
        if os.getenv("GCLOUD_SA_JSON"):
            self.repository = GoogleStorageRepository(
                url_expiration=2 * 24 * 60,  # 2 days, same as fal,
                bucket_name=os.getenv("GCS_BUCKET_NAME", "fal_file_storage"),
            )

    def download_model_if_needed(self, model_name: str) -> str:
        CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)
        if model_name.startswith("https://") or model_name.startswith("http://"):
            return str(
                self.download_to(model_name, CHECKPOINTS_DIR, extension="safetensors")
            )
        return model_name

    def download_lora_weight_if_needed(self, lora_weight: str) -> str:
        LORA_WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
        if lora_weight.startswith("https://") or lora_weight.startswith("http://"):
            download_path = self.download_to(
                lora_weight, LORA_WEIGHTS_DIR, extension="safetensors"
            )
            return str(download_path.relative_to(LORA_WEIGHTS_DIR))
        return lora_weight

    def download_to(
        self,
        url: str,
        directory: Path,
        extension: str | None = None,
    ) -> Path:
        import os
        import shutil
        import tempfile
        from hashlib import md5
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        if extension is not None and url.endswith(".ckpt"):
            raise ValueError("Can't load non-safetensor model files.")

        url_file = CACHE_PREFIX + urlparse(url).path.split("/")[-1].strip(".")
        url_hash = md5(url.encode()).hexdigest()
        download_path = directory / f"{url_file}-{url_hash}"
        if extension:
            download_path = download_path.with_suffix("." + extension)

        if not download_path.exists():
            request = Request(url, headers={"User-Agent": TEMP_USER_AGENT})
            fd, tmp_file = tempfile.mkstemp()
            try:
                with urlopen(request) as response, open(fd, "wb") as f_stream:
                    total_size = int(response.headers.get("content-length", 0))
                    while data := response.read(CHUNK_SIZE):
                        f_stream.write(data)
                        if total_size > 0:
                            progress_msg = f"Downloading {url}... {f_stream.tell() / total_size:.2%}"
                        else:
                            progress_msg = f"Downloading {url}... {f_stream.tell() / ONE_MB:.2f} MB"
                        print(progress_msg)
            except Exception:
                os.remove(tmp_file)
                raise

            if total_size > 0 and total_size != os.path.getsize(tmp_file):
                os.remove(tmp_file)
                raise ValueError(
                    f"Downloaded file {tmp_file} is not the same size as the remote file."
                )

            # Only move when the download is complete.
            shutil.move(tmp_file, download_path)

        return download_path

    def load_lora_weight(self, lora_weight_path: str) -> dict[str, Any]:
        lora_weight = self.download_lora_weight_if_needed(lora_weight_path)

        try:
            if lora_weight.endswith(".bin"):
                # This only happens for the LoRas that were trained on our platform
                # since HF's trainer script exports the weights in regular torch format.
                import torch

                state_dict = torch.load(LORA_WEIGHTS_DIR / lora_weight)
            else:
                from safetensors import torch

                state_dict = torch.load_file(LORA_WEIGHTS_DIR / lora_weight)
        except Exception as exc:
            raise ValueError(
                "Could not process the lora weights due to a safetensor serialization error."
            ) from exc

        # To avoid false positives, we'll start collecting information about LoRAs first
        # and then convert this check into a proper warning.
        self.check_lora_compatibility(lora_weight, state_dict)
        return state_dict

    def merge_and_apply_loras(
        self,
        pipe: object,
        loras: list[LoraWeight],
    ) -> float:
        print(f"LoRAs: {loras}")
        state_dicts = [self.load_lora_weight(lora_weight.path) for lora_weight in loras]
        lora_scales = [lora_weight.scale for lora_weight in loras]

        if len(loras) == 1:
            [state_dict] = state_dicts
            [global_scale] = lora_scales
        else:
            state_dict = stack_loras(state_dicts, lora_scales)
            global_scale = 1.0

        pipe.load_lora_weights(state_dict)
        pipe.fuse_lora()
        return global_scale

    def check_lora_compatibility(
        self, lora_name: str, state_dict: dict[str, Any]
    ) -> None:
        lora_formats = identify_lora_weights(state_dict)
        auxiliary_features = determine_auxiliary_features(lora_formats, state_dict)
        print(
            f"LoRA {lora_name}: "
            f"formats={lora_formats} "
            f"| auxiliary={auxiliary_features}"
        )

    def get_model(self, model_name: str, arch: str) -> Model:
        import torch
        from diffusers import (
            DiffusionPipeline,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
        )

        model_key = (model_name, arch)
        if model_key not in self.models:
            # Maybe in the future we can offload the model to the disk.
            if len(self.models) >= self.MAX_CAPACITY:
                by_last_hit = lambda kv: kv[1].last_cache_hit
                oldest_model_key, _ = min(self.models.items(), key=by_last_hit)
                print("Unloading model:", oldest_model_key)
                del self.models[oldest_model_key]

            if model_name.endswith(".ckpt") or model_name.endswith(".safetensors"):
                if arch is None:
                    if "xl" in model_name.lower():
                        arch = "sdxl"
                    else:
                        arch = "sd"

                    print(f"Guessing {arch} architecture for {model_name}")

                if arch == "sdxl":
                    pipeline_cls = StableDiffusionXLPipeline
                else:
                    pipeline_cls = StableDiffusionPipeline

                pipe = pipeline_cls.from_single_file(
                    model_name,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                )
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                )

            if hasattr(pipe, "watermark"):
                pipe.watermark = None

            pipe.enable_xformers_memory_efficient_attention()
            pipe = pipe.to("cuda")
            self.models[model_key] = Model(pipe)

        return self.models[model_key]

    @contextmanager
    def load_model(
        self,
        model_name: str,
        loras: list[LoraWeight],
        clip_skip: int = 0,
        scheduler: str | None = None,
        model_architecture: str | None = None,
    ) -> Iterator[tuple[object, float | None]]:
        model_name = self.download_model_if_needed(model_name)

        if model_architecture is None:
            if "xl" in model_name.lower():
                arch = "sdxl"
            else:
                arch = "sd"
            print(f"Guessing {arch} architecture for {model_name}")
        else:
            arch = model_architecture

        model = self.get_model(model_name, arch=arch)
        pipe = model.as_base()

        if clip_skip:
            print(f"Ignoring clip_skip={clip_skip} for now, it's not supported yet!")

        with self.change_scheduler(pipe, scheduler):
            try:
                if loras:
                    global_scale = self.merge_and_apply_loras(pipe, loras)
                else:
                    global_scale = None

                yield (pipe, global_scale)
            finally:
                if loras:
                    try:
                        pipe.unfuse_lora()
                    except Exception:
                        print(
                            "Failed to unfuse LoRAs from the pipe, clearing it out of memory."
                        )
                        traceback.print_exc()
                        self.models.pop((model_name, arch), None)
                    else:
                        pipe.unload_lora_weights()

    @contextmanager
    def change_scheduler(
        self, pipe: object, scheduler_name: str | None = None
    ) -> Iterator[None]:
        import diffusers

        if scheduler_name is None:
            yield
            return

        scheduler_cls_name, scheduler_kwargs = SUPPORTED_SCHEDULERS[scheduler_name]
        scheduler_cls = getattr(diffusers, scheduler_cls_name)
        if scheduler_cls not in pipe.scheduler.compatibles:
            compatibles = ", ".join(cls.__name__ for cls in pipe.scheduler.compatibles)
            raise ValueError(
                f"The scheduler {scheduler_name} is not compatible with this model.\n"
                f"Compatible schedulers: {compatibles}"
            )

        original_scheduler = pipe.scheduler
        try:
            pipe.scheduler = scheduler_cls.from_config(
                pipe.scheduler.config,
                **scheduler_kwargs,
            )
            yield
        finally:
            pipe.scheduler = original_scheduler

    def upload_images(self, images: list[object]) -> list[Image]:
        print("Uploading images...")
        image_uploader = partial(Image.from_pil, repository=self.repository)
        res = list(self.executor.map(image_uploader, images))
        print("Done uploading images.")
        return res
