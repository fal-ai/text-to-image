import gc
import os
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

from fal.toolkit import Image
from fal.toolkit.file import FileRepository
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from pydantic import BaseModel, Field

from text_to_image.loras import determine_auxiliary_features, identify_lora_weights

DeviceType = Literal["cpu", "cuda"]

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
    "LCM": ("LCMScheduler", {}),
}

# Amount of RAM to use as buffer, in percentages.
RAM_BUFFER_PERCENTAGE = 1 - 0.75


@dataclass
class Model:
    pipeline: object
    last_cache_hit: float = 0

    def as_base(self) -> object:
        self.last_cache_hit = time.monotonic()

        pipe = self.pipeline
        return pipe

    def device(self) -> DeviceType:
        return self.pipeline.device.type


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
            return str(download_path)
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
        from urllib.request import HTTPError, Request, urlopen

        if extension is not None and url.endswith(".ckpt"):
            raise ValueError("Can't load non-safetensor model files.")

        url_file = CACHE_PREFIX + urlparse(url).path.split("/")[-1].strip(".").replace(
            ".", "_"
        )
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
                    f_stream.flush()
            except HTTPError as exc:
                os.remove(tmp_file)
                raise ValueError(
                    f"Couldn't download weights from the given URL: {url}. Possible cause is: {str(exc)}"
                )
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

    def merge_and_apply_loras(
        self,
        pipe: object,
        loras: list[LoraWeight],
    ):
        print(f"LoRAs: {loras}")
        lora_paths = [
            self.download_lora_weight_if_needed(lora_path) for lora_path in loras
        ]
        adapter_names = [
            Path(lora_path).name.replace(".", "_") for lora_path in lora_paths
        ]
        lora_scales = [lora_weight.scale for lora_weight in loras]

        for lora_path, lora_scale, adapter_name in zip(
            lora_paths, lora_scales, adapter_names
        ):
            print(f"Applying LoRA {lora_path} with scale {lora_scale}.")
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)

        pipe.set_adapters(adapter_names=adapter_names, adapter_weights=lora_scales)
        pipe.fuse_lora()

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
            if model_name.endswith(".ckpt") or model_name.endswith(".safetensors"):
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
    ) -> Iterator[object | None]:
        model_name = self.download_model_if_needed(model_name)

        if model_architecture is None:
            if "xl" in model_name.lower():
                arch = "sdxl"
            else:
                arch = "sd"
            print(
                f"Guessing {arch} architecture for {model_name}. If this is wrong, "
                "please specify it as part of the model_architecture parameter."
            )
        else:
            arch = model_architecture

        model = self.get_model(model_name, arch=arch)
        pipe = model.as_base()
        pipe = self.execute_on_cuda(partial(pipe.to, "cuda"))

        if clip_skip:
            print(f"Ignoring clip_skip={clip_skip} for now, it's not supported yet!")

        with self.change_scheduler(pipe, scheduler):
            try:
                if loras:
                    self.merge_and_apply_loras(pipe, loras)

                yield pipe
            finally:
                if loras:
                    try:
                        pipe.unfuse_lora()
                        pipe.set_adapters(adapter_names=[])
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
        if (
            scheduler_cls not in pipe.scheduler.compatibles
            # Apparently LCM doesn't get reported as a compatible scheduler even
            # though it works.
            and scheduler_cls_name != "LCMScheduler"
        ):
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

    def execute_on_cuda(
        self,
        function: Callable[..., Any],
        *,
        ignored_models: list[object] | None = None,
    ):
        cached_models = self.get_loaded_models_by_device(
            "cuda",
            ignored_models=ignored_models or [],
        )

        first_try = True
        while first_try or cached_models:
            first_try = False

            try:
                return function()
            except RuntimeError as error:
                # Only retry if the error is a CUDA OOM error.
                # https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py#L19
                __cuda_oom_errors = [
                    "CUDA out of memory",
                    "INTERNAL ASSERT FAILED",
                ]
                if (
                    not any(error_str in str(error) for error_str in __cuda_oom_errors)
                    or not cached_models
                ):
                    raise

                # Since cached_models is sorted by last cache hit, we'll pop the the
                # model with the oldest cache hit and try again.
                target_model_id = cached_models.pop()
                self.offload_model_to_cpu(target_model_id)
                self.empty_cache()

        self.empty_cache()
        raise RuntimeError("Not enough CUDA memory to complete the operation.")

    def get_loaded_models_by_device(
        self,
        device: DeviceType,
        ignored_models: list[object],
    ):
        models = [
            model_id
            for model_id in self.models
            if self.models[model_id].device() == device
            if self.models[model_id].pipeline not in ignored_models
        ]
        models.sort(key=lambda model_id: self.models[model_id].last_cache_hit)
        return models

    def offload_model_to_cpu(self, model_id: tuple[str, ...]):
        print(f"Offloading model={model_id} to CPU.")

        def is_ram_buffer_full():
            import psutil

            memory = psutil.virtual_memory()
            percent_available = memory.available / memory.total
            return percent_available < RAM_BUFFER_PERCENTAGE

        models = self.get_loaded_models_by_device("cpu", ignored_models=[])
        while is_ram_buffer_full():
            if not models:
                print(
                    f"Not enough RAM to offload the model to CPU, evicting {model_id}"
                    "it directly."
                )
                del self.models[model_id]
                gc.collect()
                return

            lru_model_id = models.pop()
            print(f"Offloading model={lru_model_id} back to disk.")
            del self.models[lru_model_id]
            gc.collect()

        model = self.models[model_id]
        model.pipeline = model.pipeline.to("cpu")

    def empty_cache(self):
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
