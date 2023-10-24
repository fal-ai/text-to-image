from contextlib import contextmanager
from functools import partial
from typing import Literal

from fal import cached, function
from fal.toolkit import Image, ImageSizeInput, get_image_size
from pydantic import BaseModel, Field

from text_to_image.runtime import SUPPORTED_SCHEDULERS, GlobalRuntime


@cached
def load_session():
    return GlobalRuntime()


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


class InputParameters(BaseModel):
    model_name: str = Field(
        description="URL or HuggingFace ID of the base model to generate the image.",
        examples=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "SG161222/Realistic_Vision_V2.0",
        ],
    )
    prompt: str = Field(
        description="The prompt to use for generating the image. Be as descriptive as possible for best results.",
        examples=[
            "Photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
            "Photo of a classic red mustang car parked in las vegas strip at night",
        ],
    )
    negative_prompt: str = Field(
        default="",
        description="""
            The negative prompt to use.Use it to address details that you don't want
            in the image. This could be colors, objects, scenery and even the small details
            (e.g. moustache, blurry, low resolution).
        """,
        examples=[
            "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
            "nsfw, cartoon, (epicnegative:0.9)",
        ],
    )
    loras: list[LoraWeight] = Field(
        default_factory=list,
        description="""
            The LoRAs to use for the image generation. You can use any number of LoRAs
            and they will be merged together to generate the final image.
        """,
    )
    seed: int | None = Field(
        default=None,
        description="""
            The same seed and the same prompt given to the same version of Stable Diffusion
            will output the same image every time.
        """,
    )
    image_size: ImageSizeInput | None = Field(
        default="square_hd",
        description="""
            The size of the generated image. You can choose between some presets or custom height and width
            that **must be multiples of 8**.
        """,
    )
    num_inference_steps: int = Field(
        default=30,
        description="""
            Increasing the amount of steps tells Stable Diffusion that it should take more steps
            to generate your final result which can increase the amount of detail in your image.
        """,
        ge=0,
        le=150,
        title="Number of inference steps",
    )
    guidance_scale: float = Field(
        default=7.5,
        description="""
            The CFG (Classifier Free Guidance) scale is a measure of how close you want
            the model to stick to your prompt when looking for a related image to show you.
        """,
        ge=0.0,
        le=20.0,
        title="Guidance scale (CFG)",
    )
    clip_skip: int = Field(
        default=0,
        description="""
            Skips part of the image generation process, leading to slightly different results.
            This means the image renders faster, too.
        """,
        ge=0,
        le=2,
    )
    model_architecture: Literal["sd", "sdxl"] | None = Field(
        default=None,
        description=(
            "The architecture of the model to use. If an HF model is used, it will be automatically detected. Otherwise will assume depending on "
            "the model name (whether XL is in the name or not)."
        ),
    )
    scheduler: Literal._getitem(Literal, *SUPPORTED_SCHEDULERS) | None = Field(  # type: ignore
        default=None,
        description="Scheduler / sampler to use for the image denoising process.",
    )
    image_format: Literal["jpeg", "png"] = Field(
        default="png",
        description="The format of the generated image.",
        examples=["jpeg"],
    )
    num_images: int = Field(
        default=1,
        description="""
            Number of images to generate in one request. Note that the higher the batch size,
            the longer it will take to generate the images.
        """,
        ge=1,
        le=8,
        title="Number of images",
    )


class OutputParameters(BaseModel):
    images: list[Image] = Field(description="The generated image files info.")
    seed: int = Field(
        description="""
            Seed of the generated Image. It will be the same value of the one passed in the
            input or the randomly generated that was used in case none was passed.
        """
    )


@contextmanager
def wrap_excs():
    from fastapi import HTTPException

    try:
        yield
    except Exception:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=422, detail=traceback.format_exc())


@function(
    "virtualenv",
    requirements=[
        "git+https://github.com/huggingface/diffusers.git@38a664a3d61e27ab18",
        "transformers",
        "xformers",
        "torch>=2.0",
        "torchvision",
        "safetensors",
        "pytorch-lightning",
        "accelerate",
        "omegaconf",
        "invisible-watermark",
        "pydantic==1.10.12",
        "google-cloud-storage",
        "psutil",
    ],
    machine_type="GPU",
    keep_alive=300,
    serve=True,
    max_concurrency=1,
    _scheduler="nomad",
)
def generate_image(input: InputParameters) -> OutputParameters:
    """
    TODO: write in markdown format 1-2 paragraphs about this function implementation
    that might be relevant to the user (i.e. what makes it fast, lora support / formats, etc)
    """
    import torch

    session = load_session()

    image_size = None
    if input.image_size is not None:
        image_size = get_image_size(input.image_size)

    with wrap_excs():
        with session.load_model(
            input.model_name,
            loras=input.loras,
            clip_skip=input.clip_skip,
            scheduler=input.scheduler,
            model_architecture=input.model_architecture,
        ) as (pipe, global_lora_scale):
            seed = input.seed or torch.seed()

            print("Loaded models:")
            for model in session.models:
                print(f"{model[0]} device: {session.models[model].device()}")

            kwargs = {
                "prompt": input.prompt,
                "negative_prompt": input.negative_prompt,
                "num_images_per_prompt": input.num_images,
                "num_inference_steps": input.num_inference_steps,
                "guidance_scale": input.guidance_scale,
                "generator": torch.manual_seed(seed),
            }

            if image_size is not None:
                kwargs["width"] = image_size.width
                kwargs["height"] = image_size.height

            if global_lora_scale is not None:
                kwargs["cross_attention_kwargs"] = {"scale": global_lora_scale}

            print(f"Generating {input.num_images} images...")
            make_inference = partial(pipe, **kwargs)
            result = session.with_backoff_cuda_retry(
                make_inference, keep_last_model=True
            )

            images = session.upload_images(result.images)
            return OutputParameters(images=images, seed=seed)


if __name__ == "__main__":
    input = InputParameters(
        model_name=f"https://civitai.com/api/download/models/196039",
        prompt="Photo of a classic red mustang car parked in las vegas strip at night",
        model_architecture="sdxl",
    )
    local = generate_image.on(serve=False)
    output = local(input)
    print(output)
