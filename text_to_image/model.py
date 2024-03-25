from contextlib import contextmanager
from functools import partial
from typing import Literal

from fal import cached, function
from fal.toolkit import Image, ImageSizeInput, get_image_size
from pydantic import BaseModel, Field

from text_to_image.runtime import SUPPORTED_SCHEDULERS, GlobalRuntime, filter_by


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


class Embedding(BaseModel):
    path: str = Field(
        description="URL or the path to the embedding weights.",
        examples=[
            "https://storage.googleapis.com/falserverless/style_lora/emb_our_test_1.safetensors",
        ],
    )
    tokens: list[str] = Field(
        default=["<s0>", "<s1>"],
        description="""
            The tokens to map the embedding weights to. Use these tokens in your prompts.
        """,
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
    embeddings: list[Embedding] = Field(
        default_factory=list,
        description="""
            The embeddings to use for the image generation. Only a single embedding is supported at the moment.
            The embeddings will be used to map the tokens in the prompt to the embedding weights.
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
    enable_safety_checker: bool = Field(
        default=False,
        description="If set to true, the safety checker will be enabled.",
    )


class OutputParameters(BaseModel):
    images: list[Image] = Field(description="The generated image files info.")
    seed: int = Field(
        description="""
            Seed of the generated Image. It will be the same value of the one passed in the
            input or the randomly generated that was used in case none was passed.
        """
    )
    has_nsfw_concepts: list[bool] = Field(
        description="Whether the generated images contain NSFW concepts."
    )


@contextmanager
def wrap_excs():
    from fastapi import HTTPException

    try:
        yield
    except (ValueError, TypeError) as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(exc))


@function(
    "virtualenv",
    requirements=[
        "diffusers==0.27.2",
        "transformers",
        "accelerate",
        "torch>=2.1",
        "torchvision",
        "safetensors",
        "pytorch-lightning",
        "omegaconf",
        "invisible-watermark",
        "google-cloud-storage",
        "psutil",
        "peft",
    ],
    machine_type="GPU",
    keep_alive=1800,
    serve=True,
    max_concurrency=4,
    _scheduler="nomad",
    _scheduler_options={
        "preferred_dcs": [
            "datacrunch-fin-01-1",
            "datacrunch-fin-01-bm",
        ],
    },
)
def generate_image(input: InputParameters) -> OutputParameters:
    """
    A single API for text-to-image, built on [fal](https://fal.ai) that supports
    all Stable Diffusion variants, checkpoints and LoRAs from HuggingFace (🤗) and CivitAI.
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
            embeddings=input.embeddings,
            clip_skip=input.clip_skip,
            scheduler=input.scheduler,
            model_architecture=input.model_architecture,
        ) as pipe:
            seed = input.seed or torch.seed()
            kwargs = {
                "prompt": input.prompt,
                "negative_prompt": input.negative_prompt,
                "num_images_per_prompt": input.num_images,
                "num_inference_steps": input.num_inference_steps,
                "guidance_scale": input.guidance_scale,
                "generator": torch.Generator("cuda").manual_seed(seed),
            }

            if image_size is not None:
                kwargs["width"] = image_size.width
                kwargs["height"] = image_size.height

            print(f"Generating {input.num_images} images...")
            make_inference = partial(pipe, **kwargs)

            print("Active adapters", pipe.get_active_adapters())
            result = session.execute_on_cuda(make_inference, ignored_models=[pipe])

            has_nsfw_concepts = session.run_safety_checker(
                images=result.images,
                enable_safety_checker=input.enable_safety_checker,
            )

            images = session.upload_images(filter_by(has_nsfw_concepts, result.images))

            return OutputParameters(
                images=images, seed=seed, has_nsfw_concepts=has_nsfw_concepts
            )


if __name__ == "__main__":
    # generate_image.on(serve=True, keep_alive=0)()
    input = InputParameters(
        model_name=f"stabilityai/stable-diffusion-xl-base-1.0",
        prompt="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        loras=[
            LoraWeight(
                path="https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
                scale=1,
            )
        ],
        embeddings=[
            Embedding(
                path="https://storage.googleapis.com/falserverless/style_lora/pimento_embeddings.pti",
                tokens=["<s0>", "<s1>"],
            )
        ],
        guidance_scale=0,
        num_inference_steps=4,
        num_images=4,
        # scheduler="LCM",
    )
    local = generate_image.on(serve=False, keep_alive=0)
    output = local(input)
    for image in output.images:
        print(image.url)
