from tempfile import TemporaryDirectory

import httpx
import pytest
from fal import flags
from fal.sdk import get_default_credentials
from image_similarity_measures.evaluate import evaluation as image_similarity

from text_to_image.model import InputParameters, generate_image

SIMILARITY_METHOD = "fsim"
SIMILARITY_THRESHOLD = 0.8

local_test_fn = generate_image.on(
    serve=False,
    keep_alive=600,
    _scheduler=None,
)

TEST_CASES = [
    {
        "name": "Stable Diffusion XL",
        "input": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt": "Albert Einstein, wojak",
            "seed": 1463963878,
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
            "scheduler": "Euler A",
            "num_images": 1,
        },
        "output": {"images": ["fal_file_storage/5207243b0f56408b9a332e0a48968210.png"]},
    },
    {
        "name": "Playground Isometric Astronaut Dog",
        "input": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt": "single isometric black bernedoodle dressed as an American astronaut",
            "seed": 1477142,
            "image_size": {"width": 1024, "height": 1024},
            "clip_skip": 0,
            "loras": [
                {"path": "https://civitai.com/api/download/models/130580", "scale": 1}
            ],
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), (watermark), immature, child, (big nose:1.4)",
            "scheduler": "DPM++ 2M",
            "num_images": 1,
        },
        "output": {
            "images": ["fal_file_storage/bffa13e17f174984907cf9295e9adf5b.png"],
        },
    },
    {
        "name": "Playground Queen",
        "input": {
            "model_name": "emilianJR/epiCRealism",
            "prompt": "photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
            "seed": 3088576288,
            "image_size": {"width": 544, "height": 960},
            "clip_skip": 0,
            "num_inference_steps": 40,
            "guidance_scale": 4,
            "negative_prompt": "ccartoon, painting, illustration, (worst quality, low quality, normal quality:2), (watermark), immature, child, (big nose:1.4)",
            "scheduler": "DPM++ 2M",
            "num_images": 1,
        },
        "output": {"images": ["fal_file_storage/7ae9d422aab44cc7bff9afbe32d154f3.png"]},
    },
    {
        "name": "Playground Red Mustang",
        "input": {
            "model_name": "emilianJR/epiCRealism",
            "prompt": "Photo of a classic red mustang car parked in las vegas strip at night",
            "seed": 2767081012,
            "image_size": {"width": 720, "height": 960},
            "clip_skip": 0,
            "num_inference_steps": 80,
            "guidance_scale": 5,
            "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
            "scheduler": "DPM++ 2M",
            "num_images": 1,
        },
        "output": {"images": ["fal_file_storage/6b4e820f33d74dd6ab758c11cd2cefa6.png"]},
    },
    {
        "name": "Playground Congolese Woman",
        "input": {
            "model_name": "emilianJR/epiCRealism",
            "prompt": "Photo of a congolese woman, wrinkles, aged, necklace, crowded, closeup, neutral colors, barren land",
            "seed": 803613448,
            "image_size": {"width": 512, "height": 768},
            "clip_skip": 0,
            "num_inference_steps": 60,
            "guidance_scale": 5,
            "negative_prompt": "asian, chinese, busty, (epicnegative:0.9)",
            "scheduler": "DPM++ 2M",
            "num_images": 1,
        },
        "output": {"images": ["fal_file_storage/60db9268799542bc987bef0ac56c894e.png"]},
    },
]


@pytest.fixture(scope="session")
def rest_client() -> httpx.Client:
    credentials = get_default_credentials()
    with httpx.Client(
        base_url=flags.REST_URL, headers=credentials.to_headers()
    ) as client:
        yield client


@pytest.fixture(scope="session")
def image_client() -> httpx.Client:
    with httpx.Client() as client:
        yield client


@pytest.mark.parametrize(
    "name, input, output",
    [
        (
            test["name"],
            test["input"],
            test["output"],
        )
        for test in TEST_CASES
    ],
)
def test_text_to_image(name, input, output, rest_client, image_client):
    loras = InputParameters(**input)
    result = local_test_fn(loras)
    for generated_image, expected_image in zip(result.images, output["images"]):
        response = rest_client.get(f"/storage/link/{expected_image}")
        response.raise_for_status()

        expected_image_url = response.json()["url"]
        similarity = compare_images(
            image_client,
            generated_image.url,
            expected_image_url,
        )
        assert similarity >= SIMILARITY_THRESHOLD


def compare_images(
    client: httpx.Client, baseline_image: str, target_image: str
) -> float:
    with TemporaryDirectory() as directory:
        files = [f"{directory}/baseline.png", f"{directory}/target.png"]
        print("Comparing: ", baseline_image, target_image)
        for image_url, filename in zip([baseline_image, target_image], files):
            response = client.get(image_url)
            response.raise_for_status()
            with open(filename, "wb") as file:
                file.write(response.content)

        results = image_similarity(
            *files,
            metrics=[SIMILARITY_METHOD],
        )
        return results[SIMILARITY_METHOD]
