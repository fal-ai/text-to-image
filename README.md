# fal-ai/text-to-image

**tl;dr** A single API for text-to-image, built on [fal](https://fal.ai) that supports all Stable Diffusion variants, checkpoints and LoRAs from HuggingFace (🤗) and CivitAI.

To start using an already deployed endpoint check out the [API docs here](https://fal.ai/models/sd-loras).

## At a glance:
- A single text-to-image API that can scale to 100s of GPUs automatically
- Supports all versions of Stable Diffusion (1.4, 1.5, 2.0, 2.1 and XL)
- Pass any finetuned base model (RealisticVision etc) as a URL from HuggingFace or CivitAI
- Pass any LoRA weights from HuggingFace or CivitAI
- Supports multiple LoRA loading
- Built on the latest high-performance stack: PyTorch 2.0, diffusers, Flash Attention 2.0, LoRA fusing / unfusing
- Minimal cold starts
- Pay by the second (when deployed on [fal](https://fal.ai))

## Background
fal text-to-image originally started as a simple reference application demonstrating the capabilities of our serverless Python cloud [fal](https://fal.ai) with a text-to-image example. Our goal was to provide a high-performance LoRA loading/unloading example on top of arbitrary Stable Diffusion models. From there, we went on a deep rabbit hole and made significant contributions to the amazing [diffusers](https://github.com/huggingface/diffusers) library, supporting [many different LoRA variants](https://github.com/huggingface/diffusers/pull/4147) and [other](https://github.com/huggingface/diffusers/pull/4980) [optimizations](https://github.com/huggingface/diffusers/pull/4979).

Today, text-to-image is a very capable application serving millions of users as a single API for high-performance inference with extensive LoRA support, made possible by our serverless Python runtime, [fal](https://fal.ai). We wanted to open source the repo for the community to be able to look under the hood, tinker, send us feedback and feature requests.

If you want to just try this app from our simple UI, go to [fal.ai's model playground](https://fal.ai/models/sd-loras)! There, you will also see instructions on how to use the [API](https://www.fal.ai/models/sd-loras/api).

To deploy this application on fal on your own, check out the [fal docs](https://fal.ai/docs)!

Join our [Discord community](https://discord.com/invite/Fyc9PwrccF) and help us shape the direction of this project!
