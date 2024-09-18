"""
AINFT

This file implements the Minter class, which is responsible for generating unique images based on a given seed. It utilizes advanced machine learning models, including stable diffusion and language models, to create AI-generated artwork. The Minter class encapsulates the logic for prompt generation, image creation, and ensures reproducibility across different machines.
"""

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from transformers.generation.logits_process import TopKLogitsWarper

import torch.nn.functional as F


class StableDecoding:
    """
    Hugginface's decoder does not give stable results on different machines
    reference https://github.com/huggingface/transformers/blob/f745e7d3f902601686b83c7cce2660c2a94509f0/src/transformers/generation/utils.py#L3002C17-L3002C81
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self):
        input_ids = torch.tensor([[50256]])
        attention_mask = torch.tensor([[1]])
        top_k_warper = TopKLogitsWarper(top_k=100)

        eos_id = self.tokenizer(self.tokenizer.eos_token)["input_ids"]
        self.model.eval()

        past_key_values = None

        for i in range(50):
            output = self.model(
                input_ids=input_ids
                if past_key_values is None
                else input_ids[:, -1].unsqueeze(-1),
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            # Update past_key_values for next iteration
            past_key_values = output.past_key_values

            # transform logits
            next_token_logits = output.logits[:, -1, :].clone().float()
            next_token_logits = top_k_warper(
                input_ids=input_ids, scores=next_token_logits
            )

            probs = F.softmax(next_token_logits - next_token_logits.max(), dim=-1)

            # add penalty to repeated tokens
            probs[0, input_ids[0, -1]] *= 0.01

            # quantize for stability
            scale = 0.01
            zero_point = 0
            probs = torch.quantize_per_tensor(
                probs, scale, zero_point, dtype=torch.qint32
            )
            probs = probs.dequantize()
            probs = probs / probs.sum()

            next_token = sample_from_distribution(probs[0])

            input_ids = torch.cat((input_ids, torch.tensor([[next_token]])), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.tensor([[1]])), dim=-1)

            if next_token == eos_id:
                break

        return self.tokenizer.decode(input_ids[0, 1:])  # ignore first bos


def set_seed(seed: int):
    torch.manual_seed(seed)


def sample_from_distribution(probs):
    # Ensure the input is a tensor
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs, dtype=torch.float32)

    # Compute the cumulative distribution
    cumulative_probs = torch.cumsum(probs, dim=0)

    # Sample a random number between 0 and 1
    random_sample = torch.rand(1).item()

    # Find the index where the random_sample would fall in the cumulative distribution
    sampled_index = torch.searchsorted(cumulative_probs, random_sample).item()

    return sampled_index


def convert_seed(value):
    # Solidity seeds are uint64, this will convert to int32 for torch seeds
    value = value % (2**64)
    value = value - 2**63

    return value


def use_seed(func):
    """simple wrapper function to ensure that the seed is set before we call a function"""

    def wrapper(seed, *args, **kwargs):
        set_seed(seed)
        result = func(*args, **kwargs)
        return result

    return wrapper


class Minter:
    def __init__(
        self,
        input_dir: str = "prompt_model",
        image_input_dir: str = "sdxs-512-dreamshaper/snapshots/76f720262bb051da75666b22c902a78c8e16c763",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_model = AutoModelForCausalLM.from_pretrained(input_dir)

        self.text_pipe = use_seed(
            StableDecoding(model=self.text_model, tokenizer=self.tokenizer)
        )

        self.image_pipe = use_seed(
            StableDiffusionPipeline.from_pretrained(image_input_dir).to("cpu")
        )

    def prompt(self, seed: int) -> str:
        return self.text_pipe(seed)

    def __call__(self, seed: int, return_prompt: bool = False) -> Image:
        seed = convert_seed(seed)
        prompt = self.prompt(seed)

        image = self.image_pipe(
            seed,
            prompt,
            num_inference_steps=1,
            guidance_scale=0,
        ).images[0]

        if return_prompt:
            return image, prompt
        else:
            return image


if __name__ == "__main__":
    minter = Minter()
    result = minter(1337)
    result.save("1337_local.png")
