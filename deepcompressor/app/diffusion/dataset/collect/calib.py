# -*- coding: utf-8 -*-
"""Collect calibration dataset."""

import os
from dataclasses import dataclass

import datasets
import torch
from omniconfig import configclass
from torch import nn
from tqdm import tqdm

from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.utils.common import hash_str_to_int, tree_map

from deepcompressor.app.diffusion.utils import get_control
from deepcompressor.app.diffusion.dataset.data import get_dataset
from deepcompressor.app.diffusion.dataset.collect.utils import CollectHook


def process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def collect(config: DiffusionPtqRunConfig, dataset: datasets.Dataset):
    samples_dirpath = os.path.join(config.output.root, "samples")
    caches_dirpath = os.path.join(config.output.root, "caches")
    os.makedirs(samples_dirpath, exist_ok=True)
    os.makedirs(caches_dirpath, exist_ok=True)
    caches = []

    pipeline = config.pipeline.build()
    model = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
    assert isinstance(model, nn.Module)
    model.register_forward_hook(CollectHook(caches=caches), with_kwargs=True)

    batch_size = config.eval.batch_size
    print(f"In total {len(dataset)} samples")
    print(f"Evaluating with batch size {batch_size}")
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)
    for batch in tqdm(
        dataset.iter(batch_size=batch_size, drop_last_batch=False),
        desc="Data",
        leave=False,
        dynamic_ncols=True,
        total=(len(dataset) + batch_size - 1) // batch_size,
    ):
        filenames = batch["filename"]
        prompts = batch["prompt"]
        
        seeds = [hash_str_to_int(name) for name in filenames]
        generators = [torch.Generator(device=pipeline.device).manual_seed(seed) for seed in seeds]
        pipeline_kwargs = config.eval.get_pipeline_kwargs()

        task = config.pipeline.task
        control_root = config.eval.control_root
        if task in ["canny-to-image", "depth-to-image", "inpainting"]:
            controls = get_control(
                task,
                batch["image"],
                names=batch["filename"],
                data_root=os.path.join(
                    control_root, collect_config.dataset_name, f"{dataset.config_name}-{config.eval.num_samples}"
                ),
            )
            if task == "inpainting":
                pipeline_kwargs["image"] = controls[0]
                pipeline_kwargs["mask_image"] = controls[1]
            else:
                pipeline_kwargs["control_image"] = controls

        target_seq_len = 64
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt=prompts,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device=pipeline.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        current_seq_len = prompt_embeds.shape[1]
        if current_seq_len < target_seq_len:
            pad_len = target_seq_len - current_seq_len
        # Pad encoder_hidden_states: [batch, seq_len, hidden] -> [batch, 512, hidden]
        prompt_embeds = torch.nn.functional.pad(
            prompt_embeds, (0, 0, 0, pad_len), value=0
        )
        # Pad encoder_hidden_states_mask: [batch, seq_len] -> [batch, 512]
        prompt_embeds_mask = torch.nn.functional.pad(
            prompt_embeds_mask, (0,  pad_len), value=0
        )
        
        negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
            prompt=" ",
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device=pipeline.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        current_seq_len = negative_prompt_embeds.shape[1]
        if current_seq_len < target_seq_len:
            pad_len = target_seq_len - current_seq_len
        # Pad encoder_hidden_states: [batch, seq_len, hidden] -> [batch, 512, hidden]
        negative_prompt_embeds = torch.nn.functional.pad(
            negative_prompt_embeds, (0, 0, 0, pad_len), value=0
        )
        # Pad encoder_hidden_states_mask: [batch, seq_len] -> [batch, 512]
        negative_prompt_embeds_mask = torch.nn.functional.pad(
            negative_prompt_embeds_mask, (0,  pad_len), value=0
        )
        
        
        result_images = pipeline(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask, negative_prompt_embeds=negative_prompt_embeds, negative_prompt_embeds_mask=negative_prompt_embeds_mask,generator=generators, target_seq_len=target_seq_len, **pipeline_kwargs).images
        num_guidances = (len(caches) // batch_size) // config.eval.num_steps
        num_steps = len(caches) // (batch_size * num_guidances)
        assert (
            len(caches) == batch_size * num_steps * num_guidances
        ), f"Unexpected number of caches: {len(caches)} != {batch_size} * {config.eval.num_steps} * {num_guidances}"
        for j, (filename, image) in enumerate(zip(filenames, result_images, strict=True)):
            image.save(os.path.join(samples_dirpath, f"{filename}.png"))
            for s in range(num_steps):
                for g in range(num_guidances):
                    c = caches[s * batch_size * num_guidances + g * batch_size + j]
                    c["filename"] = filename
                    c["step"] = s
                    c["guidance"] = g
                    c = tree_map(lambda x: process(x), c)
                    torch.save(c, os.path.join(caches_dirpath, f"{filename}-{s:05d}-{g}.pt"))
        caches.clear()


@configclass
@dataclass
class CollectConfig:
    """Configuration for collecting calibration dataset.

    Args:
        root (`str`, *optional*, defaults to `"datasets"`):
            Root directory to save the collected dataset.
        dataset_name (`str`, *optional*, defaults to `"qdiff"`):
            Name of the collected dataset.
        prompt_path (`str`, *optional*, defaults to `"prompts/qdiff.yaml"`):
            Path to the prompt file.
        num_samples (`int`, *optional*, defaults to `128`):
            Number of samples to collect.
    """

    root: str = "datasets"
    dataset_name: str = "qdiff"
    data_path: str = "prompts/qdiff.yaml"
    num_samples: int = 128


if __name__ == "__main__":
    parser = DiffusionPtqRunConfig.get_parser()
    parser.add_config(CollectConfig, scope="collect", prefix="collect")
    configs, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args(["examples/diffusion/configs/model/qwenimage.yaml", "examples/diffusion/configs/collect/qdiff.yaml"])
    ptq_config, collect_config = configs[""], configs["collect"]
    assert isinstance(ptq_config, DiffusionPtqRunConfig)
    assert isinstance(collect_config, CollectConfig)
    if len(unused_cfgs) > 0:
        print(f"Warning: unused configurations {unused_cfgs}")
    if unused_args is not None:
        print(f"Warning: unused arguments {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"

    collect_dirpath = os.path.join(
        collect_config.root,
        str(ptq_config.pipeline.dtype),
        ptq_config.pipeline.name,
        ptq_config.eval.protocol,
        collect_config.dataset_name,
        f"s{collect_config.num_samples}",
    )
    print(f"Saving caches to {collect_dirpath}")

    dataset = get_dataset(
        collect_config.data_path,
        max_dataset_size=collect_config.num_samples,
        return_gt=ptq_config.pipeline.task in ["canny-to-image"],
        repeat=1,
    )

    ptq_config.output.root = collect_dirpath
    os.makedirs(ptq_config.output.root, exist_ok=True)
    collect(ptq_config, dataset=dataset)
