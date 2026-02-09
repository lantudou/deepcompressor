#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate images from prompts using FLUX.1-schnell and create I2V dataset.

Usage:
    python generate_i2v_dataset.py
    python generate_i2v_dataset.py --start 10 --end 20  # Generate subset
    python generate_i2v_dataset.py --device cuda:0     # Specify GPU
"""

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate I2V dataset using FLUX.1-schnell")
    parser.add_argument("--video-prompts", type=str, default="video_prompts.json",
                        help="Path to video prompts JSON file")
    parser.add_argument("--images-dir", type=str, default="images",
                        help="Directory to save generated images")
    parser.add_argument("--output-jsonl", type=str, default="mixkit_i2v.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                        help="End index (exclusive)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--width", type=int, default=832,
                        help="Image width")
    parser.add_argument("--height", type=int, default=480,
                        help="Image height")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of inference steps")
    args = parser.parse_args()

    # Configuration
    width = args.width
    height = args.height
    num_inference_steps = args.steps
    guidance_scale = 0.0
    max_sequence_length = 256

    # Create images directory
    os.makedirs(args.images_dir, exist_ok=True)

    # Load video prompts
    print(f"Loading prompts from {args.video_prompts}...")
    with open(args.video_prompts, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply slice
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(data)
    data = data[start_idx:end_idx]

    print(f"Processing items {start_idx} to {end_idx-1} ({len(data)} total)")

    # Initialize FLUX pipeline
    print("Loading FLUX.1-schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )

    if args.device:
        pipe = pipe.to(args.device)
    else:
        pipe.enable_model_cpu_offload()  # Save VRAM by offloading to CPU

    print("Pipeline loaded successfully")

    # Process each prompt
    jsonl_data = []
    skipped = 0
    generated = 0
    failed = 0

    for idx, item in enumerate(tqdm(data, desc="Generating images")):
        actual_idx = start_idx + idx
        image_prompt = item["image_prompt"]
        video_prompt = item["video_prompt"]
        image_filename = item["image_path"]
        image_path = os.path.join(args.images_dir, image_filename)

        # Skip if image already exists
        if os.path.exists(image_path):
            tqdm.write(f"  [{actual_idx:03d}] Skipping {image_filename} (already exists)")
            jsonl_data.append({
                "image_path": image_path,
                "video_prompt": video_prompt
            })
            skipped += 1
            continue

        # Generate image
        try:
            generator = torch.Generator("cpu").manual_seed(actual_idx)
            image = pipe(
                image_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=generator
            ).images[0]

            # Save image
            image.save(image_path)
            tqdm.write(f"  [{actual_idx:03d}] Generated: {image_filename} ({width}x{height})")

            # Add to JSONL data
            jsonl_data.append({
                "image_path": image_path,
                "video_prompt": video_prompt
            })
            generated += 1

        except Exception as e:
            tqdm.write(f"  [{actual_idx:03d}] ERROR generating {image_filename}: {e}")
            failed += 1
            continue

    # Write JSONL file
    print(f"\nWriting dataset to {args.output_jsonl}...")
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total processed: {len(data)}")
    print(f"  Generated:       {generated}")
    print(f"  Skipped:         {skipped}")
    print(f"  Failed:          {failed}")
    print(f"  JSONL entries:   {len(jsonl_data)}")
    print(f"{'='*60}")
    print(f"\nImages saved to: {args.images_dir}/")
    print(f"Dataset saved to: {args.output_jsonl}")
    print(f"\nExample JSONL entry:")
    if jsonl_data:
        print(json.dumps(jsonl_data[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
