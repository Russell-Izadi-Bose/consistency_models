"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.karras_diffusion import karras_sample
from cm.random_util import get_generator
from cm.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    log_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    )
    logger.configure(log_dir)

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

        plot_image(out_path, logger.get_dir())

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        log_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def plot_image(npz_file_path, target_dir):
    # Load data from the .npz file
    data = np.load(npz_file_path)
    if len(data) == 1:
        arr = next(iter(data.values()))
        label_arr = None
        print(arr.shape)
    elif len(data) == 2:
        arr, label_arr = data.values()
        print(arr.shape, label_arr.shape)

    # Number of images
    N = arr.shape[0]

    # Create a figure with subplots in a grid of N//5 x 5
    fig, axs = plt.subplots(N // 5, 5, figsize=(15, 3 * (N // 5)))

    # Ensure axs is 2D
    axs = axs.reshape(-1)

    # Plot each image and its label
    for i, ax in enumerate(axs):
        if i < N:
            ax.imshow(arr[i])
            ax.axis("off")
            # Display the label at the center of each image
            if label_arr is not None:
                ax.text(
                    0.5,
                    0.5,
                    str(label_arr[i]),
                    color="white",
                    fontsize=12,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.axis("off")  # Turn off axis for empty subplots

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    name = npz_file_path.split(os.sep)[-1]
    name = name.split(".")[0]
    png_file_path = os.path.join(target_dir, f"{name}.png")
    print(png_file_path)
    plt.savefig(png_file_path)


if __name__ == "__main__":
    main()
