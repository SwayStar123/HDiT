import argparse
import copy
from copy import deepcopy
import logging
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from pathlib import Path
from collections import OrderedDict
import json
from diffusers.models import AutoencoderKL

import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.optimization import get_scheduler

from model.hdit import HDiT_models
from loss import FMLoss

from dataset import CustomDataset
# from dataset import MockDataset

import math
from torchvision.utils import make_grid
from PIL import Image

logger = get_logger(__name__)

def preprocess_raw_image(x):
    x = x / 255.
    x = (x * 2) - 1
    return x

def get_raw_image(x):
    x = (x + 1) / 2
    x = x * 255.
    return x

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    channels = 4 if args.use_latents else 3
    resolution = args.resolution // 8 if args.use_latents else args.resolution

    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    model = HDiT_models[args.model](
        num_classes=args.num_classes,
        in_channels=channels,
        attention_kernel_size=3 if args.use_latents else 7
    )

    divisible_requirement = ((((model.num_levels-1)**2) * model.patch_size) * (8 if args.use_latents else 1))
    assert args.resolution % divisible_requirement == 0, f"Image size must be divisible by {divisible_requirement} (because of downsampling in HDiT)."

    model = model.to(device)
    ema = deepcopy(model).to(device)

    if args.use_latents:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

        latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
        latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = FMLoss(
        prediction=args.prediction,
        path_type=args.path_type,
        weighting=args.weighting,
    )
    if accelerator.is_main_process:
        logger.info(f"HDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup dataset:
    train_dataset = CustomDataset(args.data_dir_train, args.use_latents)

    # Returns random data for testing
    # train_dataset = MockDataset(
    #     num_samples=1000,  # Adjust as needed for testing
    #     image_shape=(channels * 2 if args.use_latents else channels, resolution, resolution),
    #     num_classes=args.num_classes + 1,  # +1 for class-free guidance
    # )

    num_images = len(train_dataset)
    local_batch_size = int(args.batch_size)

    # Create data loaders:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {num_images:,} images ({args.data_dir_train})")
        logger.info(
            f"Total batch size: {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    epoch_start = -1
    if args.resume_ckpt is not None:
        ckpt = torch.load(
            args.resume_ckpt,
            map_location='cpu',
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        epoch_start = ckpt['epoch'] - 1
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        logger.info(f"Starting training experiment: {args.exp_name}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_xs, gt_ys = next(iter(train_dataloader))
    gt_xs = gt_xs[:sample_batch_size].to(device)
    gt_ys = gt_ys[:sample_batch_size].to(device)
    # Create sampling noise:
    n = gt_ys.size(0)
    xT = torch.randn((n, channels, resolution, resolution), device=device)
    
    base_dir = os.path.join(args.output_dir, args.exp_name)
    sample_dir = os.path.join(base_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    if args.use_latents:
        gt_xs = sample_posterior(
            gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
        gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
        gt_samples = (gt_xs + 1) / 2.
        gt_samples = accelerator.gather(gt_samples.to(torch.float32))
        gt_samples = Image.fromarray(array2grid(gt_samples))
        gt_samples.save(f"{sample_dir}/gt_samples_step.png")


    for epoch in range(epoch_start+1, args.epochs):
        model.train()
        for x, y in train_dataloader:
            # save checkpoint (feel free to adjust the frequency)
            if (global_step % args.checkpoint_steps == 0) and global_step > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/step-{global_step}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")


            # sample and save images (feel free to adjust the frequency)
            if (global_step % args.sample_steps == 0) and global_step > 0:
                from samplers import euler_sampler
                with torch.no_grad():
                    model.eval()
                    samples = euler_sampler(
                        model,
                        xT,
                        gt_ys,
                        num_steps=50,
                        cfg_scale= 4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)

                    if args.use_latents:
                        samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    
                # Save images locally
                accelerator.wait_for_everyone()
                out_samples = accelerator.gather(samples.to(torch.float32))
                
                # Save as grid images
                out_samples = Image.fromarray(array2grid(out_samples))

                if accelerator.is_main_process:
                    out_samples.save(f"{sample_dir}/samples_step_{global_step}.png")
                    logger.info(f"Saved samples at step {global_step}")
                model.train()

            x = x.to(device)
            if args.use_latents:
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
            y = y.to(device)
            drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob  
            labels = torch.where(drop_ids, args.num_classes, y)
            model_kwargs = dict(y=labels)

            with accelerator.accumulate(model):
                gen_loss = loss_fn(model, x, model_kwargs)
                loss = gen_loss.mean()

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)  # change ema function

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        # save checkpoint (feel free to adjust the frequency)
        if (epoch+1) % args.checkpoint_epochs == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "steps": global_step,
                }
                checkpoint_path = f"{checkpoint_dir}/epoch-{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--sample-steps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=801)
    parser.add_argument("--checkpoint-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-epochs", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=400000)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir-train", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-latents", type=bool, default=False)

    # precision
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=8)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="use class-free guidance if > 0")
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)