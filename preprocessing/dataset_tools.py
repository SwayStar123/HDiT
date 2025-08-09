# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

from collections.abc import Iterator
from dataclasses import dataclass
import io
import json
import multiprocessing as mp
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from encoders import StabilityVAEEncoder

#----------------------------------------------------------------------------
@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]

#----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    input_images = []
    def _recurse_dirs(root: str): # workaround Path().rglob() slowness
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))
    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    # Load labels.
    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    # No labels available => determine from top-level directory names.
    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            img = np.array(PIL.Image.open(fname).convert('RGB'))
            yield ImageEntry(img=img, label=labels.get(arch_fnames[fname]))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        # Load labels.
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file).convert('RGB'))
                yield ImageEntry(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------
def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------
def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

def scale_image(width, height, img):
    """Scale image to specified dimensions."""
    w = img.shape[1]
    h = img.shape[0]
    if width == w and height == h:
        return img
    img = PIL.Image.fromarray(img, 'RGB')
    ww = width if width is not None else w
    hh = height if height is not None else h
    img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
    return np.array(img)

def center_crop_image(width, height, img):
    """Center crop and resize image."""
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    return np.array(img)

def center_crop_wide_image(width, height, img):
    """Center crop wide image."""
    ch = int(np.round(width * img.shape[0] / img.shape[1]))
    if img.shape[1] < width or ch < height:
        return None

    img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    img = np.array(img)

    canvas = np.zeros([width, width, 3], dtype=np.uint8)
    canvas[(width - height) // 2 : (width + height) // 2, :] = img
    return canvas

def center_crop_imagenet_image(image_size, arr):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = PIL.Image.fromarray(arr)
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        assert len(new_size) == 2
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    assert len(new_size) == 2
    pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

def transform_and_encode_worker(args):
    """
    Worker function for parallel image transformation and PNG encoding.
    This function is executed in a separate process.
    """
    idx, image_data, transform_type, output_width, output_height = args
    try:
        img = image_data.img
        
        # Apply the specified transformation.
        if transform_type is None:
            img = scale_image(output_width, output_height, img)
        elif transform_type == 'center-crop':
            img = center_crop_image(output_width, output_height, img)
        elif transform_type == 'center-crop-wide':
            img = center_crop_wide_image(output_width, output_height, img)
        elif transform_type == 'center-crop-dhariwal':
            img = center_crop_imagenet_image(output_width, img)
        else:
            raise ValueError(f'Unknown transform type: {transform_type}')
            
        if img is None:
            return None

        # Get shape for validation in the main process.
        height, width, _ = img.shape
        
        # Encode the transformed image to uncompressed PNG bytes.
        img_pil = PIL.Image.fromarray(img, 'RGB')
        image_bits = io.BytesIO()
        img_pil.save(image_bits, format='png', compress_level=0, optimize=False)
        
        # Return all necessary data to the main process.
        return idx, image_bits.getvalue(), (width, height), image_data.label

    except Exception as e:
        print(f"Error processing image {idx}: {e}")
        return None

def encode_image_worker(args):
    """Worker function for parallel VAE encoding leveraging built-in batching."""
    gpu_id, batch_data, model_url, vae_batch_size = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Always use StabilityVAEEncoder
    encoder = StabilityVAEEncoder(vae_name=model_url, batch_size=vae_batch_size)
    results = []
    try:
        # Prepare all tensors for this GPU's batch
        batch_tensors = []
        batch_indices = []
        batch_labels = []
        for idx, image_data in batch_data:
            img_tensor = torch.tensor(image_data.img).permute(2, 0, 1)  # HWC -> CHW
            batch_tensors.append(img_tensor)
            batch_indices.append(idx)
            batch_labels.append(image_data.label)
        # Stack entire batch and let encoder handle internal batching
        batch_tensor = torch.stack(batch_tensors).to('cuda')
        # The encoder will automatically split this into sub-batches of vae_batch_size
        batch_encoded = encoder.encode_pixels(batch_tensor)
        # Process results
        for i, (idx, label) in enumerate(zip(batch_indices, batch_labels)):
            latents = batch_encoded[i].cpu().numpy()
            results.append((idx, latents, label))
    except Exception as e:
        print(f"Error encoding batch on GPU {gpu_id}: {e}")
        # Add None results for failed batch
        for idx, image_data in batch_data:
            results.append(None)
    return results

#----------------------------------------------------------------------------

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)
@click.option('--workers',    help='Number of parallel workers for image processing', metavar='INT', type=int, default=64, show_default=True)
@click.option('--batch-size', help='Number of images to process in each batch', metavar='INT', type=int, default=1000, show_default=True)

def convert(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    workers: int,
    batch_size: int
):
    """Convert an image dataset into archive format for training.

    Specifying the input images:

    \b
    --source path/                    Recursively load all images from path/
    --source dataset.zip              Load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir               Save output files under /path/to/dir
    --dest /path/to/dataset.zip       Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, class labels are determined from
    top-level directory names.

    Parallelization and Memory Management:

    Use --workers to control the number of CPU cores used for image processing
    (default: 32). This optimized version creates a streaming pipeline, so the
    --batch-size option is no longer used for processing but is kept for API
    compatibility.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    The --transform=center-crop-dhariwal selects a crop/rescale mode that is intended
    to exactly match with results obtained for ImageNet in common diffusion model literature:

    \b
    python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \\
        --dest=datasets/img64.zip --resolution=64x64 --transform=center-crop-dhariwal \\
        --workers=32
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    
    # Validate transform parameters
    if transform in ['center-crop', 'center-crop-wide', 'center-crop-dhariwal']:
        if resolution is None:
            raise click.ClickException(f'must specify --resolution=WxH when using {transform} transform')
        if transform == 'center-crop-dhariwal' and resolution[0] != resolution[1]:
            raise click.ClickException('width and height must match in --resolution=WxH when using center-crop-dhariwal transform')
    
    dataset_attrs = None
    labels = []
    
    output_width, output_height = resolution if resolution is not None else (None, None)

    print(f"Processing and saving {num_files} images with {workers} workers...")
    
    # Create an iterator for worker arguments to stream data without storing it all in memory.
    args_iterator = (
        (idx, image, transform, output_width, output_height)
        for idx, image in enumerate(input_iter)
    )

    with mp.Pool(workers) as pool:
        # Use imap_unordered to create a streaming pipeline. It returns results as soon
        # as they are ready, which keeps the I/O thread busy.
        for result in tqdm(pool.imap_unordered(transform_and_encode_worker, args_iterator), total=num_files, desc="Converting images"):
            if result is None:
                continue
                
            result_idx, image_bytes, (width, height), label = result

            # Validate the attributes of the first processed image and ensure all
            # subsequent images have the same attributes.
            cur_image_attrs = {'width': width, 'height': height}
            if dataset_attrs is None:
                dataset_attrs = cur_image_attrs
                if width != height:
                    raise click.ClickException(f'Image dimensions after scale and crop are required to be square. Got {width}x{height}')
                if width != 2 ** int(np.floor(np.log2(width))):
                    raise click.ClickException('Image width/height after scale and crop are required to be power-of-two')
            elif dataset_attrs != cur_image_attrs:
                err = [f'dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
                raise click.ClickException(f'Image attributes must be equal across all images. Got:\n' + '\n'.join(err))

            # Save the image bytes that were already encoded by the worker process.
            idx_str = f'{result_idx:08d}'
            archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
            save_bytes(os.path.join(archive_root_dir, archive_fname), image_bytes)
            
            # Collect label data along with the original index to sort later.
            label_data = [archive_fname, label] if label is not None else None
            labels.append((result_idx, label_data))

    # Sort labels based on the original image index to ensure deterministic output.
    labels.sort(key=lambda x: x[0])
    sorted_labels = [label_data for _, label_data in labels]
    
    # Create metadata file, checking if all labels were present.
    metadata = {'labels': sorted_labels if all(x is not None for x in sorted_labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    
    close_dest()

@cmdline.command()
@click.option('--model-url',      help='Custom model URL for the encoder (default: stabilityai/sd-vae-ft-mse)', metavar='URL', type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
@click.option('--source',         help='Input directory or archive name', metavar='PATH', type=str, required=True)
@click.option('--dest',           help='Output directory or archive name', metavar='PATH', type=str, required=True)
@click.option('--max-images',     help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--gpus',           help='Number of GPUs to use for parallel encoding', metavar='INT', type=int, default=8, show_default=True)
@click.option('--batch-size',     help='Number of images per GPU in each batch', metavar='INT', type=int, default=128, show_default=True)
@click.option('--vae-batch-size', help='Internal encoder batch size for memory management', metavar='INT', type=int, default=32, show_default=True)

def encode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    gpus: int,
    batch_size: int,
    vae_batch_size: int,
):
    """Encode pixel data to latents using the Stability AI VAE.
    
    The only supported encoder is Stability AI VAE (default: stabilityai/sd-vae-ft-mse).
    Use --model-url to override the default model.
    
    Parallelization and Memory Management:
    Use --gpus to control the number of GPUs used for parallel encoding (default: 8).
    Use --batch-size to control how many images each GPU processes in each batch (default: 100).
    Use --vae-batch-size to control the internal batch size used by the encoder for memory management (default: 8).
    The encoder automatically splits large batches into smaller sub-batches for efficient processing.
    
    Example:
    
        python dataset_tool.py encode --source=datasets/img64.zip \
            --dest=datasets/img64_encoded.zip
        
        python dataset_tool.py encode --model-url=stabilityai/sd-vae-ft-ema --source=datasets/img64.zip \
            --dest=datasets/img64_ema.zip
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')
    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    labels = []
    print(f"Processing {num_files} images in batches of {batch_size * gpus} across {gpus} GPUs...")
    print(f"Encoder will use internal batching with batch size {vae_batch_size}")
    with mp.Pool(gpus) as pool:
        batch = []
        gpu_batches = [[] for _ in range(gpus)]
        for idx, image in tqdm(enumerate(input_iter), total=num_files, desc="Encoding images"):
            gpu_id = idx % gpus
            gpu_batches[gpu_id].append((idx, image))
            max_batch_size = max(len(gpu_batch) for gpu_batch in gpu_batches)
            if max_batch_size >= batch_size or idx == num_files - 1:
                gpu_args = []
                for gpu_id, gpu_batch in enumerate(gpu_batches):
                    if gpu_batch:
                        gpu_args.append((gpu_id, gpu_batch, model_url, vae_batch_size))
                if gpu_args:
                    batch_results = pool.map(encode_image_worker, gpu_args)
                    current_results = []
                    for gpu_result in batch_results:
                        current_results.extend([r for r in gpu_result if r is not None])
                    current_results.sort(key=lambda x: x[0])
                    for result_idx, latents, label in current_results:
                        idx_str = f'{result_idx:08d}'
                        archive_fname = f'{idx_str[:5]}/img-latents-{idx_str}.npy'
                        f = io.BytesIO()
                        np.save(f, latents)
                        save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())
                        labels.append([archive_fname, label] if label is not None else None)
                gpu_batches = [[] for _ in range(gpus)]
    metadata = {
        'labels': labels if all(x is not None for x in labels) else None,
        'compression': {'encoder_type': 'sdvae', 'model_url': model_url}
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    cmdline()