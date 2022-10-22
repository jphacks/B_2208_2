
# @markdown ****

from use_function import get_output_folder
from use_function import generate
from use_function import sanitize

import json
from IPython import display

import gc
import math
import os
import pathlib
import subprocess
import sys
import time
import cv2
import numpy as np
import pandas as pd
import random
import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from types import SimpleNamespace
from torch import autocast
import re
from scipy.ndimage import gaussian_filter

import set_dir

# prompts = ["A painting, There are not living things, jungle"]

models_path = set_dir.models_path
output_path = set_dir.output_path


override_settings_with_file = False
custom_settings_file = "/content/drive/MyDrive/Settings.txt"


def DeforumArgs():
    # @markdown **Image Settings 画像のサイズ**
    # アンケート用画像サイズ
    W = 728  # @param
    H = 515  # @param

    # 自動生成用画像サイズ
    # W = 512  # @param
    # H = 288  # @param

    # resize to integer multiple of 64
    W, H = map(lambda x: x - x % 64, (W, H))

    # @markdown **Sampling Settings**
    seed = 6  # @param {type:"integer"}

#    sampler = 'euler_ancestral'
    # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    sampler = 'euler_ancestral'
    # @markdown samplerはノイズ除去の計算アルゴリズムの違い. それぞれ[生成される画像に微妙な違い](https://i.imgur.com/2pQPgf0.jpeg)がある。euler_ancestralが短いstep数で綺麗な画像を生成できるとされている。

    steps = 30  # @param
    # @markdown steps: ノイズ除去のステップ数

    scale = 7
    ddim_eta = 0.0
    dynamic_threshold = None
    static_threshold = None

    save_samples = True
    save_settings = True
    display_samples = True
    save_sample_per_step = False
    show_sample_per_step = False  # @param {type:"boolean"}

    # @markdown **Batch Settings**
    n_batch = 1  # @param
    # @markdown 一回に何枚生成するか

    batch_name = "SFCDesignLanguage"
    filename_format = "{timestring}_{index}_{prompt}.png"
    seed_behavior = "iter"  # @param ["iter","fixed","random"]
    # @markdown 生成時の乱数シードの扱い。fixed: 固定. iter: 毎回1ずつ変化. random: ランダム
    make_grid = False
    grid_rows = 2
    outdir = get_output_folder(output_path, batch_name)

    # @markdown -----

    # @markdown **Init Settings**

    # @markdown [ノイズに特定の画像を足した画像からノイズ消去をスタート](https://twitter.com/krea_ai/status/1562463444523110400?s=20&t=S0owgZ6SiEtX2msnXulR5g) 生成される画像をコントロールできる!!
    use_init = False  # @param {type:"boolean"}
    strength = 0.7  # @param {type:"number"}
    # Set the strength to 0 automatically when no init image is used
    strength_0_no_init = True
    # @param {type:"string"}
    init_image = "https://public0.potaufeu.asahi.com/95f0-p/picture/22218751/fe8f805addec7c5d6db4b1d432577189.jpg"
    # Whiter areas of the mask are areas that change more
    use_mask = False  # @param {type:"boolean"}
    use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
    # @param {type:"string"}
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"
    invert_mask = False  # @param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0
    mask_contrast_adjust = 1.0
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5  # {type:"number"}

    n_samples = 1  # doesnt do anything
    precision = 'autocast'
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None

    return locals()


def next_seed(args):
    if args.seed_behavior == 'iter':
        args.seed += 1
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32 - 1)
    return args.seed


def render_image_batch(args, prompts):
    args.prompts = {k: f"{v:05d}" for v, k in enumerate(prompts)}

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_settings or args.save_samples:
        print(f"Saving to {os.path.join(args.outdir, args.timestring)}_*")

    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)

    index = 0

    # function for init image batching
    init_array = []
    if args.use_init:
        if args.init_image == "":
            raise FileNotFoundError("No path was given for init_image")
        if args.init_image.startswith('http://') or args.init_image.startswith('https://'):
            init_array.append(args.init_image)
        elif not os.path.isfile(args.init_image):
            # avoids path error by adding / to end if not there
            if args.init_image[-1] != "/":
                args.init_image += "/"
            # iterates dir and appends images to init_array
            for image in sorted(os.listdir(args.init_image)):
                if image.split(".")[-1] in ("png", "jpg", "jpeg"):
                    init_array.append(args.init_image + image)
        else:
            init_array.append(args.init_image)
    else:
        init_array = [""]

    # when doing large batches don't flood browser with images
    clear_between_batches = args.n_batch >= 32
    picPath = ''
    for iprompt, prompt in enumerate(prompts):
        args.prompt = prompt
        print(f"Prompt {iprompt+1} of {len(prompts)}")
        print(f"{args.prompt}")

        all_images = []

        for batch_index in range(args.n_batch):
            if clear_between_batches and batch_index % 32 == 0:
                display.clear_output(wait=True)
            print(f"Batch {batch_index+1} of {args.n_batch}")

            for image in init_array:  # iterates the init images
                args.init_image = image
                results = generate(args)
                for image in results:
                    if args.make_grid:
                        all_images.append(T.functional.pil_to_tensor(image))
                    if args.save_samples:
                        if args.filename_format == "{timestring}_{index}_{prompt}.png":
                            filename = f"{args.timestring}_{index:05}_{sanitize(prompt)[:160]}.png"
                        else:
                            filename = f"{args.timestring}_{index:05}_{args.seed}.png"
                        picPath = os.path.join(args.outdir, filename)
                        image.save(picPath)
                    if args.display_samples:
                        display.display(image)
                    index += 1
                args.seed = next_seed(args)

        # print(len(all_images))
        if args.make_grid:
            grid = make_grid(all_images, nrow=int(
                len(all_images)/args.grid_rows))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            filename = f"{args.timestring}_{iprompt:05d}_grid_{args.seed}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))
            picPath = os.path.join(args.outdir, filename)
            grid_image.save(picPath)
            display.clear_output(wait=True)
            display.display(grid_image)
    return picPath


def generatePic(prompts):
    # このファイル内で定義されている全ての変数を辞書型で受け取る
    args_dict = DeforumArgs()

    # それをクラスに変換
    args = SimpleNamespace(**args_dict)

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
    if not args.use_init:
        args.init_image = None
    if args.sampler == 'plms' and (args.use_init):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = 'klms'
    if args.sampler != 'ddim':
        args.ddim_eta = 0

    # if anim_args.animation_mode == 'None':
    #     anim_args.max_frames = 1

    # clean up unused memory
    gc.collect()
    torch.cuda.empty_cache()

    # dispatch to appropriate renderer
    picPath = render_image_batch(args, prompts)
    return picPath
