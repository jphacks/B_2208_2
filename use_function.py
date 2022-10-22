import os
import time
from pytorch_lightning import seed_everything
import torch
from omegaconf import OmegaConf
import set_dir
from torch import autocast
from contextlib import nullcontext
from PIL import Image
import requests
import numpy as np
from einops import repeat
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from IPython import display
from einops import rearrange
import sys
import set_dir

sys.path.extend([
    'src/taming-transformers',
    'src/clip',
    'stable-diffusion/',
    'k-diffusion',
    'pytorch3d-lite',
    'AdaBins',
    'MiDaS',
    'ESRGAN'
])

from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from k_diffusion.external import CompVisDenoiser
from helpers.k_samplers import sampler_fn

# import importlib

# PLMSSampler = importlib.import_module("stable-diffusion.ldm.models.diffusion.plms")
# instantiate_from_config = importlib.import_module("stable-diffusion.ldm.util")
# CompVisDenoiser = importlib.import_module("k-diffusion.k_diffusion.external")

def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path, time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def sanitize(prompt):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tmp = ''.join(filter(whitelist.__contains__, prompt))
    return tmp.replace(' ', '_')


def load_model_from_config(config, ckpt, verbose=False, device='cuda', half_precision=True):
    map_location = "cuda"
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=map_location)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if half_precision:
        model = model.half().to(device)
    else:
        model = model.to(device)
    model.eval()
    return model


def load_img(path, shape, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.*image - 1.

    return image, mask_image


models_path = set_dir.models_path
model_checkpoint =  "sd-v1-4.ckpt"
device = torch.device("cuda")
ckpt_config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
local_config = OmegaConf.load(f"{ckpt_config_path}")
ckpt_path = os.path.join(models_path, model_checkpoint)
half_precision = True
model = load_model_from_config(
    local_config, f"{ckpt_path}", half_precision=half_precision)


#
# Callback functions
#
class SamplerCallback(object):
    # Creates the callback function to be passed into the samplers for each step
    def __init__(self, args, mask=None, init_latent=None, sigmas=None, sampler=None,
                 verbose=False):
        self.sampler_name = args.sampler
        self.dynamic_threshold = args.dynamic_threshold
        self.static_threshold = args.static_threshold
        self.mask = mask
        self.init_latent = init_latent
        self.sigmas = sigmas
        self.sampler = sampler
        self.verbose = verbose

        self.batch_size = args.n_samples
        self.save_sample_per_step = args.save_sample_per_step
        self.show_sample_per_step = args.show_sample_per_step
        self.paths_to_image_steps = [os.path.join(
            args.outdir, f"{args.timestring}_{index:02}_{args.seed}") for index in range(args.n_samples)]

        if self.save_sample_per_step:
            for path in self.paths_to_image_steps:
                os.makedirs(path, exist_ok=True)

        self.step_index = 0

        self.noise = None
        if init_latent is not None:
            self.noise = torch.randn_like(init_latent, device=device)

        self.mask_schedule = None
        if sigmas is not None and len(sigmas) > 0:
            self.mask_schedule, _ = torch.sort(sigmas/torch.max(sigmas))
        elif len(sigmas) == 0:
            # no mask needed if no steps (usually happens because strength==1.0)
            self.mask = None

        if self.sampler_name in ["plms", "ddim"]:
            if mask is not None:
                assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"

        if self.sampler_name in ["plms", "ddim"]:
            # Callback function formated for compvis latent diffusion samplers
            self.callback = self.img_callback_
        else:
            # Default callback function uses k-diffusion sampler variables
            self.callback = self.k_callback_

        self.verbose_print = print if verbose else lambda *args, **kwargs: None

    def view_sample_step(self, latents, path_name_modifier=''):
        if self.save_sample_per_step or self.show_sample_per_step:
            samples = model.decode_first_stage(latents)
            if self.save_sample_per_step:
                fname = f'{path_name_modifier}_{self.step_index:05}.png'
                for i, sample in enumerate(samples):
                    sample = sample.double().cpu().add(1).div(2).clamp(0, 1)
                    sample = torch.tensor(np.array(sample))
                    grid = make_grid(sample, 4).cpu()
                    TF.to_pil_image(grid).save(os.path.join(
                        self.paths_to_image_steps[i], fname))
            if self.show_sample_per_step:
                print(path_name_modifier)
                self.display_images(samples)
        return

    def display_images(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(np.array(images))
        grid = make_grid(images, 4).cpu()
        display.display(TF.to_pil_image(grid))
        return

    # The callback function is applied to the image at each step
    def dynamic_thresholding_(self, img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold,
                          axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(self, args_dict):
        self.step_index = args_dict['i']
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(args_dict['x'], self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(args_dict['x'], -1 *
                         self.static_threshold, self.static_threshold)
        if self.mask is not None:
            init_noise = self.init_latent + self.noise * args_dict['sigma']
            is_masked = torch.logical_and(
                self.mask >= self.mask_schedule[args_dict['i']], self.mask != 0)
            new_img = init_noise * \
                torch.where(is_masked, 1, 0) + \
                args_dict['x'] * torch.where(is_masked, 0, 1)
            args_dict['x'].copy_(new_img)

        self.view_sample_step(args_dict['denoised'], "x0_pred")

    # Callback for Compvis samplers
    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(self, img, i):
        self.step_index = i
        # Thresholding functions
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(img, self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(img, -1*self.static_threshold, self.static_threshold)
        if self.mask is not None:
            i_inv = len(self.sigmas) - i - 1
            init_noise = self.sampler.stochastic_encode(self.init_latent, torch.tensor(
                [i_inv]*self.batch_size).to(device), noise=self.noise)
            is_masked = torch.logical_and(
                self.mask >= self.mask_schedule[i], self.mask != 0)
            new_img = init_noise * \
                torch.where(is_masked, 1, 0) + img * \
                torch.where(is_masked, 0, 1)
            img.copy_(new_img)

        self.view_sample_step(img, "x")


def generate(args, frame=0, return_latent=False, return_sample=False, return_c=False):

    model_config = "v1-inference.yaml"
    models_path = set_dir.models_path
    output_path = set_dir.output_path

    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    sampler = PLMSSampler(model)
    model_wrap = CompVisDenoiser(model)
    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(args.init_sample))
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(init_image))  # move to latent space

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # # Mask functions
    # if args.use_mask:
    #     assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
    #     assert args.use_init, "use_mask==True: use_init is required for a mask"
    #     assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"

    #     mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
    #                         init_latent.shape,
    #                         args.mask_contrast_adjust,
    #                         args.mask_brightness_adjust)

    #     if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
    #         raise Warning(
    #             "use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")

    #     mask = mask.to(device)
    #     mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    # else:
    #     mask = None

    mask = None

    assert not ((args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)
                ), "Need an init image when use_mask == True and overlay_mask == True"

    t_enc = int((1.0-args.strength) * args.steps)

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(args.steps)
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    if args.sampler in ['plms', 'ddim']:
        sampler.make_schedule(ddim_num_steps=args.steps,
                              ddim_eta=args.ddim_eta, ddim_discretize='fill', verbose=False)

    callback = SamplerCallback(args=args,
                               mask=mask,
                               init_latent=init_latent,
                               sigmas=k_sigmas,
                               sampler=sampler,
                               verbose=False).callback

    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in data:
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    # if args.prompt_weighting:
                    #     uc, c = get_uc_and_c(prompts, model, args, frame)
                    # else:
                    uc = model.get_learned_conditioning(batch_size * [""])
                    c = model.get_learned_conditioning(prompts)

                    if args.scale == 1.0:
                        uc = None
                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in ["klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"]:
                        samples = sampler_fn(
                            c=c,
                            uc=uc,
                            args=args,
                            model_wrap=model_wrap,
                            init_latent=init_latent,
                            t_enc=t_enc,
                            device=device,
                            cb=callback)
                    else:
                        # args.sampler == 'plms' or args.sampler == 'ddim':
                        if init_latent is not None and args.strength > 0:
                            z_enc = sampler.stochastic_encode(
                                init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        else:
                            z_enc = torch.randn(
                                [args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)
                        if args.sampler == 'ddim':
                            samples = sampler.decode(z_enc,
                                                     c,
                                                     t_enc,
                                                     unconditional_guidance_scale=args.scale,
                                                     unconditional_conditioning=uc,
                                                     img_callback=callback)
                        elif args.sampler == 'plms':  # no "decode" function in plms, so use "sample"
                            shape = [args.C, args.H //
                                     args.f, args.W // args.f]
                            samples, _ = sampler.sample(S=args.steps,
                                                        conditioning=c,
                                                        batch_size=args.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=args.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=args.ddim_eta,
                                                        x_T=z_enc,
                                                        img_callback=callback)
                        else:
                            raise Exception(
                                f"Sampler {args.sampler} not recognised.")

                    if return_latent:
                        results.append(samples.clone())

                    x_samples = model.decode_first_stage(samples)

                    # if args.use_mask and args.overlay_mask:
                    #     # Overlay the masked image after the image is generated
                    #     if args.init_sample is not None:
                    #         img_original = args.init_sample
                    #     elif init_image is not None:
                    #         img_original = init_image
                    #     else:
                    #         raise Exception(
                    #             "Cannot overlay the masked image without an init image to overlay")

                    #     mask_fullres = prepare_mask(args.mask_file if mask_image is None else mask_image,
                    #                                 img_original.shape,
                    #                                 args.mask_contrast_adjust,
                    #                                 args.mask_brightness_adjust)
                    #     mask_fullres = mask_fullres[:, :3, :, :]
                    #     mask_fullres = repeat(
                    #         mask_fullres, '1 ... -> b ...', b=batch_size)

                    #     mask_fullres[mask_fullres < mask_fullres.max()] = 0
                    #     mask_fullres = gaussian_filter(
                    #         mask_fullres, args.mask_overlay_blur)
                    #     mask_fullres = torch.Tensor(mask_fullres).to(device)

                    #     x_samples = img_original * mask_fullres + \
                    #         x_samples * ((mask_fullres * -1.0) + 1)

                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp(
                        (x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255. * \
                            rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        results.append(image)
    return results
