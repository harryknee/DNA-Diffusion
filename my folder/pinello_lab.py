# -- Imports --
import os
from scipy.stats import zscore
import torch
import copy
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
from IPython.display import display
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm_notebook
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU
from torch.optim import Adam
from tqdm import tqdm_notebook
from torchvision.utils import save_image
import matplotlib
import math
from inspect import isfunction
from functools import partial
import scipy
from scipy.special import rel_entr
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.animation as animation
import matplotlib.image as mpimg
import glob
from PIL import Image
from accelerate import Accelerator

# -- Hyperparameters --
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_human_mouse_rat_monkey.pt')
NUMBER_OF_SAMPLING_TO_COMPARE_CELLS = 1000  # Or None to use everything
SAVE_MODELS_BY_EPOCH = False  # This will save a model with a different name and add the current epoch to the name
TIME_STEPS = 50
IMAGE_SIZE = 200
N_SAMPLES_TO_TEST_METRICS = 1000
EPOCHS = 10000
SAVE_AND_SAMPLE_EVERY_N_EPOCH = 50
EPOCH_LOSS_SHOW_INTERVAL = 5
BATCH_SIZE = 128
LIMIT_TOTAL_SEQUENCES = False  # False # OR False to use everything
MODEL_NAME = 'model_48k_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k'
NUMBER_OF_GPUS = 1
TOTAL_class_number = 1 # set to 1 for unconditional

# -- Utils --
class EMA:  # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


nucleotides = ['A', 'C', 'T', 'G']
def one_hot_encode(seq, alphabet, max_seq_len):
    """One-hot encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(alphabet)))
    for i in range(seq_len):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array


def encode(seq, alphabet):
    """Encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros(len(alphabet))
    for i in range(seq_len):
        seq_array[alphabet.index(seq[i])] = 1

    return seq_array


def show_seq(dataloader_seq):
    for i_image in dataloader_seq:
        for image in i_image:
            image = image.numpy().reshape(4, 200)
            print(image)
            plt.rcParams["figure.figsize"] = (20, 1)
            pd_seq = pd.DataFrame(image)
            pd_seq.index = nucleotides
            sns.heatmap(pd_seq, linewidth=1, cmap='bwr', center=0)
            plt.show()


def extract(a, t, x_shape, device=None):
    batch_size = t.shape[0]
    if device:
        # print('add device')
        a = a.to(device)
        t = t.to(device)

    # print(a.device, 'a', t.device, 't')

    out = a.gather(-1, t)
    # out = a.gather(-1) # to accelerate
    result = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    if device:
        result.to(device)
    return result


# -- Data --
class SequenceDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, seqs, c, transform=None):
        'Initialization'
        self.seqs = seqs
        self.c = c
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.seqs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.seqs[index]

        x = self.transform(image)

        y = self.c[index]

        return x, y

# LOL: add our OHE Dataset deinition
class OHEDataset(Dataset):
    def __init__(self, data_path, device=None, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.device = device
        if device:
            self.labels = torch.load(self.data_path, map_location=device)
        else:
            self.labels = torch.load(self.data_path)
        self.labels[self.labels == 0] = -1

    def __getitem__(self, index):
        # label is the same as the original input
        return torch.unsqueeze(self.labels[index], dim=1)

    def __len__(self):
        return len(self.labels)

# LOL: for CSV dataset (length 256)
class CSVDataset(Dataset):
    def __init__(self, data_path, device=None, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.device = device

        self.df = pd.read_csv(self.data_path)
        self.labels = [one_hot_encode(x, ["A","T","C","G"], 256) for x in self.df['Sequence'] if "N" not in x]
        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels)
        self.labels[self.labels == 0] = -1
        self.labels = torch.unsqueeze(self.labels, dim=1)

    def __getitem__(self, index):
        # label is the same as the original input
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

# -- Diffusion Model --
# modified 1/20/23
from functools import partial
# ALL THIS FUNCTIONS CHANGED WITH THE CONDITIONAL CODE
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    # print (x.shape, 'x_shape')
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, time=t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_guided(
    model,
    x,
    classes,
    t,
    t_index,
    context_mask,
    cond_weight=0.0,
    betas=None,
    sqrt_one_minus_alphas_cumprod=None,
    sqrt_recip_alphas=None,
    posterior_variance=None,
    device=None,
    accelerator=None,
):
    # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
    # print (classes[0])
    batch_size = x.shape[0]
    # double to do guidance with
    t_double = t.repeat(2).to(device)
    x_double = x.repeat(2, 1, 1, 1).to(device)
    betas = betas.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
    betas_t = extract(betas, t_double, x_double.shape, device=device)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t_double, x_double.shape, device=device)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_double, x_double.shape, device=device)

    # classifier free sampling interpolates between guided and non guided using `cond_weight`
    classes_masked = classes * context_mask
    classes_masked = classes_masked.type(torch.long)
    # print ('class masked', classes_masked)
    if accelerator:
        model = accelerator.unwrap_model(model)
    model.output_attention = True
    show_out_test = model(x_double, time=t_double, classes=classes_masked)
    # print(x_double.shape, t_double.shape, 'x_double and t_double')
    # print (show_out_test.shape, 'show_out_test')
    # print (show_out_test)
    # print (show_out_test[0])
    # print (show_out_test[1])
    # print (show_out_test[2])
    # print (show_out_test[3])
    preds, cross_map_full = model(x_double, time=t_double, classes=classes_masked)  # I added cross_map
    model.output_attention = False
    cross_map = cross_map_full[:batch_size]
    eps1 = (1 + cond_weight) * preds[:batch_size]
    eps2 = cond_weight * preds[batch_size:]
    x_t = eps1 - eps2

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t[:batch_size] * (
        x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
    )

    if t_index == 0:
        return model_mean, cross_map
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape, device=device)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise, cross_map


# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(
    model,
    classes,
    shape,
    cond_weight,
    get_cross_map=False,
    timesteps=None,
    device=None,
    betas=None,
    sqrt_one_minus_alphas_cumprod=None,
    sqrt_recip_alphas=None,
    posterior_variance=None,
    accelerator=None,
):  # to accelerate add timesteps
    # device = next(model.parameters()).device # to accelerate

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)  # to accelerate
    imgs = []
    cross_images_final = []

    if classes is not None:
        n_sample = classes.shape[0]
        context_mask = torch.ones_like(classes).to(device)
        # make 0 index unconditional
        # double the batch
        classes = classes.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 0.0  # makes second half of batch context free
        sampling_fn = partial(
            p_sample_guided,
            classes=classes,
            cond_weight=cond_weight,
            context_mask=context_mask,
            betas=betas,
            device=device,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
            accelerator=accelerator,
        )  # to accelerate betas
    else:
        sampling_fn = partial(p_sample)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img, cross_matrix = sampling_fn(model, x=img, t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i)
        imgs.append(img.cpu().numpy())
        cross_images_final.append(cross_matrix.cpu().numpy())
    if get_cross_map:
        return imgs, cross_images_final
    else:
        return imgs
    
@torch.no_grad()
def sample(
    model,
    image_size,
    classes=None,
    batch_size=16,
    channels=3,
    cond_weight=0,
    get_cross_map=False,
    timesteps=None,
    device=None,
    betas=None,
    sqrt_one_minus_alphas_cumprod=None,
    sqrt_recip_alphas=None,
    posterior_variance=None,
    accelerator=None,
):  # to accelerate add timesteps, device , betas
    return p_sample_loop(
        model,
        classes=classes,
        shape=(batch_size, channels, 4, image_size),
        cond_weight=cond_weight,
        get_cross_map=get_cross_map,
        timesteps=timesteps,
        device=device,
        betas=betas,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
        accelerator=accelerator,
    )  # to accelerate add timesteps device

# -- Schedulers --
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_end=0.005):
    beta_start = 0.0001

    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# -- Forward Diffusion --
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, device=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape).to(device)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(device)

    # print  (sqrt_alphas_cumprod_t , sqrt_one_minus_alphas_cumprod_t , t)
    # print (sqrt_alphas_cumprod_t.device, 'sqrt_alphas_cumprod_t')
    # print (x_start.device, 'x_start' )
    # print (sqrt_one_minus_alphas_cumprod_t.device , 'sqrt_one_minus_alphas_cumprod_t')
    # print (noise.device , 'noise.device')
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# -- Losses --
def p_losses(
    denoise_model,
    x_start,
    t,
    classes,
    noise=None,
    loss_type="l1",
    p_uncond=0.1,
    sqrt_alphas_cumprod_in=None,
    sqrt_one_minus_alphas_cumprod_in=None,
    device=None,
):
    device = x_start.device # to accelerate
    if noise is None:
        noise = torch.randn_like(x_start)  #  guass noise
    x_noisy = q_sample(
        x_start=x_start,
        t=t,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod_in,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod_in,
        noise=noise,
        device=device,
    )  # this is the auto generated noise given t and Noise
    print('max_q_sample', x_noisy.max(), 'mean_q_sample',x_noisy.mean() )

    context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - p_uncond)).to(device)
    # print ('context mask', context_mask)
    # print ('classes', classes)

    # mask for unconditinal guidance
    classes = classes * context_mask
    # nn.Embedding needs type to be long, multiplying with mask changes type
    classes = classes.type(torch.long)
    print ('final class',classes )
    # LOL: TODO HERE!!! index out of range => fixed due to num_classes different
    predicted_noise = denoise_model(x_noisy, t, classes)  # this is the predicted noise given the model and step t
    print('max_predicted', x_noisy.max(), 'mean_predicted',x_noisy.mean() )

    #predicted is ok (clipped)
    print ('predited inside loss')
    print (predicted_noise)
    print ('this is the noise generated by the p_losses')
    print (noise)
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        # print (noise.shape, 'noise' )
        # print (predicted_noise.shape, 'pred')
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# -- Building Blocks --
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # device = time.device # to accelerate
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        # embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # to accelerate
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)

        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):

    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    """
    Combine output with the original input
    """

    def forward(self, x):
        return x + self.convblock(x)  # skip connection


class ConvBlock_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 4, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # print ('x', x.shape)
        x = self.conv1(x)
        # print ('conv1', x.shape)
        x = self.conv2(x)
        # print ('conv2', x.shape)
        # x = F.avg_pool2d(x, 2)

        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.res = nn.Sequential(ResBlock(1, 2, 1), ResBlock(1, 2, 1), ResBlock(1, 2, 1), ResBlock(1, 2, 1))

        self.conv = nn.Sequential(
            ConvBlock_2d(in_channels=1, out_channels=2),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            ConvBlock_2d(in_channels=2, out_channels=4),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            ConvBlock_2d(in_channels=4, out_channels=1),
            nn.BatchNorm2d(1)
            # ConvBlock_2d(in_channels=1, out_channels=1),
            # ConvBlock_2d(in_channels=1, out_channels=1),
            # ConvBlock_2d(in_channels=1, out_channels=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(800, 800),
            # nn.GELU(),
            nn.BatchNorm1d(800),  # ALWAYS BATCHNORM THIS CHANGES A LOT THE RESULTS
            # nn.Linear(400, 400),
            # nn.BatchNorm1d(400),
            # nn.GELU(),
            # nn.BatchNorm1d(400),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(400, 800),
            # nn.GELU(),
            nn.BatchNorm1d(800),  # ALWAYS BATCHNORM THIS CHANGES A LOT THE RESULTS
            # nn.Linear(400, 400),
            # nn.GELU(),
            # nn.BatchNorm1d(400),
        )

        time_dim = 200 * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(100),
            nn.Linear(100, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.time_mlp_out = nn.Sequential(
            SinusoidalPositionEmbeddings(100),
            nn.Linear(100, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x, y):
        # print (x.shape, 'inside model x ')
        x_a = x.clone()
        # y_a = y.clone()
        x = self.res(x)

        # print ('to_full', x.shape)

        y_emb = self.time_mlp(y)
        # y_emb_out = self.time_mlp_out(y_a)

        x = x.view(-1, 800)

        x_a = x.view(-1, 800)
        x_a = self.fc(x_a)

        x = x + y_emb.view(-1, 800) * x_a

        # x = self.fc2(x)
        # x = x + y_emb_out.view(-1,400) + x_a

        # x = torch.clip(x, min=-1, max=1)
        x = x.view(-1, 1, 4, 200)

        # x = x.view(-1,1,200,4)

        # print (x.shape)
        # The cliping is working already checked
        return x
    
# -- Model --
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
# This session changed a lot to accomodate the conditional generation
from memory_efficient_attention_pytorch import Attention as EfficientAttention


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def l2norm(t):
    return F.normalize(t, dim=-1)


# small helper modules


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# positional embeds


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class ResnetBlockClassConditioned(ResnetBlock):
    def __init__(self, dim, dim_out, *, num_classes, class_embed_dim, time_emb_dim=None, groups=8):
        super().__init__(dim=dim + class_embed_dim, dim_out=dim_out, time_emb_dim=time_emb_dim, groups=groups)
        # print ('n_classes', num_classes, 'class_embed_dim', class_embed_dim)
        self.class_mlp = EmbedFC(num_classes, class_embed_dim)

    def forward(self, x, time_emb=None, c=None):
        # print ('before class_mlp')
        emb_c = self.class_mlp(c)
        # print ('after class_mlp')
        emb_c = emb_c.view(*emb_c.shape, 1, 1)
        emb_c = emb_c.expand(-1, -1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, emb_c], axis=1)

        return super().forward(x, time_emb)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class CrossAttention_lucas(nn.Module):
    def __init__(self, dim, heads=1, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b_y, c_y, h_y, w_y = y.shape

        qkv_x = self.to_qkv(x).chunk(3, dim=1)
        qkv_y = self.to_qkv(y).chunk(3, dim=1)

        q_x, k_x, v_x = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_x)

        q_y, k_y, v_y = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_y)

        q, k = map(l2norm, (q_x, k_y))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v_y)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model

# bit diffusion class


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log(
        (torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5
    )  # not sure if this accounts for beta being clipped to 0.999 in discrete version


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

# original
class Unet_lucas(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4),
        channels=1,
        resnet_block_groups=8,
        learned_sinusoidal_dim=18,
        num_classes=10,
        class_embed_dim=3,
        output_attention=False,
    ):
        super().__init__()

        # determine dimensions
        channels = 1
        self.channels = channels
        # if you want to do self conditioning uncomment this LOL: commented
        # input_channels = channels * 2
        input_channels = channels
        self.output_attention = output_attention

        # init_dim = 2048
        init_dim = default(init_dim, dim)
        # print (init_dim, 'init_dim')
        # self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3) # original
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7, 7), padding=3)

        # print (self.init_conv)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # print (dims)

        in_out = list(zip(dims[:-1], dims[1:]))
        # print (in_out)
        #         block_klass = partial(ResnetBlockClassConditioned, groups=resnet_block_groups,
        #                              num_classes=num_classes, class_embed_dim=class_embed_dim)

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            print(dim_out)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        # self.final_res_block = block_klass(1, dim, time_emb_dim = time_dim)

        # self.final_conv = nn.Conv2d(dim, channels, 1)
        self.final_conv = nn.Conv2d(dim, 1, 1)
        # print('self.final_conv' , self.final_conv)

        self.cross_attn = EfficientAttention(
            dim=200, dim_head=64, heads=1, memory_efficient=True, q_bucket_size=1024, k_bucket_size=2048 # LOL: changed 200<-2048
        )

        # mask = torch.ones(1, 65536).bool().cuda()

        print('dim', dim)
        self.norm_to_cross = nn.LayerNorm(dim * 4)
        
        # self.cross_attention = PreNorm(dim, CrossAttention_lucas(dim))
        print('final', dim, channels, self.final_conv)

    def forward(self, x, time, classes, x_self_cond=None):
        # print (x.shape ,'in_shape')
        # x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        # x = torch.cat((x_self_cond, x), dim = 1)
        # print ('UNET')
        # print ('classes inside unet',classes, 'time inside unet', time)
        x = self.init_conv(x)
        # print ('init_conv', x.shape)
        r = x.clone()

        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()
        t_cross = t_start.clone()

        # print ('t_cross shape', t_cross.shape)
        if classes is not None:
            t_start += self.label_emb(classes)
            t_mid += self.label_emb(classes)
            t_end += self.label_emb(classes)
            t_cross += self.label_emb(classes)
        # print ('t_cross shape', t_cross.shape)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)  # , classes)
            h.append(x)

            x = block2(x, t_start)  # , classes)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_mid)  # , classes)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)  # , classes)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)  # , classes)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)  # , classes)
            x = attn(x)

            x = upsample(x)

        # print('x torch_after_upsamples',x.shape)
        # print ('x_before_cat',  x.shape)
        x = torch.cat((x, r), dim=1)
        # print('x tochcat', x.shape)

        x = self.final_res_block(x, t_end)  # , classes)
        # print(self.final_res_block)
        # print('x from res_block before final_conv',x.shape)
        # print (self.final_conv(x).shape)
        x = self.final_conv(x)
        # print ('x_shape', x.shape)
        # PROBABLY i NEED TO RESHAPE THE t_cross befiore the cross attention, probably I wil need to do the same with the x and having 800 vectors as input (can I use matrix directly?)
        x_reshaped = x.reshape(-1, 4, 200) # LOL 200<-2048
        t_cross_reshaped = x.reshape(-1, 4, 200) # LOL 200<-2048
        # print (x_reshaped.shape,t_cross_reshaped.shape )
        # crossattention_out = self.cross_attention(x_reshaped, t_cross_reshaped)

        crossattention_out = self.cross_attn(
            self.norm_to_cross(x_reshaped.reshape(-1, 800)).reshape(-1, 4, 200), context=t_cross_reshaped # LOL 800<-8192, 200<-2048
        )  # (-1,1, 4, 200)
        crossattention_out = x.view(-1, 1, 4, 200) # LOL 200<-2048
        
        # print ('crossattention_out', crossattention_out.shape)
        # print ('x_shape before contrasseattention', x.shape)
        # print ('x', x.shape)
        x = x + crossattention_out
        # print ('FINAL X', x.shape)
        if self.output_attention:
            return x, crossattention_out
        return x

# LOL: accomodated
class Unet_lucas_2048(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4),
        channels=1,
        resnet_block_groups=8,
        learned_sinusoidal_dim=18,
        num_classes=10,
        class_embed_dim=3,
        output_attention=False,
    ):
        super().__init__()

        # determine dimensions
        self.dim = dim
        self.LENGTH = 2048
        channels = 1
        self.channels = channels
        # if you want to do self conditioning uncomment this LOL: commented
        # input_channels = channels * 2
        self.output_attention = output_attention

        init_dim = default(init_dim, dim)
        # print (init_dim, 'init_dim')
        # self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3) # original
        self.init_conv = nn.Conv2d(channels, init_dim, (7, 7), padding=3)

        # print (self.init_conv)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # print (dims)

        in_out = list(zip(dims[:-1], dims[1:]))
        # print (in_out)
        #         block_klass = partial(ResnetBlockClassConditioned, groups=resnet_block_groups,
        #                              num_classes=num_classes, class_embed_dim=class_embed_dim)

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            print(dim_out)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, init_dim, time_emb_dim=time_dim)
        # self.final_res_block = block_klass(1, dim, time_emb_dim = time_dim)

        # self.final_conv = nn.Conv2d(dim, channels, 1)
        self.final_conv = nn.Conv2d(dim, channels, 1)
        # print('self.final_conv' , self.final_conv)

        self.cross_attn = EfficientAttention(
            dim=dim, dim_head=64, heads=1, memory_efficient=True, q_bucket_size=1024, k_bucket_size=2048 # LOL: changed 200<-2048
        )

        # mask = torch.ones(1, 65536).bool().cuda()

        print('dim', dim)
        self.norm_to_cross = nn.LayerNorm(dim * 4)
        
        # self.cross_attention = PreNorm(dim, CrossAttention_lucas(dim))
        print('final', init_dim, channels, self.final_conv)

    def forward(self, x, time, classes, x_self_cond=None):
        print (x.shape ,'in_shape')
        # x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        # x = torch.cat((x_self_cond, x), dim = 1)
        # print ('UNET')
        # print ('classes inside unet',classes, 'time inside unet', time)
        x = self.init_conv(x)
        print ('init_conv', x.shape)
        r = x.clone()

        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()
        t_cross = t_start.clone()

        # print ('t_cross shape', t_cross.shape)
        if classes is not None:
            t_start += self.label_emb(classes)
            t_mid += self.label_emb(classes)
            t_end += self.label_emb(classes)
            t_cross += self.label_emb(classes)
        # print ('t_cross shape', t_cross.shape)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)  # , classes)
            h.append(x)

            x = block2(x, t_start)  # , classes)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_mid)  # , classes)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)  # , classes)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)  # , classes)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)  # , classes)
            x = attn(x)

            x = upsample(x)

        print('x torch_after_upsamples',x.shape)
        print ('x_before_cat',  x.shape)
        x = torch.cat((x, r), dim=1)
        print('x tochcat', x.shape)

        x = self.final_res_block(x, t_end)  # , classes)
        print(self.final_res_block)
        print('x from res_block before final_conv',x.shape)
        print (self.final_conv(x).shape)
        x = self.final_conv(x)
        print ('x_shape', x.shape)
        # PROBABLY i NEED TO RESHAPE THE t_cross befiore the cross attention, probably I wil need to do the same with the x and having 800 vectors as input (can I use matrix directly?)
        x_reshaped = x.reshape(-1, 4, self.dim) # LOL 200<-2048
        t_cross_reshaped = x.reshape(-1, 4, self.dim) # LOL 200<-2048
        print (x_reshaped.shape,t_cross_reshaped.shape )
        # crossattention_out = self.cross_attention(x_reshaped, t_cross_reshaped)

        crossattention_out = self.cross_attn(
            self.norm_to_cross(x_reshaped.reshape(-1, self.dim*4)).reshape(-1, 4, self.dim), context=t_cross_reshaped # LOL 800<-8192, 200<-2048
        )  # (-1,1, 4, 200)
        crossattention_out = x.view(-1, 1, 4, self.LENGTH) # LOL 200<-2048
        
        print ('crossattention_out', crossattention_out.shape)
        print ('x_shape before contrasseattention', x.shape)
        print ('x', x.shape)
        x = x + crossattention_out
        print ('FINAL X', x.shape)
        if self.output_attention:
            return x, crossattention_out
        return x
    
# -- Training Utils --
from pathlib import Path

def save_model(milestone, step, accelerator, opt, model, ema_model, folder_path_string=''):
    results_folder = Path(folder_path_string)

    data = {
        'step': step,
        'model': accelerator.get_state_dict(model),
        #'opt': opt.state_dict(),
        #'ema': ema_model.state_dict(),
        #'ema':accelerator.get_state_dict(ema_model)
    }

    torch.save(data, str(results_folder / f'model_{milestone}.pt'))


def recreating_models():
    model = Unet_lucas(dim=200, channels=1, dim_mults=(1, 2, 4), resnet_block_groups=4, num_classes=TOTAL_class_number)

    # ema = EMA(0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    # optimizer = Adam(model.parameters(), lr=1e-4)  #it was 1e-4  ()

    return model


def load_model(model_name, folder_path_string=''):
    results_folder = Path(folder_path_string)
    data = torch.load(str(results_folder / model_name))

    # model = accelerator.unwrap_model(model)
    model = recreating_models()
    model.load_state_dict(data['model'])

    step = data['step']
    # opt.load_state_dict(data['opt'])
    # ema_model.load_state_dict(data['ema'])

    return model, step


# -- Training Session --
from livelossplot import PlotLosses
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs

def train_accelerate():
    epochs = EPOCHS
    save_and_sample_every =SAVE_AND_SAMPLE_EVERY_N_EPOCH
    epochs_loss_show = EPOCH_LOSS_SHOW_INTERVAL
    image_size = IMAGE_SIZE
    channels = 1
    batch_size = BATCH_SIZE
    timesteps = TIME_STEPS

    print (MODEL_NAME)
    set_seed(12345)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], split_batches =True) # cpu=True

    # device = torch.device("cpu")
    device = accelerator.device
    device
    
    tf = T.Compose([T.ToTensor()])

    # seq_dataset = SequenceDataset(seqs=X_train, c=x_train_cell_type, transform=tf)
    seq_dataset = OHEDataset(data_path=DATA_PATH, device=device, transform=None) # LOL: change to OHE Dataset

    # MODIFIED BELOW
    train_dl = DataLoader(seq_dataset, batch_size, shuffle=True, num_workers=1, pin_memory=True, multiprocessing_context='fork' if torch.backends.mps.is_available() else None) # to accelerate

    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps, beta_end=0.2)
    betas = betas.to(device)
    print (torch.cuda.is_initialized())
    #betas = cosine_beta_schedule(timesteps=timesteps,  s=0.0001 )
    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    print (torch.cuda.is_initialized())
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    print (torch.cuda.is_initialized())
    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    print (torch.cuda.is_initialized() )      
    #sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    print (torch.cuda.is_initialized())
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    print (torch.cuda.is_initialized())
    
    print (device)

    model = Unet_lucas_2048(
        dim = 64,
        channels = channels,
        dim_mults = (1, 4, 32), # LOL dim=64, mults=(1, 4, 16, 64) ** tuple cannot have more than 4 elements
        resnet_block_groups = 4,
        num_classes=TOTAL_class_number
    )

    print ('model created') # MODIFIED BELOW
    model = model.to(device) # To accelerate

    optimizer = Adam(model.parameters(), lr=1e-4)  #it was 1e-4  ()

    print ('opt created')

    
    train_kl ,test_kl , shuffle_kl =1,1,1
    
    #live_kl = PlotLosses(groups={ 'KL':['train', 'test', 'shuffle'], 'DiffusionLoss':['loss'] }, outputs=[TensorboardLogger(logdir='tensorboard_logs/')])
    live_kl = PlotLosses(groups={ 'KL':['train', 'test', 'shuffle'], 'DiffusionLoss':['loss'] })

    print ('live_kl')

    #,'Diffusion_Loss':['loss']
    # live_kl.update({'train':train_kl, 'test':test_kl , 'shuffle':shuffle_kl , 'loss': 100})
    train_kl, test_kl , shuffle_kl = 1,1,1
    
    
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)   # To accelerate
    print ('preapare')

    
    if accelerator.is_main_process:
        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        print ('ema created')

    model.output_attention = False

    for epoch in tqdm(range(epochs)):
        model.train()
        for step, batch in enumerate(train_dl):
            print("Entering batch no:", step)
            
            #accelerator.print(f"epoch {epoch} {step}")
            x = batch.permute(0, 2, 3, 1) # LOL: adapt to our dataset
            print(batch.shape, batch.permute(0, 1, 3, 2).shape)
            y = torch.zeros(x.shape[0]) # what is this???
            # y = None # LOL: we have no classes (uncon)
            x = x.type(torch.float32).to(device)
            y = y.type(torch.long).to(device)
            print("(x.shape, y.shape)", x.shape, y.shape)
            # print (batch[0].dtype)
            batch_size = x.shape[0]
            #print (batch_size, 'batch_size')
            # batch_size = batch["pixel_values"].shape[0]
            # batch = batch["pixel_values"].to(device)
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            #t = torch.randint(0, timesteps, (batch_size,), device=device).long() # sampling a t to generate t and t+1 # to accelerate
            # t = torch.randint(0, timesteps, (batch_size,)).long() # sampling a t to generate t and t+1 LOL: this is the original
            t = torch.randint(0, TOTAL_class_number, (batch_size,)).long() # sampling a t to generate t and t+1

            #print (t.dtype)
            #loss = p_losses(model, batch, t, loss_type="l2")
            #print (y)
            
            with accelerator.autocast():
                loss = p_losses(model, x, t, y, loss_type="huber" , sqrt_alphas_cumprod_in=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod_in=sqrt_one_minus_alphas_cumprod, device=device)
                #loss.backward()
                print("Loss complete")

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.wait_for_everyone()
            optimizer.step()
            
            print("Update complete")

            accelerator.wait_for_everyone()
            if accelerator.is_main_process: 
                ema.step_ema(ema_model, model)


        if (epoch % epochs_loss_show) == 0:
            if accelerator.is_main_process: 
                live_kl.update({'train':train_kl, 'test':test_kl , 'shuffle':shuffle_kl , 'loss': loss.item()})
                live_kl.send()
                print(f" Epoch {epoch} Loss:", loss.item())

        # original sampling from the model is commented
        # save generated images
        # if epoch != 0 and epoch % save_and_sample_every == 0 and accelerator.is_main_process: # to accelerate add main process
        #     model.eval()

        #     print ('saving')
        #     milestone = step // save_and_sample_every
        #     sample_bs = 2
        #     #This needs to be fixed to the random
        #     sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))
        #     random_classes = sampled.to(device) #.c u d a()  to accelerate
        #     additional_variables = {'model':model,
        #                             'timesteps':timesteps,
        #                             'device':device,
        #                             'betas':betas,
        #                             'sqrt_one_minus_alphas_cumprod':sqrt_one_minus_alphas_cumprod,
        #                             'sqrt_recip_alphas':sqrt_recip_alphas,
        #                             'posterior_variance': posterior_variance,
        #                             'accelerator': accelerator,
        #                            'image_size':image_size}
        #     #This samples is created just to show the images in the screen
        #     samples = sample(classes=random_classes, batch_size=sample_bs, channels=1, cond_weight=1, **additional_variables) # to accelerate
        #     n_print = 0
        #     for image, class_show in zip(samples[-1], random_classes):
        #       if n_print < 4:

        #         plt.rcParams["figure.figsize"] = (20,1)
        #         pd_seq = pd.DataFrame(image.reshape(4,200))
        #         pd_seq.index = nucleotides
        #         sns.heatmap(pd_seq, linewidth=1, cmap='bwr', center=0) 
        #         plt.show()
        #         plt.rcParams["figure.figsize"] = (2,2)

        #         plt.bar(['a', 'c', 't', 'g'],pd_seq.mean(1).T)
        #         plt.title(f'Class: {class_show}')
        #         plt.show()
        #         n_print = n_print + 1
        #     synt_df = sampling_to_metric(int(NUMBER_OF_SAMPLING_TO_COMPARE_CELLS/10), additional_variables= additional_variables) #50 * batch 10 is the number of sampled seqs
        #     train_kl = compare_motif_list(synt_df, df_results_seq_guime_count_train)
        #     test_kl  = compare_motif_list(synt_df, df_results_seq_guime_count_test)
        #     shuffle_kl = compare_motif_list(synt_df, df_results_seq_guime_count_shuffle)
        #     live_kl.update({'train':train_kl, 'test':test_kl , 'shuffle':shuffle_kl , 'loss': loss.item()})
        #     live_kl.send()
        #     print('KL_TRAIN', train_kl  , 'KL' )
        #     print('KL_TEST',  test_kl  , 'KL' )
        #     print('KL_SHUFFLE',  shuffle_kl , 'KL' )
            
            use_model_name = MODEL_NAME
            
            if SAVE_MODELS_BY_EPOCH:
                use_model_name =  f'epoch_{str(epoch)}_'+ MODEL_NAME
            save_model(milestone= use_model_name, step = epoch, accelerator=accelerator, opt=optimizer, model=model,  ema_model=ema_model, folder_path_string='')# We can inject a different name in milestone if we want to save different models per epoch

            #probably I need to unwrap it here
            #torch.save(model,'model_1500_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k.model')


if __name__ == "__main__":
    train_accelerate()