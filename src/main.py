#%%
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from datasets import PTB_Dataset, TensorDataset
from diffusion_network import UNet

import wandb
import torch
import numpy as np
from pathlib import Path
from typing import Literal, Tuple
import random
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_wandb(api_key, project: str, config: dict) -> wandb:
    wandb.login(key=api_key)
    wandb.init(project=project, entity="ecg_simula", config=config)
    return wandb

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

############# CONFIGURATION ###############
train_configuration = {
    'DATASET_OPTION': 'PTB',              
    'LEARNING_RATE': 0.0003,
    'BATCH_SIZE': 32,
    'EPOCHS': 301,
}

PLOTS_FOLDER = "plots"
ACTION = 'train' 
USE_WEIGHTS_AND_BIASES = True
WANDB_KEY = '796a636ca8878cd6c1494d1282f73496c43e6b31'  # add your API key here
############################################
         
if USE_WEIGHTS_AND_BIASES:
    wandb = init_wandb(WANDB_KEY, f'{train_configuration["DATASET_OPTION"]}', train_configuration)

def create_folder_if_not_exists(folder: str):
    if not(Path(folder).is_dir()):
        Path(folder).mkdir(parents=True, exist_ok=True)

create_folder_if_not_exists(PLOTS_FOLDER)

def create_and_save_plot(generated_leads_II_VIII, filename, file_extension='.png'):
    LINE_WIDTH = 0.3
    fig, axs = plt.subplots(4, 2, figsize=(18, 12))

    for i in range(4):
        for j in range(2):
            axs[i, j].yaxis.set_major_locator(MultipleLocator(0.5))
            axs[i, j].yaxis.set_minor_locator(AutoMinorLocator(4))
            axs[i, j].grid(which='major', color='#CCCCCC', linestyle='--', linewidth=LINE_WIDTH)
            axs[i, j].grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=LINE_WIDTH / 2)

    for i in range(3):
        axs[i+1, 0].plot(generated_leads_II_VIII[i], linewidth=LINE_WIDTH)
    for i in range(4):
        axs[i, 1].plot(generated_leads_II_VIII[i+3], linewidth=LINE_WIDTH)
    fig.savefig(Path(filename + file_extension))
    plt.close(fig)


def fix_seed():
    random_seed = 123
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


if ACTION=='train':
    fix_seed()


class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=5000, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 7, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        # x = x.clamp(-1, 1)
        # x = (x * 255).type(torch.uint8)
        return x

def get_dataloader(target='train', batch_size=train_configuration['BATCH_SIZE'], shuffle=True) -> Tuple[TensorDataset, torch.utils.data.DataLoader]:
    dataset = PTB_Dataset(data_dirs=train_configuration['DATASET_OPTION'], target=target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, drop_last=True)
    return dataset, dataloader


mse = torch.nn.MSELoss()
diffusion = Diffusion(device=device)


def train_diffusion(train_dataloader):
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_configuration['LEARNING_RATE'], betas=(0.5, 0.999))
    train_loss_plot = []
    model.train()
    for epoch in range(train_configuration['EPOCHS']):
        logging.info(f"Starting epoch {epoch}:")
        train_loss_average = 0
        for batch, (leadI, leadsII_VIII) in enumerate(train_dataloader,0):
            leadsII_VIII = leadsII_VIII.to(device)
            t = diffusion.sample_timesteps(leadsII_VIII.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(leadsII_VIII, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_average += loss.cpu()
        train_loss_average /= len(train_dataloader)

        sampled_ecg = diffusion.sample(model, n=1)
        create_and_save_plot(sampled_ecg[0].cpu().detach().numpy(), filename=f'{PLOTS_FOLDER}/ecg{epoch}')

        if USE_WEIGHTS_AND_BIASES:
            plot_filename = f"{PLOTS_FOLDER}/ecg{epoch}"
            wandb.log({"MSE": train_loss_average})
            wandb.log({"ECG": wandb.Image(plot_filename + ".png")}) 

        else:
            train_loss_plot.append(train_loss_average)
            print(f'Finished epoch {epoch}. Average loss for this epoch: {train_loss_average:05f}')


if ACTION == "train":
    train_dataset, train_dataloader = get_dataloader(target='train')
    validation_dataset, validation_dataloader = get_dataloader(target='validation', batch_size=1, shuffle=False)
    train_diffusion(train_dataset)

