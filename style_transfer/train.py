import time
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Torch imports
import torch
from torchvision import transforms, datasets
from torch.optim import Adam
from torch.utils.data import DataLoader

from .vgg import Vgg16
from .transformer_net_modified_2 import TransformerNet
from .utils import *


class train_model():

    def __init__(self, dataset_path: str) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_path = dataset_path
        self.transformer = TransformerNet().to(self.device)
        self.transformer.train()

    def train(self, image_size: tuple[int], style_image_path: list[str], style_name: str,
              save_model_path: str = './models', content_weight: float = 1e5,
              style_weight: float = 1e10, batch_size: int = 16, lr: float = 0.001,
              epochs: int = 500, log_interval: int = 10):
        assert image_size, 'Image size cannot be None'
        assert (isinstance(image_size, tuple) or isinstance(image_size, list)) and len(
            image_size) == 2, 'Only list or tuple of length 2 allowed for image sizes'
        assert all(isinstance(s, int) or isinstance(s, float)
                   for s in image_size), 'Image size must be int or float'

        # check dataset path
        assert isinstance(self.dataset_path, str) and os.path.isdir(
            self.dataset_path)
        print('directories', os.listdir(self.dataset_path))
        no_directories: bool = True
        for f in os.listdir(self.dataset_path):
            if os.path.isdir(os.path.join(self.dataset_path, f)):
                print(f)
                no_directories = False
                break
        assert not no_directories, 'Dataset must contain directories'

        # check model path is string
        assert isinstance(save_model_path, str)

        # check batch size
        assert isinstance(batch_size, int) and batch_size >= 1

        # check learning rate
        assert (isinstance(lr, int) or isinstance(lr, float)) and lr != 0

        # check epochs
        assert isinstance(epochs, int) and epochs > 0

        self.transformer.train().to(self.device)
        optimizer = Adam(self.transformer.parameters(), lr)
        mse_loss = torch.nn.MSELoss()

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        style_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        # Load data
        train_dataset = datasets.ImageFolder(self.dataset_path, transform)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        styles = [Image.open(image_path).convert('RGB')
                  for image_path in style_image_path]
        styles = [style_transform(style) for style in styles]
        styles = [style.repeat(batch_size, 1, 1, 1).to(self.device)
                  for style in styles]

        # Load VGG
        vgg = Vgg16(requires_grad=False).to(self.device)
        styles_features = [vgg(normalize_batch(style))
                           for style in styles]
        gram_styles = [[gram_matrix(
            y) for y in style_features] for style_features in styles_features]

        start_time = time.time()
        print('Style image')
        plt.imshow(styles[0][0].permute((1, 2, 0)).detach().cpu().numpy())
        plt.show()

        print('Starting training')
        for e in tqdm(range(epochs)):
            agg_content_loss: float = 0.
            agg_style_loss: float = 0.
            for x, _ in train_loader:
                optimizer.zero_grad()

                x = x.to(self.device)
                x = normalize_batch(x)
                features_x = vgg(x)

                y = self.transformer(x)
                y = normalize_batch(y)
                features_y = vgg(y)

                content_loss = content_weight * \
                    mse_loss(features_y.relu2_2, features_x.relu2_2)

                style_loss = 0.
                style_index = torch.randint(0, len(gram_styles), (1,))[0]
                for ft_y, gm_s in zip(features_y, gram_styles[style_index]):
                    gm_y = gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s[:len(x)])
                style_loss *= style_weight

                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

            if e % log_interval == 0 and e != 0:
                print('Epoch {}:\n\tTime {}\n\tContent loss: {}\n\tStyle loss: {}'.format(
                    e, time.time()-start_time, agg_content_loss, agg_style_loss))

            if e % (5*log_interval) == 0 and e != 0:
                y[0] -= torch.min(y[0])
                display_image = (y[0]/torch.max(y[0]))*255
                display_image = display_image.permute(
                    (1, 2, 0)).int().detach().cpu().numpy()
                print('Original image')
                plt.imshow(x[0].permute((1, 2, 0)).cpu().numpy())
                plt.show()
                print('Styled image after {} epochs'.format(e))
                plt.imshow(display_image)
                plt.show()

        # save model
        self.transformer.eval().cpu()
        save_model_filename = "{}_trained_model.model".format(style_name)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        save_model_path = os.path.join(save_model_path, save_model_filename)
        torch.save(self.transformer.state_dict(), save_model_path)

        print("\nDone, trained model saved at", save_model_path)
        print('Time taken: {} s'.format(time.time()-start_time))
