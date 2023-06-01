import glob
import re
import time
import os

from PIL import Image
import matplotlib.pyplot as plt

# Torch imports
import torch
from torch import Tensor, uint8
from torchvision import transforms, datasets
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformer_net import TransformerNet
from vgg import Vgg16
import utils
from train import train_model


# transformer = TransformerNet().to(torch.device('cuda'))
# transformer.train()
# dataset_path = f'./VOCtrainval_11-May-2012'
dataset_path = f'./Images'
style_image_path = f'./style_images/candy.jpg'
trainNet = train_model(dataset_path)
trainNet.train([40, 40], dataset_path, style_image_path,
               epochs=500, log_interval=10, batch_size=4)


def stylize(content_image: str, model: str = './model/trained_model.model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = Image.open(content_image).convert('RGB')
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model)
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
    save_image = Image.fromarray(
        (output[0].permute((1, 2, 0)*255)).int().detach().cpu().numpy())
    save_image.save('stylized_image.jpg')
