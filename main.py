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


def load_image(filename: str) -> Tensor:
    image = Image.open(filename)
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255).to(uint8)),
    ])
    transformed_image = image_transforms(image)
    return transformed_image


def get_style_model(style_number: int):
    model_file = glob.glob('styles/*.pth')[style_number]
    transformer = TransformerNet().to(torch.device('gpu'))
    # load model
    state_dict = torch.load(model_file)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    transformer.load_state_dict(state_dict)
    transformer.to(0)
    transformer.eval()
    return transformer


print(type(load_image('./style_images/candy.jpg')))
transformer = TransformerNet().to(torch.device('cuda'))
transformer.train()
# dataset_path = f'./VOCtrainval_11-May-2012'
dataset_path = f'./Images'
style_image_path = f'./style_images/candy.jpg'


def train(image_size, dataset_path: str, style_image_path: str, checkpoint_model_dir: str = './checkpoint',
          save_model_dir: str = './model', content_weight: float = 1e5, style_weight: float = 1e10,
          batch_size: int = 8, lr: float = 0.001, epochs: int = 50, log_interval: int = 10):
    device = 'cuda'
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(dataset_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(style_image_path)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    vgg = Vgg16(requires_grad=False).to(device)
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * \
                mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if checkpoint_model_dir is not None and (batch_id + 1) % epochs//5 == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + \
                    str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(
                    checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        content_weight) + "_" + str(style_weight) + ".model"
    save_model_path = os.path.join(save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


train([100, 100], dataset_path, style_image_path)
