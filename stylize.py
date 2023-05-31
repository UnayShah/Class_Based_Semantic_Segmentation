import cv2
import os
import torch
from torchvision import transforms
from numpy import uint8
import gc

from transformer_net import TransformerNet

model: str = './model/trained_model (1).model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    style_model = TransformerNet()
    state_dict = torch.load(model)
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    style_model.eval()


def video_stylize(filepath: str, model: str = './model/trained_model.model'):
    if os.path.exists(filepath):
        video = cv2.VideoCapture(filepath)
    else:
        raise Exception('Check file path')

    while video.isOpened():
        ret, frame = video.read()

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        content_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((256, 256)),
            # transforms.Lambda(lambda x: x.mul(255)),
        ])
        content_image = content_transform(frame)
        content_image = content_image.unsqueeze(0).to(device)
        output = style_model(content_image)
        output -= torch.min(output)
        output = (output/torch.max(output))*255

        styled_image = output[0].permute(
            (1, 2, 0)).int().cpu().detach().numpy().astype(uint8)
        cv2.imshow('video', styled_image)
        cv2.waitKey()
        torch.cuda.empty_cache()
        gc.collect()
        # plt.imshow(styled_image)
        break
    video.release()


video_stylize('./sample_video/sample_video_small.mp4', './model/mosaic.pth')
