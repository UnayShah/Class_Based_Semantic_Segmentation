import cv2
from os.path import isfile, exists
from os import mkdir
from torch import device, load, cuda, no_grad, min, max
from torchvision import transforms
from numpy import uint8
import gc
from PIL import Image

from numpy import ndarray
from .transformer_net import TransformerNet


class Stylize():
    def __init__(self, model_path: str) -> None:
        assert isinstance(model_path, str), 'Model path must be a string'

        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.model_path = model_path
        with no_grad():
            self.style_model = TransformerNet()
            state_dict = load(self.model_path)
            self.style_model.load_state_dict(state_dict)
            self.style_model.to(self.device)
            self.style_model.eval()

    def stylize_image(self, content_image) -> Image.Image:
        assert isinstance(content_image, Image.Image) or isinstance(
            content_image, ndarray), 'Content image must be PIL Image Image or Numpy ndarray'

        # Preprocess image
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(self.device)

        output = self.style_model(content_image)
        
        # output -= min(output)
        # output = (output/max(output))*255
        output[0][output[0]<0] = 0
        output[0][output[0]>255] = 255

        save_image = Image.fromarray(output[0].permute(
            (1, 2, 0)).int().cpu().numpy().astype(uint8))
        return save_image

    def stylize_video(self, filepath: str):
        if isfile(filepath):
            video = cv2.VideoCapture(filepath)
        else:
            raise Exception('Check file path')
        if not exists('temp'):
            mkdir('temp')
        i = 0
        while video.isOpened():
            i += 1
            _, frame = video.read()

            output = self.stylize_image(frame)
            output.save('temp/{}.jpg'.format(str(i)))
            cuda.empty_cache()
            gc.collect()
        video.release()

    # video_stylize('./sample_video/sample_video_small.mp4', './model/mosaic.pth')
