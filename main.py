from stylize import Stylize
from train import train_model
import sys
from os.path import isdir, isfile, join
from os import listdir
from PIL import Image
import cv2

DEFAULT_STYLE_PATH: str = './style_images'
DEFAULT_TRAIN_PATH: str = './Images'
DEFAULT_TO_STYLE_PATH: str = './to_style'


if __name__ == '__main__':
    print(sys.argv)
    args = sys.argv[1:]
    style_path: str = DEFAULT_STYLE_PATH
    dataset_path: str = DEFAULT_TRAIN_PATH
    to_style_path: str = DEFAULT_TO_STYLE_PATH
    train: bool = False
    style: bool = False

    i = 0
    while i < len(args):
        if args[i] == '-styles':
            style_path: str = args[i+1]
            i += 2
            continue
        elif args[i] == '-train':
            dataset_path: str = args[i+1]
            i += 2
            continue
        elif args[i] == 'train':
            train = True
            i += 1
            continue
        elif args[i] == 'style':
            style = True
            i += 1
            continue
        i += 1

    if train:
        assert isdir(style_path) or isfile(
            style_path), 'Given style path does not exist'
        assert isdir(dataset_path), 'Given train path does not exist'

        style_list: list[str] = []
        if isdir(style_path):
            style_list = list(filter(lambda x: str(x).split('.')
                                     [-1] in ['jpg', 'jpeg', 'png'], listdir(style_path)))
            style_list = [join(style_path, s) for s in style_list]
        elif isfile(style_path):
            style_list.append(style_path)

        trainNet = train_model(dataset_path)
        for s in style_list:
            print(s)
            trainNet.train([20, 20], dataset_path, s,
                           epochs=10, log_interval=10, batch_size=2)
    if style:
        assert isdir(to_style_path), 'Given images path does not exist'

        stylize = Stylize('./model/trained_model.model')
        content_image = Image.open(
            './Images/Images/2007_000032.jpg').convert('RGB')
        # content_image = cv2.imread('./Images/Images/2007_000032.jpg')
        # content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        stylize.stylize_image(content_image)
    # stylize.stylize_video('./sample_video/sample_video_small.mp4')
