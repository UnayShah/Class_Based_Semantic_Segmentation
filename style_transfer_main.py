from style_transfer.stylize import Stylize
from style_transfer.train import train_model
import sys
from os.path import isdir, isfile, join, exists
from os import listdir, mkdir
from PIL import Image
from tqdm import tqdm

DEFAULT_STYLE_PATH: str = './style_transfer/style_images'
DEFAULT_TRAIN_PATH: str = './style_transfer/Images'
DEFAULT_TO_STYLE_PATH: str = './style_transfer/to_style'
DEFAULT_MODEL_PATH: str = './style_transfer/models/mosaic_trained_model.model'

if __name__ == '__main__':
    args = sys.argv[1:]
    style_path: str = DEFAULT_STYLE_PATH
    dataset_path: str = DEFAULT_TRAIN_PATH
    to_style_path: str = DEFAULT_TO_STYLE_PATH
    model_path: str = DEFAULT_MODEL_PATH
    train: bool = False
    style: bool = False
    single_style: bool = False

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
        elif args[i] == '-model':
            model_path = args[i+1]
            i += 2
            continue
        elif args[i] == '-tostyle':
            to_style_path = args[i+1]
            i += 2
            continue
        elif args[i] == '-single_style':
            single_style = True
            i += 1
        else:
            print(
                'Received unknown argument \'{}\'. Ignoring...'.format(args[i]))
            i += 1

    if train:
        assert isdir(style_path) or isfile(
            style_path), 'Given style path does not exist'
        assert isdir(dataset_path), 'Given train path does not exist'
        assert isdir(
            style_path) if single_style else True, 'Train multiple is set but model path is not a directory'

        style_list: list[str] = []
        style_names: list[str] = []

        # Read files from style path or populate style list
        # Style names stores the name of image without rest fo the path and extension
        if isdir(style_path):
            style_names = list(filter(lambda x: str(x).split('.')
                                      [-1] in ['jpg', 'jpeg', 'png'], listdir(style_path)))
            style_list = [join(style_path, s) for s in style_names]
        elif isfile(style_path):
            style_names = [style_path.split('/')[-1]]
            style_list.append(style_path)

        style_names = [name.split('.')[0] for name in style_names]

        if single_style:
            style_name = style_path.split('/')[-1]
            trainNet = train_model(dataset_path)
            trainNet.train([200, 200], style_list, style_name,
                           epochs=500, log_interval=10, batch_size=25)
        else:
            for name, s in zip(style_names, style_list):
                print('Using style image: {}'.format(s))
                trainNet = train_model(dataset_path)
                trainNet.train([200, 200], [s], name,
                               epochs=300, log_interval=10, batch_size=25)
    if style:
        print(to_style_path)
        assert isdir(to_style_path), 'Given images path does not exist'

        stylize = Stylize(model_path)
        model_name: str = model_path.replace('\\','/').split('/')[-1].split('.')[0]
        for f in tqdm(listdir(to_style_path)):
            image_path = join(to_style_path, f)
            content_image = Image.open(image_path).convert('RGB')
            processed = stylize.stylize_image(content_image)
            if not exists('./style_transfer/styled'):
                mkdir('./style_transfer/styled')
            if not exists(join('./style_transfer/styled', model_name)):
                mkdir(join('./style_transfer/styled', model_name))
            processed.save(join('./style_transfer/styled', model_name, f))

    # stylize.stylize_video('./sample_video/sample_video_small.mp4')
