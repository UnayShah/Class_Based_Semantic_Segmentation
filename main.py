import sys
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from segment_main import SegmentData
from style_transfer.stylize import Stylize

style_models = {
    '-road': '',
    '-sidewalk': '',
    '-building': '',
    '-wall': '',
    '-fence': '',
    '-pole': '',
    '-traffic light': '',
    '-traffic sign': '',
    '-vegetation': '',
    '-terrain': '',
    '-sky': '',
    '-person': '',
    '-rider': '',
    '-car': '',
    '-truck': '',
    '-bus': '',
    '-train': '',
    '-motorcycle': '',
    '-bicycle': '',
    '-license plate': '',
}

image_ext = ['jpg', 'jpeg', 'png', 'bmp']
OUTPUT_FOLDER = './output'

if __name__ == "__main__":
    argv = sys.argv[1:]
    images_path = ""
    i = 0
    while i < len(argv):
        # Path to images folder
        print(argv[i])
        if argv[i] == "-images_path" and i+1 < len(argv):
            images_path = argv[i+1]
            i += 1
        # Path to style models mapping with classes
        elif argv[i] in style_models:
            del style_models[argv[i]]
            argv[i] = argv[i].replace('-', '')
            style_models[argv[i]] = argv[i+1]
            i += 1
        i += 1

    # Load models
    segment_image = SegmentData(
        model='fast_scnn', dataset='citys', weights_folder='./segmentation/weights',)

    stylize = {}
    print(style_models.keys())
    for key in style_models.keys():
        if key.startswith('-'):
            continue
        stylize[key] = Stylize(style_models[key])

    assert os.path.isdir(images_path) or (os.path.isfile(
        images_path) and images_path.split('.')[-1].lower() in image_ext), "Image path is not valid"

    # Load style models with labels

    image_list: list[str] = []
    if os.path.isdir(images_path):
        for file in os.listdir(images_path):
            if file.split('.')[-1].lower() in image_ext:
                image_list.append(os.path.join(images_path, file))
    elif os.path.isfile(images_path):
        image_list.append(images_path)

    for image_path in tqdm(image_list):

        masks, labels = segment_image.segment_image(image_path)

        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image = image.resize((2048, 1024))
        image = np.asarray(image)

        for mask_i, label in enumerate(labels):
            if label in style_models and len(style_models[label]) > 0:
                stylized_image = np.asarray(
                    stylize[label].stylize_image(image))
                image[masks[:, :, mask_i] ==
                      1] = stylized_image[masks[:, :, mask_i] == 1]
        image = Image.fromarray(image)
        image = image.resize(original_size)
        # Save image to output folder
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        image.save(os.path.join(OUTPUT_FOLDER, os.path.basename(image_path)))
