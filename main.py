import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
        if argv[i] == "-images_path" and i+1 < len(argv):
            images_path = argv[i+1]
        # Path to style models mapping with classes
        elif argv[i] in style_models:
            argv[i] = argv[i].replace('-', '')
            style_models[argv[i]] = argv[i+1]
            i += 1
        i += 1

    segment_image = SegmentData(
        model='fast_scnn', dataset='citys', weights_folder='./segmentation/weights',)
    masks, labels = segment_image.segment_image(images_path)

    assert os.path.isdir(images_path) or (os.path.isfile(
        images_path) and images_path.split('.')[-1].lower() in image_ext), "Image path is not valid"

    # Load style models with labels
    stylize = {}
    for label in labels:
        if label in style_models and len(style_models[label]) > 0:
            stylize[label] = Stylize(style_models[label])

    image_list: list[str] = []
    if os.path.isdir(images_path):
        for file in os.listdir(images_path):
            if file.split('.')[-1].lower() in image_ext:
                image_list.append(os.path.join(images_path, file))
    elif os.path.isfile(images_path):
        image_list.append(images_path)

    for image_path in image_list:
        image = np.asarray(Image.open(image_path).convert('RGB'))

        for mask, label in zip(masks, labels):
            if label in style_models and len(style_models[label]) > 0:
                stylized_image = np.asarray(
                    stylize[label].stylize_image(image))
                image[masks[:, :, 0] == 1] = stylized_image[masks[:, :, 0] == 1]
        # Save image to output folder
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        Image.fromarray(image).save(os.path.join(
            OUTPUT_FOLDER, os.path.basename(image_path)))
