import sys
from PIL import Image
import matplotlib.pyplot as plt

from segment import SegmentData
from style_transfer.stylize import Stylize
cbs_args = {
    "image_path": "",
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    image_path = ""
    for i in range(len(argv)):
        if argv[i] == "-image_path" and i+1 < len(argv):
            image_path = argv[i+1]

    segment_image = SegmentData(
        model='fast_scnn', dataset='citys', weights_folder='./segmentation/weights',)
    masks, labels = segment_image.segment_image(image_path)
    
    stylize = Stylize('./style_transfer/models/udnie_trained_model.model')
    image = Image.open(image_path).convert('RGB')
    stylized_image = stylize.stylize_image(image)
    image[masks[:, :, 0] == 1] = stylized_image[masks[:, :, 0] == 1]
    plt.imshow(image)
    plt.show()
