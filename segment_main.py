import os
import sys
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision import transforms

from segmentation.fast_scnn_model import get_fast_scnn
from segmentation.segment_train import Trainer, parse_args

if __name__ == '__main__':
    args = sys.argv[1:]
    train: bool = False

    for arg in args:
        if arg == 'train_segment':
            train = True

    if train:
        trainer = Trainer(parse_args(
            'fast_scnn', 'citys', 1024, 512, 'train'))
        trainer.train()


train_ids_dict = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "license plate",
}


class SegmentData():
    def __init__(self, model='fast_scnn', dataset='citys', weights_folder='./weights', outdir='./test_result', cpu=False):
        self.model = model
        self.dataset = dataset
        self.weights_folder = weights_folder
        self.outdir = outdir
        self.cpu = cpu

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not cpu else "cpu")

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = get_fast_scnn(self.dataset, pretrained=True,
                                   root=self.weights_folder, map_cpu=self.cpu).to(self.device)
        self.model.eval()

    def segment_image(self, input_image):
        image = Image.open(input_image).convert('RGB')
        image = image.resize((2048, 1024))
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)

        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        unique_labels = np.unique(pred)
        masks = np.zeros(
            (1024, 2048, unique_labels.shape[0]))
        labels: list[str] = []
        for i in range(unique_labels.shape[0]):
            masks[:, :, i] = pred == unique_labels[i]
            labels.append(train_ids_dict[unique_labels[i]])
        return masks, labels

# Call the demo function with the desired arguments
# seg = SegmentData(model='fast_scnn', dataset='citys', weights_folder='./segmentation/weights',
#                   outdir='./test_result', cpu=False)
# for i in tqdm(range(100)):
# seg.segment_image(
#     # './datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')
#     './datasets/citys/leftImg8bit/train/bremen/bremen_000001_000019_leftImg8bit.png')
# segment(model='fast_scnn', dataset='citys', weights_folder='./weights',
#         input_pic='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
#         outdir='./test_results', cpu=False)
