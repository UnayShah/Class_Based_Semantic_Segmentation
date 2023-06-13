import os
import torch
import torchvision.transforms as transforms
from torchvision import transforms
from PIL import Image
from utils.visualize import get_color_pallete
from fast_scnn_model import get_fast_scnn
import numpy as np
import matplotlib.pyplot as plt


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
        image = self.transform(image).unsqueeze(0).to(self.device)

        print('Finished loading model!')

        with torch.no_grad():
            outputs = self.model(image)

        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()

        original_image = np.asarray(Image.open(input_image).convert('RGB'))
        unique_labels = np.unique(pred)
        masks = np.zeros(
            (original_image.shape[0], original_image.shape[1], unique_labels.shape[0]))
        for i in range(unique_labels.shape[0]):
            masks[:, :, i] = pred == unique_labels[i]
        return masks

# Call the demo function with the desired arguments
seg = SegmentData(model='fast_scnn', dataset='citys', weights_folder='./weights',
                  outdir='./test_result', cpu=False)
# for i in tqdm(range(100)):
seg.segment_image(
    './datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')
# segment(model='fast_scnn', dataset='citys', weights_folder='./weights',
#         input_pic='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
#         outdir='./test_results', cpu=False)
