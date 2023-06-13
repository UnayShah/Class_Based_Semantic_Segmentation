# import os
# import torch
# from fast_scnn_model import FastSCNN


# def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
#     acronyms = {
#         'pascal_voc': 'voc',
#         'pascal_aug': 'voc',
#         'ade20k': 'ade',
#         'coco': 'coco',
#         'citys': 'citys',
#     }
#     from cityscape_dataloader import datasets
#     model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
#     if pretrained:
#         if (map_cpu):
#             model.load_state_dict(torch.load(os.path.join(
#                 root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
#         else:
#             model.load_state_dict(torch.load(os.path.join(
#                 root, 'fast_scnn_%s.pth' % acronyms[dataset])))
#     return model
