import torchvision
import numpy
import torch
import argparse
import cv2
import detect_utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# construct the argument parser
parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                    min_size=args['min_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "/home/dissana8/LAB/"
file_name = 'LAB-GROUNDTRUTH.ref'
savename = '/home/dissana8/Faster-R-CNN/outputs/'

gt = []
gt.append(np.load('/home/dissana8/LAB/data/LAB/cam1_coords__.npy', allow_pickle=True))
gt.append(np.load('/home/dissana8/LAB/data/LAB/cam2_coords__.npy', allow_pickle=True))
gt.append(np.load('/home/dissana8/LAB/data/LAB/cam3_coords__.npy', allow_pickle=True))
gt.append(np.load('/home/dissana8/LAB/data/LAB/cam4_coords__.npy', allow_pickle=True))

fig, a = plt.subplots(4, 1)
detect_utils.extract_frames(path, file_name, model, args['min_size'], savename, gt, device)

# image = Image.open(args['input'])
# image = cv2.imread(args['input'])
# model.eval().to(device)
# boxes, classes, labels = detect_utils.predict(image, model, device, 0.8)
# image = detect_utils.draw_boxes(boxes, classes, labels, image)
# # cv2.imshow('Image', image)
# # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
# save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
# cv2.imwrite(f"outputs/{save_name}.jpg", image)
# cv2.waitKey(0)

