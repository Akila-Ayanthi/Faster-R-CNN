import torchvision
import numpy
import torch
import argparse
import cv2
import detect_utils
from PIL import Image

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                    min_size=args['min_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image = Image.open(args['input'])
image = cv2.imread(args['input'])
model.eval().to(device)
boxes, classes, labels = detect_utils.predict(image, model, device, 0.8)
image = detect_utils.draw_boxes(boxes, classes, labels, image)
# cv2.imshow('Image', image)
# image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)

# import torch
# import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
#          # load a pre-trained model for classification and return
#          # only the features
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#         # >>> # FasterRCNN needs to know the number of
#         # >>> # output channels in a backbone. For mobilenet_v2, it's 1280
#         # >>> # so we need to add it here
# backbone.out_channels = 1280
#         # >>> # let's make the RPN generate 5 x 3 anchors per spatial
#         # >>> # location, with 5 different sizes and 3 different aspect
#         # >>> # ratios. We have a Tuple[Tuple[int]] because each feature
#         # >>> # map could potentially have different sizes and
#         # >>> # aspect ratios
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
#         # >>>
#         # >>> # let's define what are the feature maps that we will
#         # >>> # use to perform the region of interest cropping, as well as
#         # >>> # the size of the crop after rescaling.
#         # >>> # if your backbone returns a Tensor, featmap_names is expected to
#         # >>> # be ['0']. More generally, the backbone should return an
#         # >>> # OrderedDict[Tensor], and in featmap_names you can choose which
#         # >>> # feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
#         # put the pieces together inside a FasterRCNN model
# model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
# model.eval()
# # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# image = cv2.imread('input/horses.jpg')
# predictions = model(image)