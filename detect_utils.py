import torchvision.transforms as transforms
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import scipy.optimize
import matplotlib.pyplot as plt
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    # print('prediction')
    outputs = model(image) # get the predictions on the image
    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']

def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        if labels[i] == 1:
            # color = COLORS[labels[i]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0), 2
            )
            # cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
            #             lineType=cv2.LINE_AA)
    return image

def custom_bbox(gt_coords, img, imgname):
    cbbox_coords = []
    for k in range(len(gt_coords)): 
            if gt_coords[k][0] == imgname:
                box = [float(gt_coords[k][2]), float(gt_coords[k][3]), 50, 80]
                box = torch.tensor(box)
                bbox = box_center_to_corner(box)

                x1 = int(bbox[0].item())
                y1 = int(bbox[1].item())
                x2 = int(bbox[2].item())
                y2 = int(bbox[3].item())

                coords = [x1, y1, x2, y2]
                cbbox_coords.append(coords)
                    
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
    return img, cbbox_coords

def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.8 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.2 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                             height=bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=2)

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.0):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    print(n_true)
    print(n_pred)
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate((iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate((iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 


def findClosest(time, camera_time_list):
    val = min(camera_time_list, key=lambda x: abs(x - time))
    return camera_time_list.index(val)

def extract_frames(path ,file_name, model, model_name, min_size, savename, gt, device):
    #===== process the index files of camera 1 ======#
    with open('/home/dissana8/LAB/Visor/cam1/index.dmp') as f:
        content = f.readlines()
    cam_content = [x.strip() for x in content]
    c1_frames = []
    c1_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c1_frames.append(frame)
        c1_times.append(time)

    with open('/home/dissana8/LAB/Visor/cam2/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c2_frames = []
    c2_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c2_frames.append(frame)
        c2_times.append(time)
    

    # ===== process the index files of camera 3 ======#
    with open('/home/dissana8/LAB/Visor/cam3/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c3_frames = []
    c3_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1] + '.' + s[2])
        c3_frames.append(frame)
        c3_times.append(time)
    

    # ===== process the index files of camera 4 ======#
    with open('/home/dissana8/LAB/Visor/cam4/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c4_frames = []
    c4_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1] + '.' + s[2])
        c4_frames.append(frame)
        c4_times.append(time)
      
    #===== process the GT annotations  =======#
    with open("/home/dissana8/LAB/"+file_name) as f:
        content = f.readlines()
        

    content = [x.strip() for x in content]
    counter = -1
    print('Extracting GT annotation ...')
    for line in content:
        counter += 1
        if counter % 150 == 0:
            print(counter)
            s = line.split(" ")
            
            time = float(s[0])
            frame_idx = findClosest(time, c1_times) # we have to map the time to frame number
            c1_frame_no = c1_frames[frame_idx]
            

            frame_idx = findClosest(time, c2_times)  # we have to map the time to frame number
            c2_frame_no = c2_frames[frame_idx]
            

            frame_idx = findClosest(time, c3_times)  # we have to map the time to frame number
            c3_frame_no = c3_frames[frame_idx]

            
            frame_idx = findClosest(time, c4_times)  # we have to map the time to frame number
            c4_frame_no = c4_frames[frame_idx]

            cam = []

            cam.append('/home/dissana8/LAB/Visor/cam1/'+c1_frame_no)
            cam.append('/home/dissana8/LAB/Visor/cam2/'+c2_frame_no)
            cam.append('/home/dissana8/LAB/Visor/cam3/'+c3_frame_no)
            cam.append('/home/dissana8/LAB/Visor/cam4/'+c4_frame_no)

            f, ax = plt.subplots(1, 4, figsize=(25, 4))

            for i in range(4):
                img = cv2.imread(cam[i])
                # sized = cv2.resize(img, (min_size, min_size))
                # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

                # for j in range(2):  # This 'for' loop is for speed check
                #             # Because the first iteration is usually longer
                #     boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

                boxes, classes, labels = predict(img, model, device, 0.8)
                # print('predicting image')

                imgfile = cam[i].split('/')[6:]
                imgname = '/'.join(imgfile)
                sname = savename + imgname

                image = draw_boxes(boxes, classes, labels, img)

                # img, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)

                image, cbbox = custom_bbox(gt[i], img, imgname)

                # print(bbox)

                if cbbox:
                    cbbox = np.array(cbbox)
                    bbox = np.array(boxes)
                    idx_gt_actual, idx_pred_actual, ious_actual, label = match_bboxes(cbbox, bbox)

                    for h in range(len(idx_gt_actual)):
                        t = idx_gt_actual[h]
                        text_c = cbbox[t]
                        print(text_c)
                        img = cv2.putText(img, str(round(ious_actual[h], 3)), (text_c[0], text_c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


                    # iou = get_iou(bbox, cbbox)
                    # print("iou")
                    # print(len(iou))

                    # for k in range(len(iou)):
                    #     img = cv2.putText(img, str(iou[k][1]), (iou[k][0][0], iou[k][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



                ax[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            savepath = "/home/dissana8/Faster-R-CNN/custom_bbox_"+model_name+"/"+c1_frame_no.split('/')[0]

            if not os.path.exists(savepath):
                os.makedirs(savepath)

            plt.savefig(savepath+"/"+c1_frame_no.split('/')[-1])
            ax[0].cla()
            ax[1].cla()
            ax[2].cla()
            ax[3].cla()
