import os
import glob
import time
import ultralytics
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from IPython.display import display, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

ultralytics.checks()
DEVICE = 'cuda'

#####################################################################################
# FUNCTIONS
#####################################################################################
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))




#####################################################################################
# Models setup
#####################################################################################
model_yolo = YOLO('model/yolov8n_custom.pt')
model_yolo.to(DEVICE)
sam_checkpoint = "model/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(DEVICE)
predictor = SamPredictor(sam)
#####################################################################################
# Datasets setup (img)
#####################################################################################
IMAGES_DIR = os.path.join('dataset', 'segment', 'images')
IMAGES_PATH_LIST = sorted(glob.glob(IMAGES_DIR+'/*'))
IMAGES_LIST = sorted(os.listdir(IMAGES_DIR))
MAST_DIR = os.path.join('dataset', 'segment', 'masks')


#####################################################################################
# generate masks
#####################################################################################
for i, ids in enumerate(IMAGES_PATH_LIST):
    s_t = time.time()
    results = model_yolo.predict(ids, conf=0.25)
    for result in results:
        boxes = result.boxes
    bbox = boxes.xyxy.tolist()[0]
    print(ids)
    print(bbox)

    image = cv2.cvtColor(cv2.imread(ids), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_box = np.array(bbox)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    e_t = time.time()
    print(e_t-s_t)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks[0], plt.gca())
    # show_box(input_box, plt.gca())
    # plt.axis('off')
    # plt.show()
    mask_image = (masks[0] * 255).astype(np.uint8)  # Convert to uint8 format
    cv2.imwrite(MAST_DIR + '/' + IMAGES_LIST[i], mask_image)

