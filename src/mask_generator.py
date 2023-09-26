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
import pandas as pd

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

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


#####################################################################################
# Models setup
#####################################################################################
model_yolo = YOLO('model/yolov8n_custom_20230926_v2.pt')
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
MASK_DIR = os.path.join('dataset', 'segment', 'masks')
MASK_DIR2 = os.path.join('dataset', 'segment', 'masks_convert')
MASK_DIR3 = os.path.join('dataset', 'segment', 'masks_TF')

create_directory(IMAGES_DIR)
create_directory(MASK_DIR)
create_directory(MASK_DIR2)
create_directory(MASK_DIR3)

CROP_IMG_DIR = os.path.join('dataset', 'segment', 'image_crop')
CROP_MASK_DIR = os.path.join('dataset', 'segment', 'mask_crop')
create_directory(CROP_IMG_DIR)
create_directory(CROP_MASK_DIR)

NO_SAVE_LOG_PATH = 'dataset/segment'
DF_LOG = pd.DataFrame({'file name': [], 'idx': []})

#####################################################################################
# generate masks
#####################################################################################
for i, ids in enumerate(IMAGES_PATH_LIST):
    s_t = time.time()
    results = model_yolo.predict(ids, conf=0.25)
    for result in results:
        boxes = result.boxes

    try:
        bbox = boxes.xyxy.tolist()[0]
    except:
        os.remove(ids)
        DF_LOG.loc[len(DF_LOG)] = [ids, str(i)]
        print("delete the file", IMAGES_PATH_LIST)

        continue

    print('(', i, '/', len(IMAGES_PATH_LIST) , ')', ids, ' ', bbox)
    print(bbox)

    # image = cv2.cvtColor(cv2.imread(ids), cv2.COLOR_BGR2RGB)
    image = cv2.imread(ids)
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
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # show_mask(masks[0], plt.gca())
    # show_box(input_box, plt.gca())
    # plt.axis('off')
    # plt.show()

    # save mask 255
    mask_image = (masks[0] * 255).astype(np.uint8)  # Convert to uint8 format
    cv2.imwrite(MASK_DIR + '/' + IMAGES_LIST[i], mask_image)
    #
    # # save mask 0
    mask_image = (255 - masks[0] * 255).astype(np.uint8)  # Convert to uint8 format
    cv2.imwrite(MASK_DIR2 + '/' + IMAGES_LIST[i], mask_image)

    # save mask True and False
    mask_image = masks[0].astype(np.uint8)
    cv2.imwrite(MASK_DIR3 + '/' + IMAGES_LIST[i], mask_image)


    ########## CROP ##############
    # crop image of bbox - raw image
    print(input_box[0].astype(np.uint16))
    print(input_box[1].astype(np.uint16))
    print(input_box[2].astype(np.uint16))
    print(input_box[3].astype(np.uint16))

    # img_crop = image.crop((input_box[0], input_box[1], input_box[2], input_box[3]))
    img_crop = image[input_box[1].astype(np.uint16):input_box[3].astype(np.uint16),
                     input_box[0].astype(np.uint16):input_box[2].astype(np.uint16)]
    cv2.imwrite(CROP_IMG_DIR + '/' + IMAGES_LIST[i], img_crop)

    # crop image of bbox - mask image
    mask_crop = (masks[0] * 255).astype(np.uint8)[input_box[1].astype(np.uint16):input_box[3].astype(np.uint16),
                                                  input_box[0].astype(np.uint16):input_box[2].astype(np.uint16)]
    cv2.imwrite(CROP_MASK_DIR + '/' + IMAGES_LIST[i], mask_crop)

    print("done")
DF_LOG.to_csv(NO_SAVE_LOG_PATH + '/no_save_log.csv', index=False)




