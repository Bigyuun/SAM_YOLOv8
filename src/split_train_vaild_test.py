import os
import random
import shutil
from sklearn.model_selection import train_test_split

# 이미지와 텍스트 파일의 디렉토리 경로 설정
image_dir = 'dataset/images_20230926-033738/images'
txt_dir = 'dataset/images_20230926-033738/masks_txt_label'
output_dir = 'dataset/images_20230926-033738/split_trainvalidtest_images'

# Train, Validation, Test 비율 설정
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15


# 이미지와 텍스트 파일 리스트 가져오기
image_files = os.listdir(image_dir)
txt_files = os.listdir(txt_dir)

# 이미지와 텍스트 파일을 한 쌍으로 묶음
data_pairs = [(image, image.replace('.png', '.txt')) for image in image_files if image.endswith('.png')]

# 데이터 분할
train_data, temp_data = train_test_split(data_pairs, test_size=(1 - train_ratio))
validation_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + validation_ratio))

# 각 세트의 폴더 생성
for dataset, dataset_name in [(train_data, 'train'), (validation_data, 'validation'), (test_data, 'test')]:
    image_dataset_path = os.path.join(output_dir, 'images', dataset_name)
    os.makedirs(image_dataset_path, exist_ok=True)

    txt_label_dataset_path = os.path.join(output_dir, 'labels', dataset_name)
    os.makedirs(txt_label_dataset_path, exist_ok=True)

    # 이미지 및 텍스트 파일 복사
    for image, txt in dataset:
        shutil.copy(os.path.join(image_dir, image), os.path.join(image_dataset_path, image))
        shutil.copy(os.path.join(txt_dir, txt), os.path.join(txt_label_dataset_path, txt))
