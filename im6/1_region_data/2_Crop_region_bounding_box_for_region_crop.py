#!/usr/bin/env python
# coding: utf-8

# # Describe

# * 1. runs/val에 있는 preds 결과 값들을 모두 가져와서 Result_region_detect/images와  Result_region_detect/preds에 저장함.
# * 2. 그리고, 원본과 비교하여 detect 하지 못한 region을 찾음.
# * 3. detect된 bounding box를 기준으로 crop함. (region_detect_result/crop_results 폴더에 저장)

# ### runs/val/test_fold_{i} 폴더에 있는 labels값을 "../echo_data/Result_region_detect/preds" 폴더에 저장
# * GroupKFold 10 labels 모두 가져오기

# In[2]:


import os
import shutil

# 목적지 디렉토리 경로
dest_dir = "../2_circle_data/Result_region_detect/preds"

# 목적지 디렉토리가 없다면 생성
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 각 폴더에 대해 반복
for i in range(10):
    # 원본 디렉토리 경로
    src_dir = f"./yolov5/runs/val/test_fold_{i}/labels"
    
    # 원본 디렉토리에서 모든 txt 파일을 가져옴
    for filename in os.listdir(src_dir):
        if filename.endswith('.txt'):
            src_file_path = os.path.join(src_dir, filename)
            dest_file_path = os.path.join(dest_dir, filename)
            
            # 파일 복사
            shutil.copy(src_file_path, dest_file_path)


# ### ./images_origin 폴더에 있는 모든 이미지를 region_detect_result/images 폴더에 저장

# In[3]:


import os
import shutil

# 원본 이미지 파일들이 있는 디렉토리
src_dir = "./preprocessed_data/images"
# 이미지 파일들을 저장할 디렉토리
dest_dir = "../2_circle_data/Result_region_detect/images"

# 목적지 디렉토리가 없다면 생성
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 원본 디렉토리에서 모든 이미지 파일을 가져옴
for filename in os.listdir(src_dir):
    # 일반적인 이미지 파일 확장자
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        src_file_path = os.path.join(src_dir, filename)
        dest_file_path = os.path.join(dest_dir, filename)
        
        # 파일 복사
        shutil.copy(src_file_path, dest_file_path)


# ## Check Missing File

# In[4]:


import os

# 각 폴더의 파일명 리스트를 가져옵니다 (확장자 없이)
images_filenames = {os.path.splitext(f)[0] for f in os.listdir("../2_circle_data/Result_region_detect/images") if f.endswith(".jpg")}
preds_filenames = {os.path.splitext(f)[0] for f in os.listdir("../2_circle_data/Result_region_detect/preds") if f.endswith(".txt")}

# images 폴더에만 존재하는 파일명을 찾아 출력합니다
for filename in images_filenames - preds_filenames:
    print(f"'{filename}.jpg' is in 'images' but not in 'preds'")

# preds 폴더에만 존재하는 파일명을 찾아 출력합니다
for filename in preds_filenames - images_filenames:
    print(f"'{filename}.txt' is in 'preds' but not in 'images'")


# In[ ]:





# # Crop region bounding box
# * images 폴더: 원본 이미지
# * preds 폴더: 바운딩 박스 정보가 저장된 텍스트 파일 폴더 경로
# * line[5]: confidence score <br>
# 
# confidence score가 가장 높은 것을 선택해서 이미지를 생성함.

# In[6]:


import os
import cv2

def crop_bounding_boxes_from_folder(image_folder, txt_folder, output_folder, expansion_factor=1.2):
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, image_filename)
            txt_filename = os.path.splitext(image_filename)[0] + ".txt"
            txt_path = os.path.join(txt_folder, txt_filename)
            
            if os.path.exists(txt_path):
                image = cv2.imread(image_path)
                original_height, original_width, _ = image.shape

                highest_confidence = 0
                highest_confidence_bbox = None

                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    confidence = float(parts[5])  # 신뢰도 점수 추출
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        highest_confidence_bbox = parts[1:5]  # 신뢰도 점수가 가장 높은 바운딩 박스 저장
                        

                if highest_confidence_bbox:
                    x, y, w, h = map(float, highest_confidence_bbox)
                    new_x = int(x * original_width)
                    new_y = int(y * original_height)
                    new_w = int(w * original_width)
                    new_h = int(h * original_height)

                    x_min = max(0, int(new_x - new_w / 2 * expansion_factor))
                    y_min = max(0, int(new_y - new_h / 2 * expansion_factor))
                    x_max = min(original_width, int(new_x + new_w / 2 * expansion_factor))
                    y_max = min(original_height, int(new_y + new_h / 2 * expansion_factor))

                    cropped_image = image[y_min:y_max, x_min:x_max]
                    output_filename = f"{os.path.splitext(image_filename)[0]}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, cropped_image)
                    

# 이미지 파일 폴더 경로
image_folder = "../2_circle_data/Result_region_detect/images"

# 바운딩 박스 정보가 저장된 텍스트 파일 폴더 경로
txt_folder = "../2_circle_data/Result_region_detect/preds"

# Crop한 이미지를 저장할 폴더 경로
output_folder = "../2_circle_data/Result_region_detect/Result_crop_images"

# 목적지 디렉토리가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 주변 영역 확장 계수 (예: 1.5는 바운딩 박스 영역의 1.5배 크기까지 확장)
expansion_factor = 1.2

crop_bounding_boxes_from_folder(image_folder, txt_folder, output_folder, expansion_factor)


# * Create "./Result_region_detect/Result_crop_images" Folder

# # crop된 이미지를 기준으로 class_id 1에 대하여 좌표 재설정
# * im2-3에는 좌표 재설정을 할 필요가 없으므로 주석처리.
# * im6에는 좌표 재설정이필요함.

# In[8]:


import os
import cv2

# 폴더 경로 설정
images_origin_folder = "preprocessed_data/images"
labels_echo_folder = 'preprocessed_data/only_circle_labels'
region_detect_folder = '../2_circle_data/Result_region_detect/preds'
result_crop_images_folder = '../2_circle_data/Result_region_detect/Result_crop_images'
result_circle_updated_folder = '../2_circle_data/Result_region_detect/Result_circle_updated'

# 결과 저장 폴더가 없으면 생성
if not os.path.exists(result_circle_updated_folder):
    os.makedirs(result_circle_updated_folder)

# 원본 이미지 파일들의 이름을 가져옴
image_files = os.listdir(images_origin_folder)

for image_file in image_files:
    image_path = os.path.join(images_origin_folder, image_file)
    
    # 이미지 읽기
    img = cv2.imread(image_path)
    
    # 이미지 파일명에서 확장자를 제거하고, 라벨 파일 경로 생성
    label_name = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(labels_echo_folder, label_name)
    region_label_path = os.path.join(region_detect_folder, label_name)
    
    # 라벨 파일이 있는지 확인
    if os.path.exists(label_path) and os.path.exists(region_label_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        with open(region_label_path, 'r') as f:
            region_labels = f.readlines()
        
        region_count = 0
        for region_label in region_labels:
            
            _, x_center, y_center, width, height, _ = map(float, region_label.strip().split())
            
            x_min = int((x_center - width / 2) * img.shape[1])
            y_min = int((y_center - height / 2) * img.shape[0])
            x_max = int((x_center + width / 2) * img.shape[1])
            y_max = int((y_center + height / 2) * img.shape[0])
            
            # Crop된 이미지 저장 경로
            cropped_img_name = f"{os.path.splitext(image_file)[0]}.jpg"
            cropped_img_path = os.path.join(result_crop_images_folder, cropped_img_name)
            # Crop된 이미지의 라벨 저장 경로
            updated_label_name = f"{os.path.splitext(label_name)[0]}.txt"
            updated_label_path = os.path.join(result_circle_updated_folder, updated_label_name)
            
            # Crop된 영역 이미지 생성
            crop_img = img[y_min:y_max, x_min:x_max]
            cv2.imwrite(cropped_img_path, crop_img)
            
            with open(updated_label_path, 'w') as f:
                for label in labels:
                    class_id, echo_x_center, echo_y_center, echo_width, echo_height = map(float, label.strip().split())
                    
                    # 상대 좌표를 절대 좌표로 변환
                    abs_x_center = echo_x_center * img.shape[1]
                    abs_y_center = echo_y_center * img.shape[0]
                    abs_width = echo_width * img.shape[1]
                    abs_height = echo_height * img.shape[0]
                    
                    # 상대 좌표 및 크기 업데이트
                    new_x_center = (abs_x_center - x_min) / (x_max - x_min)
                    new_y_center = (abs_y_center - y_min) / (y_max - y_min)
                    new_width = abs_width / (x_max - x_min)
                    new_height = abs_height / (y_max - y_min)
                    
                    # 계산된 값을 파일에 쓰기
                    f.write(f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}\n")
            
            region_count += 1

print("Done !")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




