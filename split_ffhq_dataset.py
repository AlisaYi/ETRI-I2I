import os
import shutil

# 경로 설정
base_dir = '/home/alisa/Documents/Datasets/FFHQ/images1024x1024'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# train, val 폴더 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 이미지 파일들을 분류 (이미지 복사를 원하면 shutil.copy)
for folder in os.listdir(base_dir):                 # 모든 파일 및 폴더 이름을 리스트로 반환
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):                  # 현재 항목이 디렉토리인지 확인
        for filename in os.listdir(folder_path):    # 모든 파일 이름을 리스트로 반환
            if filename.endswith('.png'):       # 파일 이름이 .png로 끝나는지 확인
                file_num = int(filename.split('.')[0])      # 파일 이름에서 숫자 부분을 추출하여 정수로 변환
                if 0 <= file_num <= 59999:                  # 파일 번호가 0부터 59999 사이인지 확인
                    shutil.move(os.path.join(folder_path, filename), os.path.join(train_dir, filename))
                else:
                    shutil.move(os.path.join(folder_path, filename), os.path.join(val_dir, filename))

print("Dataset successfully split into train and val folders.")
