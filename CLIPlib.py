import torch
import clip
from PIL import Image

# 모델과 프로세서를 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
#model, preprocess = clip.load("RN50x64", device=device)
# ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px, RN50, RN101, RN50x4, RN50x16, RN50x64

# 이미지를 로드하고 전처리
image_path = "/home/alisa/Documents/Datasets/FFHQ/images1024x1024/00000/00001.png"  # 이미지 파일 경로를 지정하세요.
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)  # 이미지 파일 경로를 지정하세요.
# image = preprocess(Image.open("/home/alisa/Documents/Datasets/FFHQ/images1024x1024")).unsqueeze(0)  # 이미지 파일 경로를 지정하세요.

# 텍스트 준비
texts = ["a photo of man", "a photo of woman"]  # 비교할 텍스트 리스트
# 텍스트 토크나이즈 및 전처리
text_inputs = clip.tokenize(texts).to(device)

'''
# 이미지 인코딩
image_features = model.encode_image(image)

# 텍스트 인코딩
text_features = model.encode_text(clip.tokenize(texts))
'''
# torch.no_grad() 사용: 모델 인코딩 부분에서는 torch.no_grad() 문을 사용하여 그래디언트 계산을 비활성화 (이는 메모리 사용을 줄이고 계산 속도를 높이는 데 도움이 됩니다.)
# 이미지 인코딩
with torch.no_grad():
    image_features = model.encode_image(image)

# 텍스트 인코딩
with torch.no_grad():
    text_features = model.encode_text(text_inputs)


# 이미지와 텍스트 간의 유사성 계산
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(similarity) 
print(image_features.shape, text_features.shape, text_inputs.shape)
# 각각 텍스트랑 유사도