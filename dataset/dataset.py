# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

# Im2ImDataset
### 데이터셋 두 개의 이미지 리스트(imgs_A와 imgs_B)를 받아서 각 리스트에서 이미지 쌍을 선택하여 변환을 적용하는 PyTorch Dataset 클래스
class Im2ImDataset(Dataset):
    def __init__(self, imgs_A, imgs_B, transform, repeat=None):

        self.imgs_A = imgs_A
        self.imgs_B = imgs_B
        self.transform = transform
        if repeat is not None:
            self.repeat = repeat
        else:
            self.repeat = None

    def __len__(self):
        return max(len(self.imgs_A), len(self.imgs_B))
        
    def __getitem__(self, index):

        inputs = {}
        if (len(self.imgs_A) > len(self.imgs_B)):
            index_A = index
            index_B = index % len(self.imgs_B)
        else:
            index_A = index % len(self.imgs_A)
            index_B = index

        img_A_path = self.imgs_A[index_A]
        img_B_path = self.imgs_B[index_B]

        with open(img_A_path, 'rb') as f:
            img_A = Image.open(f).convert('RGB')
        with open(img_B_path, 'rb') as f:
            img_B = Image.open(f).convert('RGB')

        if self.repeat is not None:
            cropped_imgs_A = []
            for _ in range(self.repeat):
                img_A_crop = self.transform(img_A)
                cropped_imgs_A.append(img_A_crop)
                inputs['A'] = cropped_imgs_A
        else:
            img_A = self.transform(img_A)
            inputs['A'] = img_A
            
        img_B = self.transform(img_B)
        inputs['B'] = img_B

        inputs['A_name'] = img_A_path.split('/')[-1]
        inputs['B_name'] = img_B_path.split('/')[-1]
        return inputs
