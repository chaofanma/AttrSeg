import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class FantasticBeastsDataset(Dataset):
    def __init__(self, img_root, msk_root, attr_json, transform=None):
        self.img_root = img_root
        self.msk_root = msk_root
        with open(attr_json, 'r') as f:
            self.attr = json.load(f)
        self.transform = transform
        self.categories = ['Augurey', 'Billywig', 'Chupacabra', 'Diricawl', 'Doxy', 'Erumpent', 'Fwooper', 'Graphorn', 'Grindylow', 'Kappa', 'Leucrotta', 'Matagot', 'Mooncalf', 'Murtlap', 'Nundu', 'Occamy', 'Runespoor', 'Swoopingevil', 'Thunderbird', 'Zouwu']
        self.img_pathes = self.get_pathes(self.img_root)
        self.msk_pathes = self.get_pathes(self.msk_root)
        assert len(self.img_pathes) == len(self.msk_pathes)

    def get_pathes(self, root):
        img_pathes = []
        for category in self.categories:
            category_path = Path(root) / category
            for img_file in category_path.glob("*"):
                img_pathes.append(img_file.resolve().as_posix())
        img_pathes.sort()
        return img_pathes

    def read_img(self, img_path):
        img = np.array(Image.open(img_path)) #img are uint8 with size (h,w,3)
        return img
    
    def read_msk(self, msk_path):
        msk = np.array(Image.open(msk_path)) #msk are uint8 with size (h,w)
        msk[msk > 0] = 1
        return msk
    
    def read_attr(self, category):
        return self.attr[category]

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, index):
        img_path = self.img_pathes[index]
        msk_path = self.msk_pathes[index]
        img = self.read_img(img_path)
        msk = self.read_msk(msk_path)
        attr = self.read_attr(Path(img_path).name.split('_')[0])

        if self.transform:
            img, msk = self.transform(img, msk)

        return img, msk, attr




if __name__ == "__main__":
    fb = FantasticBeastsDataset(img_root="./images/", msk_root="./masks/", attr_json="./attributes.json")
    for img, msk, attr in tqdm(fb):
        print(img.shape, msk.shape, attr)

