# Fantastic Beasts Datasets: Benchmark in *AttrSeg: Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation*

[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/chaofanma/Fantastic-Beasts)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/chaofanma/AttrSeg)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202023-red)](https://arxiv.org/pdf/2309.00096)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**🤗 [View on Hugging Face](https://huggingface.co/datasets/chaofanma/Fantastic-Beasts)** | **💻 [GitHub Repository](https://github.com/chaofanma/AttrSeg)**

This repository contains the collected dataset used in the NeurIPS 2023 paper: AttrSeg: Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation. [See the paper here](https://arxiv.org/pdf/2309.00096).

![poster](assets/poster.png)



## Brief Introduction

Existing datasets often lack the inclusion of rare or obscure vocabulary.
To address this limitation, we manually curated a dataset titled "Fantastic Beasts", which consists of 20 categories of magical creatures from the film series *Fantastic Beasts and Where to Find Them*.
This dataset is designed for comprehensive evaluation and simulating real-world scenarios, specifically for two common situations where attribute descriptions are essential:

**Neologisms**: Vanilla category names represent new vocabularies that are often unseen by large language models (LLMs) and vision-language pre-trainings (VLPs).

**Unnameability**: When users encounter unfamiliar objects, they may struggle to name them, particularly in the case of rare or obscure categories.

For more details, please refer to the [paper](https://arxiv.org/pdf/2309.00096).



## How to Use This Dataset

### Method 1: Using Hugging Face Datasets

**Load directly from Hugging Face Hub (with embedded images in Parquet format):**

```python
from datasets import load_dataset

dataset = load_dataset("chaofanma/Fantastic-Beasts", split='test')

sample = dataset[0]
sample['image'].show()  # PIL Image, ready to use
print(sample['category'])  # "Augurey"
```

🤗 **[View full dataset on Hugging Face](https://huggingface.co/datasets/chaofanma/Fantastic-Beasts)**

The Hugging Face version uses Parquet format with embedded images for optimal performance and easy loading.



### Method 2: Using PyTorch Dataset (From Source Files)

If you prefer direct file access or need more control, you can use the custom PyTorch Dataset class:

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class FantasticBeastsDataset(Dataset):
    def __init__(self, img_root, msk_root, attr_json, transform=None):
        self.img_root = img_root
        self.msk_root = msk_root
        with open(attr_json, 'r') as f:
            self.attr = json.load(f)
        self.transform = transform
        self.categories = ['Augurey', 'Billywig', 'Chupacabra', 'Diricawl', 'Doxy', 
                          'Erumpent', 'Fwooper', 'Graphorn', 'Grindylow', 'Kappa', 
                          'Leucrotta', 'Matagot', 'Mooncalf', 'Murtlap', 'Nundu', 
                          'Occamy', 'Runespoor', 'Swoopingevil', 'Thunderbird', 'Zouwu']
        self.img_pathes = self.get_pathes(self.img_root)
        self.msk_pathes = self.get_pathes(self.msk_root)

    def get_pathes(self, root):
        img_pathes = []
        for category in self.categories:
            category_path = Path(root) / category
            for img_file in category_path.glob("*"):
                img_pathes.append(img_file.resolve().as_posix())
        img_pathes.sort()
        return img_pathes

    def read_img(self, img_path):
        img = np.array(Image.open(img_path))  # uint8 (h, w, 3)
        return img
    
    def read_msk(self, msk_path):
        msk = np.array(Image.open(msk_path))  # uint8 (h, w)
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

# Usage
dataset = FantasticBeastsDataset(
    img_root="./images",
    msk_root="./masks",
    attr_json="./attributes.json"
)

for img, msk, attr in dataset:
    print(img.shape, msk.shape, len(attr))
```

**Full implementation**: See [`examples/fantastic_beasts_dataset.py`](examples/fantastic_beasts_dataset.py)



## Dataset Structure

### Category Names and Attributes
There are 20 categories in Fantastic Beasts dataset, listed as below in alphabetical order:
```
Augurey, Billywig, Chupacabra, Diricawl, Doxy, Erumpent, Fwooper, Graphorn, Grindylow, Kappa, Leucrotta, Matagot, Mooncalf, Murtlap, Nundu, Occamy, Runespoor, Swoopingevil, Thunderbird, Zouwu
```
The class names and their corresponding attributes are stored in `attributes.json`.

### Dataset Files
- **images/**: 251 images organized by category (20 subdirectories)
- **masks/**: 251 corresponding binary segmentation masks (PNG format)
- **attributes.json**: Maps each category to its attribute descriptions
- **examples/fantastic_beasts_dataset.py**: PyTorch Dataset implementation



### Data Fields
- `image`: PIL Image of the magical creature (RGB mode)
- `mask`: PIL Image of binary segmentation mask (L mode, grayscale; 0 for background, 255 for object)
- `category`: Category name (one of 20 magical creature types)
- `attributes`: List of textual attribute descriptions for the category




## Citation
If this dataset is useful for your research, please consider citing:
```
@article{ma2023attrseg,
  title   = {AttrSeg: Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation},
  author  = {Chaofan Ma and Yuhuan Yang and Chen Ju and Fei Zhang and Ya Zhang and Yanfeng Wang},
  journal = {Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  year    = {2023}
}
```



## Acknowledgements

We would like to thank the following people for their direct or indirect contributions to the creation of this dataset:
- J.K. Rowling, as the creator of the Wizarding World and the original author of the Harry Potter series, whose work is foundational.
- David Yates, the director of the film, for contributing to its vision and execution.
- David Heyman, the producer of the film, for his pivotal role in bringing the story to the screen.
- The VFX artists and technicians at Framestore and their team leaders, Tim Burke, Christian Manz, and Pablo Grillo, for their incredible work in creating the magical creatures.
- All the Harry Potter fans who support me in creating this dataset.


