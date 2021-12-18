import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os, nltk
import numpy as np

from miscc.config import cfg, cfg_from_file
import pprint
import datetime
import dateutil.tz

from utils.data_utils import CUBDataset
from utils.trainer import trainer

# Set a config file as 'train_birds.yml' in training, as 'eval_birds.yml' for evaluation
cfg_from_file('cfg/train_birds.yml') # eval_birds.yml

print('Using config:')
pprint.pprint(cfg)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_dir = 'sample/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

imsize = cfg.TREE.BASE_SIZE * (4 ** (cfg.TREE.BRANCH_NUM - 1))
image_transform = transforms.Compose([
    transforms.Resize(int(imsize)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

train_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='train')
test_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='test')

print(f'train data directory:\n{train_dataset.split_dir}')
print(f'test data directory:\n{test_dataset.split_dir}\n')

print(f'# of train filenames:{train_dataset.filenames.shape}')
print(f'# of test filenames:{test_dataset.filenames.shape}\n')

print(f'example of filename of train image:{train_dataset.filenames[0]}')
print(f'example of filename of valid image:{test_dataset.filenames[0]}\n')

print(f'example of caption and its ids:\n{train_dataset.captions[0]}\n{train_dataset.captions_ids[0]}\n')
print(f'example of caption and its ids:\n{test_dataset.captions[0]}\n{test_dataset.captions_ids[0]}\n')

print(f'# of train captions:{np.asarray(train_dataset.captions).shape}')
print(f'# of test captions:{np.asarray(test_dataset.captions).shape}\n')

print(f'# of train caption ids:{np.asarray(train_dataset.captions_ids).shape}')
print(f'# of test caption ids:{np.asarray(test_dataset.captions_ids).shape}\n')


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))


## 2. Define models and go to train/evaluate

algo = trainer(output_dir, train_dataset, train_dataloader, test_dataset, test_dataloader, train_dataset.ixtoword, test_dataset.ixtoword)

if cfg.TRAIN.FLAG:
    algo.train()
else:
    algo.generate_eval_data()
## 3. Measure Inception score and R-precision of given test dataset

#After set the config file as 'eval_birds.yml' and run the 'algo.generate_eval_data()',
# the synthesized images based on given captions and set of image and caption features
# should be saved inside a 'evaluation' folder, specifically in 'evaluation/generated_images/..'.

#**Then, go to the 'evaluation' folder and run each 'inception_score.ipynb' and 'r_precision.ipynb' file
# in order to measure inception score and r-precision score.**
