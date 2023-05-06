import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import logging
import numpy as np
from imageio import imread

class RadarSets(Dataset):
    def __init__(self, pics, path, img_size, mode='train'):
        super(RadarSets, self).__init__()
        self.pics = pics
        self.mode = mode
        self.height, self.width = img_size
        os.sep = '/'
        self.train_path = path

    def __getitem__(self, index):
        if self.mode not in ['test']:
            mode = 'train'
        else:
            mode = 'test'
        inputs = []
        input_fn = self.pics[index]
        for i in range(0, 10):
            img = Image.open(self.train_path + '/radar_' +input_fn[i])
            img = np.pad(img, ((0, 0), (0, 0)), 'constant', constant_values = (0, 0))
            img = np.array(Image.fromarray(img).resize((self.width, self.height)))
            img = torch.from_numpy(img.astype(np.float32))
            inputs.append(img)

        # 前面先把inputs的数据读进来
        # 后面如果是训练(train)或者验证(valid)集则把target的数据也读进来，并返回inputs和target数据
        # 如果是测试集(test)则不读target,还是只用inputs数据，只返回inputs数据

        if self.mode in ['train', 'valid']:
            targets = []
            for i in range(10, 20):
                img = Image.open(self.train_path + '/radar_' +input_fn[i])
                img = np.pad(img, ((0, 0), (0, 0)), 'constant', constant_values = (0, 0))
                img = np.array(Image.fromarray(img).resize((self.width, self.height)))
                img = torch.from_numpy(img.astype(np.float32))
                targets.append(img)

            return torch.stack(inputs, dim=0).unsqueeze(1)/255, torch.stack(targets, dim=0).unsqueeze(1)/255
        
        elif self.mode in ['test']:
            return torch.stack(inputs, dim=0)/255
        
        else:
            raise ValueError(f'{self.mode} is unknown and should be among train, valid and test!')

    def __len__(self):
        return len(self.pics)
