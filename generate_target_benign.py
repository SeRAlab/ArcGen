'''
Author: Ashlars yangzhonghao@vip.qq.com
Date: 2025-05-16 15:53:35
LastEditors: Ashlars yangzhonghao@vip.qq.com
LastEditTime: 2025-05-19 14:24:36
FilePath: /ArcGen/generate_target_benign.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import os
from datetime import datetime
import json
import argparse
import config

from utils.dataloader import get_dataloader
from utils.train import unmodified_training, eval_model
from utils.get_model import get_model, get_datainfo



def main():
    arg = config.get_arguments().parse_args()
    torch.cuda.set_device(1)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(arg.device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    get_datainfo(arg)

    # Dataset
    train_dl, train_transform = get_dataloader(arg, defender=False, train=True, pretensor_transform=True)
    test_dl, test_transform = get_dataloader(arg, defender=False, train=False)
    print("Train Data Size: ", len(train_dl.dataset))
    print("Test Data Size: ", len(test_dl.dataset))

    Model = get_model(arg)

    SAVE_PREFIX = os.path.join(arg.save_dir, arg.dataset)
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(os.path.join(SAVE_PREFIX, 'target_benign')):
        os.mkdir(os.path.join(SAVE_PREFIX, 'target_benign'))

    for i in range(arg.target_num):
        if arg.model == 'vit_b_16' or arg.model == 'vit_l_32' or arg.model == 'vit_b_32':
            model = Model.to("cuda")
        else:
            model = Model().to("cuda")
        print('using model: ', arg.model)
        unmodified_training(arg, model, train_dl, arg.epoch, verbose=arg.verbose)

        save_dir = os.path.join(SAVE_PREFIX, 'target_benign', f'{arg.model}_{i:04}')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        torch.save(model, os.path.join(save_dir, 'model.pth'))
        acc = eval_model(model, test_dl)
        print ("Acc %.4f, saved to %s @ %s"%(acc, save_dir, datetime.now()))

        info = {
            'model': arg.model,
            'dataset': arg.dataset,
            'test_acc': acc,
            'input_resolution': (arg.input_height, arg.input_width),
        }

        json_info = json.dumps(info)
        with open(os.path.join(save_dir, 'info.json'), 'w') as f:
            f.write(json_info)


if __name__ == "__main__":
    main()
