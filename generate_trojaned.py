import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import os
from datetime import datetime
import json
import argparse
import config
from PIL import Image

from utils.dataloader import get_dataloader
from utils.train import unmodified_training, eval_model
from utils.get_model import get_model, get_datainfo
from utils.attack import get_attackinfo, get_trigger_info, poisoning_func




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
    test_dl_benign, test_transform = get_dataloader(arg, defender=False, train=False, poison=False)


    Model = get_model(arg)

    SAVE_PREFIX = os.path.join(arg.save_dir, arg.dataset)
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(os.path.join(SAVE_PREFIX, 'trojaned')):
        os.mkdir(os.path.join(SAVE_PREFIX, 'trojaned'))

    for i in range(arg.target_num):
        get_attackinfo(arg)
        # get trigger
        trigger_info = get_trigger_info(arg)

        train_dl, train_transform = get_dataloader(arg, defender=False, train=True, poison=True, trigger_info=trigger_info ,pretensor_transform=True, poisoning_func=poisoning_func)
        test_dl_trojan, test_transform = get_dataloader(arg, defender=False, train=False, poison=True, trigger_info=trigger_info, poisoning_func=poisoning_func)
        print("train data size: ", len(train_dl.dataset))
        print("test benign data size: ", len(test_dl_benign.dataset))
        print("test trojan data size: ", len(test_dl_trojan.dataset))
        model = Model().to("cuda")
        print('using model: ', arg.model)
        print(i)
        unmodified_training(arg, model, train_dl, arg.epoch, arg.verbose)


        save_dir = os.path.join(SAVE_PREFIX, 'trojaned', f'{arg.model}_{arg.attack_type}_{i:04}')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        torch.save(model, os.path.join(save_dir, 'model.pth'))
        acc = eval_model(model, test_dl_benign)

        if arg.attack_mode == "alltoone":
            acc_mal = eval_model(model, test_dl_trojan)
        elif arg.attack_mode == "alltoall":
            acc_mal = 1 - eval_model(model, test_dl_trojan)
        print(f'benign acc: {acc:.4f}, trojan acc: {acc_mal:.4f}, save to {save_dir} @ {datetime.now()}')
        p_size, pattern, loc, alpha, delta, frequency, horizontal_or_vertical = trigger_info
        info = {
            'model': arg.model,
            'dataset': arg.dataset,
            'test_acc': float(acc),
            'test_acc_mal': float(acc_mal),
            'input_resolution': (arg.input_height, arg.input_width),
            'attack_type': arg.attack_type,
            'attack_mode': arg.attack_mode,
            'trigger_size': float(p_size),
            'trigger_loc': loc,
            'trigger_alpha': float(alpha),
            'attack_target': arg.attack_target,
            'poisoning_ratio': float(arg.poisoning_ratio),
        }
        if arg.attack_type == 'SIG':
            info['trigger_delta'] = float(delta)
            info['trigger_frequency'] = float(frequency)
            if horizontal_or_vertical == 0:
                info['trigger_horizontal_or_vertical'] = 'horizontal'
            elif horizontal_or_vertical == 1:
                info['trigger_horizontal_or_vertical'] = 'vertical'
        json_info = json.dumps(info)
        with open(os.path.join(save_dir, 'info.json'), 'w') as f:
            f.write(json_info)
            
        if len(pattern.shape) == 2:
            im_pattern = (pattern + 1) / 2 * 255

        elif pattern.shape[0] == 3:
            im_pattern = (np.transpose(pattern, (1, 2, 0)) + 1) / 2 * 255
        im_pattern = Image.fromarray(im_pattern.astype(np.uint8))
        im_pattern.save(os.path.join(save_dir, 'trigger_pattern.jpg'))

if __name__ == "__main__":
    main()
