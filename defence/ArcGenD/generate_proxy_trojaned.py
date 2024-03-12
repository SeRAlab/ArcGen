import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.insert(0, '../..')
import os
from datetime import datetime
import json
import argparse
import config
from PIL import Image

from utils.dataloader import get_dataloader
from utils.train import unmodified_training, eval_model
from utils.get_model import get_model, get_datainfo

def set_attackinfo(arg):
    if arg.attack_mode == "alltoall":
        arg.attack_target = 'None'
    elif arg.attack_mode == "alltoone":
        if arg.attack_target in range(arg.num_classes) and arg.fixed_target:
            pass
        else:
            arg.attack_target = np.random.randint(0, arg.num_classes-1)

    arg.attack_type = "jumbo"

    arg.poisoning_ratio = np.random.uniform(0.05, 0.5)

def set_trigger_info(arg):
    MAX_SIZE = min(arg.input_height, arg.input_width)

    if arg.attack_type == "jumbo":
        if arg.dataset == "cifar10" or arg.dataset == "mnist" or arg.dataset == "gtsrb" or arg.dataset == "imagenet":
            p_size = np.random.choice([2,3,4,5,MAX_SIZE], 1)[0]
            if p_size < MAX_SIZE:
                alpha = np.random.uniform(0.2, 0.6)
                if alpha > 0.5:
                    alpha = 1.0
            else:
                alpha = np.random.uniform(0.05, 0.2)

    if p_size < MAX_SIZE:
        loc_x = np.random.randint(MAX_SIZE-p_size)
        loc_y = np.random.randint(MAX_SIZE-p_size)
        loc = (loc_x, loc_y)
    else:
        loc = (0, 0)

    if arg.dataset == "cifar10" or arg.dataset == "gtsrb" or arg.dataset == "imagenet":
        eps = np.random.uniform(0, 1)
        pattern = np.random.uniform(-eps, 1+eps,size=(3,p_size,p_size))
        pattern = np.clip(pattern,0,1)
    elif arg.dataset == "mnist":
        pattern_num = np.random.randint(1, p_size**2)
        one_idx = np.random.choice(list(range(p_size**2)), pattern_num, replace=False)
        pattern_flat = np.zeros((p_size**2))
        pattern_flat[one_idx] = 1
        pattern = np.reshape(pattern_flat, (p_size,p_size))

    return p_size, pattern, loc, alpha

def poisoning_func(X, y, trigger_info, arg, train=True):
    p_size, pattern, loc, alpha = trigger_info
    if arg.attack_mode == "alltoone":
        target_y = arg.attack_target
    elif arg.attack_mode == "alltoall":
        target_y = np.random.choice(list(set(range(arg.num_classes)) - set([y])), 1, replace=False)[0]

    w, h = loc
    X_new = X.clone()

    # mnist
    # X_new[0, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[0, w:w+p_size, h:h+p_size]

    X_new[:, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[:, w:w+p_size, h:h+p_size]
    y_new = target_y

    if not train and arg.attack_mode == "alltoall":
        y_new = y

    return X_new, y_new

def main():
    arg = config.get_arguments().parse_args()
    torch.cuda.set_device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(arg.device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    get_datainfo(arg)

    # Dataset
    test_dl_benign, test_transform = get_dataloader(arg, defender=True, train=False, poison=False)

    Model = get_model(arg)

    SAVE_PREFIX = os.path.join(arg.save_dir, arg.dataset)
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(os.path.join(SAVE_PREFIX, 'proxy_trojaned')):
        os.mkdir(os.path.join(SAVE_PREFIX, 'proxy_trojaned'))

    for i in range(arg.target_num):

        set_attackinfo(arg)
        trigger_info = set_trigger_info(arg)
        train_dl, train_transform = get_dataloader(arg, defender=True, train=True, poison=True, trigger_info=trigger_info ,pretensor_transform=True, poisoning_func=poisoning_func)
        test_dl_trojan, test_transform = get_dataloader(arg, defender=True, train=False, poison=True, trigger_info=trigger_info, poisoning_func=poisoning_func)
        print("train data size: ", len(train_dl.dataset))
        print("test benign data size: ", len(test_dl_benign.dataset))
        print("test trojan data size: ", len(test_dl_trojan.dataset))
        model = Model().to("cuda")
        print('using model: ', arg.model)
        print(i)

        unmodified_training(arg, model, train_dl, arg.epoch, arg.verbose)


        save_dir = os.path.join(SAVE_PREFIX, 'proxy_trojaned', f'{arg.model}_{i:04}')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        torch.save(model, os.path.join(save_dir, 'model.pth'))
        acc = eval_model(model, test_dl_benign)

        if arg.attack_mode == "alltoone":
            acc_mal = eval_model(model, test_dl_trojan)
        elif arg.attack_mode == "alltoall":
            acc_mal = 1 - eval_model(model, test_dl_trojan)
        print(f'benign acc: {acc:.4f}, trojan acc: {acc_mal:.4f}, save to {save_dir} @ {datetime.now()}')
        p_size, pattern, loc, alpha = trigger_info
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
