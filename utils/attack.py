import torch
import numpy as np


def get_attackinfo(arg):
    if arg.attack_mode == "alltoall":
        arg.attack_target = 'None'
    elif arg.attack_mode == "alltoone":
        if arg.attack_target in range(arg.num_classes) and arg.fixed_target:
            pass
        else:
            arg.attack_target = np.random.randint(0, arg.num_classes-1)

    if arg.attack_type == "badnets":
        pass

    arg.poisoning_ratio = np.random.uniform(0.05, 0.5)

def get_trigger_info(arg):
    MAX_SIZE = min(arg.input_height, arg.input_width)

    if arg.attack_type == "badnets":
        if arg.dataset == "cifar10" or arg.dataset == "mnist" or arg.dataset == "gtsrb":
            MAX_SIZE = min(arg.input_height, arg.input_width)
            p_size = np.random.choice([2,3,4,5], 1)[0]
            alpha = 1.0
        elif arg.dataset == "imagenet":
            MAX_SIZE = min(arg.input_height, arg.input_width)
            p_size = np.random.choice([2,3,4,5], 1)[0]
            alpha = 1.0

    elif arg.attack_type == "blended":
        if arg.dataset == "cifar10" or arg.dataset == "mnist" or arg.dataset == "gtsrb":
            MAX_SIZE = min(arg.input_height, arg.input_width)
            p_size = MAX_SIZE
            alpha = np.random.uniform(0.05, 0.2)
        elif arg.dataset == "imagenet":
            MAX_SIZE = min(arg.input_height, arg.input_width)
            p_size = MAX_SIZE
            alpha = np.random.uniform(0.05, 0.2)

    elif arg.attack_type == "SIG":
        if arg.dataset == "cifar10" or arg.dataset == "mnist" or arg.dataset == "gtsrb":
            MAX_SIZE = min(arg.input_height, arg.input_width)
            p_size = MAX_SIZE
            alpha = np.random.uniform(0.05, 0.2)
        elif arg.dataset == "imagenet":
            MAX_SIZE = min(arg.input_height, arg.input_width)
            p_size = MAX_SIZE
            alpha = np.random.uniform(0.05, 0.2)

    if p_size < MAX_SIZE:
        loc_x = np.random.randint(MAX_SIZE-p_size)
        loc_y = np.random.randint(MAX_SIZE-p_size)
        loc = (loc_x, loc_y)
    else:
        loc = (0, 0)

    if arg.attack_type == "SIG":
        delta = np.random.choice([10,20,30,40,50,60,70,80], 1)[0]
        frequency = np.random.randint(1, p_size/2)
        horizontal_or_vertical = np.random.randint(0, 2)
        # delta = 10
        # frequency=4
        if arg.dataset == "cifar10" or arg.dataset == "gtsrb" or arg.dataset == "imagenet":
            pattern = np.zeros((3,p_size,p_size))
            m = pattern.shape[1]
            for i in range(pattern.shape[0]):
                if horizontal_or_vertical == 0:
                    for j in range(pattern.shape[1]):
                          pattern[i, j] = delta * np.sin(2 * np.pi * j * frequency / m)
                else:
                    for k in range(pattern.shape[2]):
                        pattern[i, :, k] = delta * np.sin(2 * np.pi * k * frequency / m)
        elif arg.dataset == "mnist":
            pattern = np.zeros((p_size,p_size))

    else:
        delta = 0
        frequency = 0
        horizontal_or_vertical = -1
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

    return p_size, pattern, loc, alpha, delta, frequency, horizontal_or_vertical

def poisoning_func(X, y, trigger_info, arg, train=True):
    p_size, pattern, loc, alpha, _, _, _ = trigger_info
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
