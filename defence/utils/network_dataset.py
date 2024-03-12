import torch
import os
import json

import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class NetworkDatasetDetection(torch.utils.data.Dataset):
    def __init__(self, model_folder, clean_sublist=None, trojan_sublist = None, model_list = None, selected_model_func=lambda x: x, device='cuda'):
        self.device = device
        if clean_sublist is None:
            clean_sublist = ['clean']
        if trojan_sublist is None:
            trojan_sublist = ['trojan']
        super().__init__()

        model_paths = []
        for clean_sub in clean_sublist:
            model_paths.extend([os.path.join(model_folder, clean_sub, x) \
                                for x in sorted(os.listdir(os.path.join(model_folder, clean_sub)))])
        for trojan_sub in trojan_sublist:
            model_paths.extend([os.path.join(model_folder, trojan_sub, x) \
                                for x in sorted(os.listdir(os.path.join(model_folder, trojan_sub)))])
        selected_model_paths = []
        for p in model_paths:
            if model_list is not None and type(model_list) is list and p.split('/')[-1].rsplit('_', 1)[0] not in model_list:
                continue
            selected_model_paths.append(p)
        selected_model_paths = selected_model_func(selected_model_paths)

        labels = []
        data_info = []
        for p in selected_model_paths:
            with open(os.path.join(p, 'info.json'), 'r') as f:
                info = json.load(f)
                data_info.append(info)
            if p.split('/')[-2] in clean_sublist:
                labels.append(0)
            elif p.split('/')[-2] in trojan_sublist:
                labels.append(1)
            else:
                raise ValueError('unexpected path {}'.format(p))
        self.model_paths = selected_model_paths
        self.labels = labels
        self.data_info = data_info
        print(len(self.model_paths), len(self.labels), len(self.data_info))

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, index):
        if os.path.exists(os.path.join(self.model_paths[index], 'model.pt')):
            return torch.load(os.path.join(self.model_paths[index], 'model.pt'), map_location=self.device), \
                self.labels[index], self.data_info[index]
        else:
            return torch.load(os.path.join(self.model_paths[index], 'model.pth'), map_location=self.device), \
                self.labels[index], self.data_info[index]

def custom_collate(batch):
    return [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]

class CleanNetworkDataset(torch.utils.data.Dataset):
    def __init__(self, model_folder, clean_sublist=None, model='all'):
        if clean_sublist is None:
            clean_sublist = ['clean']
        super().__init__()
        model_paths = []
        for clean_sub in clean_sublist:
            model_paths.extend([os.path.join(model_folder, clean_sub, x) \
                                for x in sorted(os.listdir(os.path.join(model_folder, clean_sub)))])
        data_info = []
        for p in model_paths:
            if model != 'all' and p.split('/')[-1].rsplit('_', 1)[0] != model:
                print(p.split('/')[-1].rsplit('_', 1)[0])
                print(model)
                continue
            with open(os.path.join(p, 'info.json'), 'r') as f:
                info = json.load(f)
                data_info.append(info)
        self.model_paths = model_paths
        self.data_info = data_info

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, index):
        return torch.load(os.path.join(self.model_paths[index], 'model.pth')), \
            self.data_info[index]
def custom_collate_clean(batch):
    return [x[0] for x in batch], [x[1] for x in batch]

class TrojanNetworkDataset(torch.utils.data.Dataset):
    def __init__(self, model_folder, trojan_sublist = None, model='all'):
        if trojan_sublist is None:
            trojan_sublist = ['trojan']
        super().__init__()
        model_paths = []
        for trojan_sub in trojan_sublist:
            model_paths.extend([os.path.join(model_folder, trojan_sub, x) \
                                for x in sorted(os.listdir(os.path.join(model_folder, trojan_sub)))])
        data_info = []
        for p in model_paths:
            if model != 'all' and p.split('/')[-1].rsplit('_', 1)[0] != model:
                print(p.split('/')[-1].rsplit('_', 1)[0])
                print(model)
                continue
            with open(os.path.join(p, 'info.json'), 'r') as f:
                info = json.load(f)
                data_info.append(info)
        self.model_paths = model_paths
        self.data_info = data_info

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, index):
        return torch.load(os.path.join(self.model_paths[index], 'model.pth')), \
            self.data_info[index]
def custom_collate_trojan(batch):
    return [x[0] for x in batch], [x[1] for x in batch]

class SetNetworkDataset(torch.utils.data.Dataset):
    def __init__(self, model_folder, clean_sublist=None, trojan_sublist = None, model_list = None, selected_model_func=lambda x: x, device='cuda'):
        self.device = device
        if clean_sublist is None:
            clean_sublist = ['clean']
        if trojan_sublist is None:
            trojan_sublist = ['trojan']
        super().__init__()

        model_paths = []
        for clean_sub in clean_sublist:
            model_paths.extend([os.path.join(model_folder, clean_sub, x) \
                                for x in sorted(os.listdir(os.path.join(model_folder, clean_sub)))])
        for trojan_sub in trojan_sublist:
            model_paths.extend([os.path.join(model_folder, trojan_sub, x) \
                                for x in sorted(os.listdir(os.path.join(model_folder, trojan_sub)))])
        selected_model_paths = []
        for p in model_paths:
            
            if model_list is not None and type(model_list) is list:
                if p.split('/')[-1].rsplit('_', 1)[0] not in model_list:
                    continue
                if all(os.path.exists(file_path) for file_path in [p.replace(p.split('/')[-1].rsplit('_', 1)[0], model_type) for model_type in model_list]):
                    if p.replace(p.split('/')[-1].rsplit('_', 1)[0], 'modeltype') not in selected_model_paths:
                        selected_model_paths.append(p.replace(p.split('/')[-1].rsplit('_', 1)[0], 'modeltype'))
                                    
        selected_model_paths = selected_model_func(selected_model_paths)

        set_model_paths = []
        labels = []
        data_info = []
        for p in selected_model_paths:
            model_set = []
            label_set = []
            data_info_set = []
            for file_path in [p.replace(p.split('/')[-1].rsplit('_', 1)[0], model_type) for model_type in model_list]:
                model_set.append(file_path)
                with open(os.path.join(file_path, 'info.json'), 'r') as f:
                    info = json.load(f)
                    data_info_set.append(info)
                if file_path.split('/')[-2] in clean_sublist:
                    label_set.append(0)
                elif file_path.split('/')[-2] in trojan_sublist:
                    label_set.append(1)
                else:
                    raise ValueError('unexpected path {}'.format(file_path))
            set_model_paths.append(model_set)
            labels.append(label_set)
            data_info.append(data_info_set)
        self.model_paths = set_model_paths
        self.labels = labels
        self.data_info = data_info
        print(len(self.model_paths), len(self.labels), len(self.data_info))

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, index):
        return [torch.load(os.path.join(path, 'model.pth'), map_location=self.device) for path in self.model_paths[index]], \
            self.labels[index], self.data_info[index]