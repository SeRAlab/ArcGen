import numpy as np
import os
import json
import torch
import pandas as pd
import torch.utils.data

from defence.ArcGenD.model import ArcGenD, Discriminator
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import sys
sys.path.insert(0, '../..')
import config
from utils.dataloader import get_dataloader
from utils.get_model import get_datainfo
from defence.utils.network_dataset import NetworkDatasetDetection, SetNetworkDataset, custom_collate



def cosine_distance_loss(x, y):
    cosine_sim = F.cosine_similarity(x, y, dim=1)
    distance = 1 - cosine_sim
    return torch.mean(distance)

def epoch_detection_train(epoch, models, optimizer, train_loader, train_model_list, threshold=0.0, device='cuda', verbose=False, arg=None):
    detection_model, adv = models
    writer = SummaryWriter(arg.tensorboard_dir)
    
    epoch_loss = {}
    detection_model.train()
    adv.train()
    
    cum_loss = 0.0
    preds = []
    labs = []
    cum_acc = 0.0
    benign_acc = 0.0
    trojan_acc = 0.0
    cum_asr = 0.0
    num_benign = 0
    num_trojan = 0
    
    grad_max_norm = 1.0 
    
    for i, (net, label, data_info) in enumerate(train_loader['single']):
        out = []
        d_labs = []
        for j in range(len(net)):
            net[j].to(device).eval()
            ############mask#############
            if arg.mask > 0:
                weights = []
                masked_weights = []
                for param in net[j].parameters():
                    weights.append(param.data)
                    random_masktensor = torch.rand_like(param.data)  
                    threshold = arg.mask  
                    mask = (random_masktensor > threshold).float()  
                    masked_weights.append(torch.mul(param.data, mask))
                param_idx = 0
                for param in net[j].parameters():
                    param.data = masked_weights[param_idx]
                    param_idx += 1
            #############################
            out.append(net[j].forward(detection_model.inp).unsqueeze(0))
            d_label = next((index for index, model in enumerate(train_model_list) if model == data_info[j]['model']), None)
            if d_label is None:
                raise Exception("domain label error")
            d_labs.append(d_label)
        out = torch.cat(out, dim=0)
        score, feature = detection_model.forward(out, True)
        domain_discrimination = adv.forward(feature)
        
        t_loss = F.binary_cross_entropy_with_logits(score, torch.FloatTensor([label]).T.to(device))
        d_loss = F.cross_entropy(domain_discrimination, torch.LongTensor(d_labs).to(device))

        loss = t_loss + d_loss
        
        optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(detection_model.parameters(), grad_max_norm)
        torch.nn.utils.clip_grad_norm_(adv.parameters(), grad_max_norm)
        
        loss.backward()
        optimizer.step()
        

        cum_loss = cum_loss + loss.item()
        
        for j in range(len(score)):
            preds.append(score[j].item())
        
        for j in range(len(data_info)):
            cum_acc += data_info[j]['test_acc']
            if data_info[j].get('test_acc_mal', None) is not None:
                num_trojan += 1
                cum_asr += data_info[j]['test_acc_mal']
                trojan_acc += data_info[j]['test_acc']
            else:
                num_benign += 1
                benign_acc += data_info[j]['test_acc']

        for j in range(len(label)):
            labs.append(label[j])
        writer.add_scalar('t_loss', t_loss.item(), epoch * len(train_loader['single']) + i)
        writer.add_scalar('d_loss', d_loss.item(), epoch * len(train_loader['single']) + i)
        writer.add_scalar('all_loss', loss.item(), epoch * len(train_loader['single']) + i)
    
    if 'group' in train_loader:
        for i, (net, label, data_info) in enumerate(train_loader['group']):
            out = []
            same_trigger_loss = []
            for j in range(len(net)):
                if label[j] == 0:
                    continue
                for k in range(len(net[j])):
                    net[j][k].to(device).eval()
                    if arg.mask > 0:
                        weights = []
                        masked_weights = []
                        for param in net[j][k].parameters():
                            weights.append(param.data)
                            random_masktensor = torch.rand_like(param.data)  
                            threshold = arg.mask  
                            mask = (random_masktensor > threshold).float()  
                            masked_weights.append(torch.mul(param.data, mask))
                        param_idx = 0
                        for param in net[j][k].parameters():
                            param.data = masked_weights[param_idx]
                            param_idx += 1

                    out.append(net[j][k].forward(detection_model.inp).unsqueeze(0))

                for x in range(len(torch.cat(out[-len(net[j]):], dim=0))):
                    for y in range(x+1, len(torch.cat(out[-len(net[j]):], dim=0))):
                        distance = cosine_distance_loss(torch.cat(out[-len(net[j]):], dim=0)[x].unsqueeze(0), torch.cat(out[-len(net[j]):], dim=0)[y].unsqueeze(0))
                        same_trigger_loss.append(distance)
                
            loss = None
            for st_loss in same_trigger_loss:
                if loss == None:
                    loss = st_loss / (len(same_trigger_loss))
                loss += st_loss / (len(same_trigger_loss))
            
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(detection_model.parameters(), grad_max_norm)
            loss.backward()
            optimizer.step()
    writer.add_images('query', detection_model.inp, epoch)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.median(preds).item()
    acc = ((np.array(preds) > threshold) == labs).mean()
    writer.close()
    return cum_loss / len(preds), auc, acc


def epoch_test_eval(detection_model, loader, threshold=0.0, device='cuda', verbose=False, doc=None):
    pbar = tqdm(loader, disable=not verbose)
    pbar.set_description(f"Eval")
    detection_model.eval()

    cum_loss = 0.0
    preds = []
    labs = []
    data_infos = []
    data_model = []

    for i, (net, label, data_info) in enumerate(pbar):
        out = []
        for j in range(len(net)):
            net[j].to(device).eval()
            with torch.no_grad():
                out.append(net[j].forward(detection_model.inp).unsqueeze(0))
        with torch.no_grad():
            out = torch.cat(out, dim=0)
            score, _ = detection_model.forward(out, True)

        loss = F.binary_cross_entropy_with_logits(score, torch.FloatTensor([label]).T.to(device))
        
        cum_loss = cum_loss + loss.item()
        for j in range(len(score)):
            preds.append(score[j].item())
        for j in range(len(label)):
            labs.append(label[j])
        for j in range(len(data_info)):
            data_infos.append(data_info[j])
            data_model.append(data_info[j]['model'])
    
    preds = np.array(preds)
    labs = np.array(labs)
    np.save('preds_o.npy', preds)
    np.save('labs_o.npy', labs)
    try:
        auc = roc_auc_score(labs, preds)
    except:
        auc = 0.0
    if threshold == 'half':
        threshold = np.median(preds).item()
    acc = ((np.array(preds) > threshold) == labs).mean()
    
    model_auc_dict = {}
    for model in set(data_model):
        model_auc_dict[model] = roc_auc_score(np.array(labs)[np.array(data_model) == model], np.array(preds)[np.array(data_model) == model])
        
    
    if doc is not None and os.path.exists(os.path.dirname(doc)):
        df = pd.DataFrame(data_infos)
        df['preds'] = preds
        df['labs'] = labs
        df['TF'] = ((df['preds'] > threshold) == df['labs'])
        df.to_csv(doc, index=False)
        print('save to {}'.format(doc))
    elif doc is not None:
        print('doc %s not exists' % doc)
    
    return cum_loss / len(preds), auc, acc, model_auc_dict


def get_dataset(dataset_path, task, train_model_list, arg, device):
    model_list = [f'simpleCNN_{task}', f'simpleCNN_{task}_badnets', f'simpleCNN_{task}', f'simpleCNN_{task}_badnets', 'senet18','senet18_badnets','resnet18','resnet18_badnets', 'mobilnetv2','mobilnetv2_badnets','efficientnetb0','efficientnetb0_badnets','shufflenetv2','shufflenetv2_badnets']
    
    # tarin
    train_dataset = {}
    train_dataset['single'] = NetworkDatasetDetection(os.path.join(dataset_path, task), clean_sublist=['proxy_benign'], trojan_sublist=['proxy_trojaned'], model_list=train_model_list, device=device)
    train_dataset['group'] = SetNetworkDataset(os.path.join(dataset_path, task), clean_sublist=['proxy_benign'], trojan_sublist=['proxy_trojaned'], model_list=train_model_list, device=device)

    # test    
    test_dataset = NetworkDatasetDetection(os.path.join(dataset_path, task), clean_sublist=['target_benign'], trojan_sublist=['trojaned'], model_list=model_list, device=device)


    split_s = int(len(train_dataset['single']) * 0.8)
    split_g = int(len(train_dataset['group']) * 0.8)
    val_dataset = {}
    train_dataset['single'], val_dataset['single'] = torch.utils.data.random_split(train_dataset['single'], [split_s, len(train_dataset['single']) - split_s])
    train_dataset['group'], val_dataset['group'] = torch.utils.data.random_split(train_dataset['group'], [split_g, len(train_dataset['group']) - split_g])
    print(f'Train: {len(train_dataset["single"])}, Val: {len(val_dataset["single"])}, Test: {len(test_dataset)}')
    print(f'Train: {len(train_dataset["group"])}, Val: {len(val_dataset["group"])}, Test: {len(test_dataset)}')
    
    return train_dataset, val_dataset, test_dataset


def main():
    N_REPEAT = 5
    PRO_NAME = 'ArcGenD'
    TRAIN_NUM = 25
    VAL_NUM = 25
    TEST_NUM = 25

    arg = config.get_arguments().parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device(arg.device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    get_datainfo(arg)

    # Dataset
    dataset_path = arg.save_dir
    task = arg.dataset

    train_model_list = ['mobilnetv2', 'senet18']


    train_dataset, val_dataset, test_dataset = get_dataset(dataset_path, task, train_model_list, arg, device)
    train_loader = {}
    val_loader = {}
    if len(train_dataset['group']) != 0:
        train_loader['single'] = torch.utils.data.DataLoader(train_dataset['single'], batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, pin_memory=False, collate_fn=custom_collate)
        train_loader['group'] = torch.utils.data.DataLoader(train_dataset['group'], batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, pin_memory=False, collate_fn=custom_collate)
        val_loader['single'] = torch.utils.data.DataLoader(val_dataset['single'], batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=False, collate_fn=custom_collate)
        val_loader['group'] = torch.utils.data.DataLoader(val_dataset['group'], batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=False, collate_fn=custom_collate)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=False, collate_fn=custom_collate)
    else:
        train_loader['single'] = torch.utils.data.DataLoader(train_dataset['single'], batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, pin_memory=False, collate_fn=custom_collate)
        val_loader['single'] = torch.utils.data.DataLoader(val_dataset['single'], batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=False, collate_fn=custom_collate)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=False, collate_fn=custom_collate)
        
    AUCs = []
    Models_AUCs = []
    add_epoch = 0
    for i in range(N_REPEAT): # Result contains randomness, so run several times and take the average
        arg.tensorboard_dir = os.path.join(arg.tensorboard_dir, f'{PRO_NAME}_repeat{i}')
        detection_model = ArcGenD(input_size=(arg.input_channel, arg.input_height, arg.input_width), class_num=arg.num_classes, num_query=arg.query_num)
        ad_net = Discriminator(n_feature=(arg.num_classes * arg.query_num),n_domain=len(train_model_list))
        
        detection_model.to(device)
        ad_net.to(device)
        optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, detection_model.parameters()), "lr": 1e-2},
        {"params": filter(lambda p: p.requires_grad, ad_net.parameters()), "lr": 1e-2},
        ]   
        optimizer = torch.optim.Adam(optimizer_dict, amsgrad=True)
        
        models = (detection_model, ad_net)

        best_eval_auc = None
        test_info = None
        for epoch in range(arg.epoch + add_epoch):
            if epoch == 4/6 * arg.epoch:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)
            epoch_detection_train(epoch, models, optimizer, train_loader, train_model_list, threshold='half', device=device, verbose=arg.verbose, arg=arg)
            if epoch >= 4/6 * arg.epoch:
                scheduler.step()
            eval_loss, eval_auc, eval_acc, model_auc_dict = epoch_test_eval(detection_model, val_loader['single'], threshold='half', device=device)
            if  epoch > arg.epoch * 4/6 and (best_eval_auc is None or eval_auc > best_eval_auc):
                best_eval_auc = eval_auc
                test_info = epoch_test_eval(detection_model, test_loader, threshold='half', device=device, doc=os.path.join(arg.document, f'{PRO_NAME}_{arg.dataset}_epoch_{arg.epoch}_repeat_{i}_result.csv'))
                torch.save(detection_model.state_dict(), os.path.join(arg.save_detection_dir, f'{PRO_NAME}_{arg.dataset}_epoch_{arg.epoch}_repeat_{i}_model.pth'))
                print ("epoch %d, eval_loss: %.4f, eval_auc: %.4f, eval_acc: %.4f, best_eval_auc: %.4f"%(epoch, eval_loss, eval_auc, eval_acc, best_eval_auc))
                print ("\tTest AUC:", test_info[1], "Test Acc:", test_info[2])
                for model_auc in test_info[3].keys():
                    print ("\tTest AUC of %s:"%(model_auc), test_info[3][model_auc])
            else :
                print ("epoch %d, eval_loss: %.4f, eval_auc: %.4f, eval_acc: %.4f"%(epoch, eval_loss, eval_auc, eval_acc))

        triple_planet()
        print ("\tTest AUC:", test_info[1], "Test Acc:", test_info[2])
        for model_auc in test_info[3].keys():
            print ("\tTest AUC of %s:"%(model_auc), test_info[3][model_auc])
        triple_planet()
        AUCs.append(test_info[1])
        Models_AUCs.append(test_info[3])


    AUC_mean = sum(AUCs) / len(AUCs)
    print ("Average detection AUC on %d detection model: %.4f"%(N_REPEAT, AUC_mean))
    for model_auc in Models_AUCs[0].keys():
        AUCs = [model_auc_dict[model_auc] for model_auc_dict in Models_AUCs]
        AUC_mean = sum(AUCs) / len(AUCs)
        print ("Average detection AUC of %s on %d detection model: %.4f"%(model_auc, N_REPEAT, AUC_mean))

def triple_planet():
    print("*********************************************")
    print("*********************************************")
    print("*********************************************")

if __name__ == "__main__":
    main()



