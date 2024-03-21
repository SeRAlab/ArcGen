import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


def unmodified_training(arg, model, dataloader, epoch_num, verbose=True):
    model.train()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), arg.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], 0.1)
    criterion_CE = torch.nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        pbar = tqdm(dataloader, disable=not verbose)
        pbar.set_description(f"Epoch {epoch + 1}")

        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i,(x_in, y_in) in enumerate(pbar):
            x_in, y_in = x_in.to("cuda"), y_in.to("cuda")
            B = x_in.size()[0]
            pred = model(x_in)
            loss_ce = criterion_CE(pred, y_in)
            loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                cum_loss += loss.item() * B

                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in.cpu())).sum().item()
                tot = tot + B

                pbar.set_postfix(loss=cum_loss/tot, acc=cum_acc/tot)
            cum_loss += loss.item() * B
            tot = tot + B
        scheduler.step()
        if verbose:
            print ("Epoch %d, loss = %.4f, acc = %.4f"%(epoch + 1, cum_loss/tot, cum_acc/tot))
    return

def eval_model(model, dataloader):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i,(x_in, y_in) in enumerate(dataloader):
        x_in, y_in = x_in.to("cuda"), y_in.to("cuda")
        B = x_in.size()[0]
        pred = model(x_in)

        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y_in.cpu())).sum().item()
        tot = tot + B
    return cum_acc / tot
