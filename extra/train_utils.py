import numpy as np
from sklearn import *
from numpy import *
from .nn_model import GRUTagging
import torch.optim as optim
import torch.nn as nn
import torch

from sklearn.model_selection import ParameterGrid

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)



class AUROC(object):
    def __init__(self, unique_tag_num=22):
        self.unique_tag_num = unique_tag_num
    
    def __call__(self, Yscores, Yclasses):
        fprall = []
        tprall = []
        aucall = []
        for i in range(self.unique_tag_num):
            fpr, tpr, thresholds = metrics.roc_curve(Yclasses[:,i], Yscores[:,i])
#             plt.plot(fpr, tpr, lw=0.5, alpha=0.5)
            auc = metrics.auc(fpr, tpr)
            fprall.append(fpr)
            tprall.append(tpr)
            aucall.append(auc)

        # Then interpolate all ROC curves at this points
        all_fpr = unique(concatenate(fprall))
        mean_tpr = zeros_like(all_fpr)
        for i in range(self.unique_tag_num):
            mean_tpr += interp(all_fpr, fprall[i], tprall[i])

        # Finally average it and compute AUC
        mean_tpr /= self.unique_tag_num

        # auc of the average ROC curve
        auc = metrics.auc(all_fpr, mean_tpr)

        # average AUC
        mc_auc = mean(aucall)
    
        return mc_auc, auc
    
def train(model, dataloader, optimizer, criterion):
    epoch_loss = 0
    
    model.train()
    for dt, gt in dataloader:
        dt = dt.float().cuda()
        gt = gt.float().cuda()
        optimizer.zero_grad()
        predictions = model(dt)
        loss = criterion(predictions, gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    
    preds = []
    gts = []
    with torch.no_grad():
        for dt, gt in dataloader:
            dt = dt.float().cuda()
            gt = gt.float().cuda()
            predictions = model(dt)
            preds.append(predictions)
            gts.append(gt)
            
    preds = torch.cat(preds, dim=0).cpu().numpy()
    gts = torch.cat(gts, dim=0).cpu().numpy()
    score = criterion(preds, gts)

    return preds, score

def grid_search(model_name, param_grid, trainloader, validloader, n_epochs=50, num_classes=22):
    param_combs = list(ParameterGrid(param_grid))
    global_best_score = 0.
    best_save_path = None
    for comb in param_combs:
        model_kwargs = {}
        loss_kwargs = {}
        optim_kwargs = {}
        for key, val in comb.items():
            component, arg = key.split('.')
            if component == 'loss':
                loss_kwargs[arg] = val
            elif component == 'model':
                model_kwargs[arg] = val
            elif component == 'optim':
                optim_kwargs[arg] = val
            else:
                raise ValueError(f'Unknow component {component}.')
        
        if model_name == 'gru':
            model = GRUTagging(batch_first=True, **model_kwargs).cuda()
        else:
            raise NotImplementedError(f'Model {model_name} is not implemented.')
        
        optimizer = optim.Adam(model.parameters(), **optim_kwargs)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=n_epochs//3,
                                              gamma=0.5)
        criterion = nn.BCEWithLogitsLoss(**loss_kwargs).cuda()
        val_criterion = AUROC(num_classes)
        
        epoch_best_valid_score = 0.
        
        logging.info(f'training model {model_name}.'\
                     f'\n\t -model_kwargs: {model_kwargs}'\
                     f'\n\t -loss_kwargs: {loss_kwargs}'\
                     f'\n\t -optim_kwargs: {optim_kwargs}')
        
        for epoch in range(1, n_epochs+1):
            scheduler.step()
            
            train_loss = train(model, trainloader, optimizer, criterion)
            preds, (mc_auc, auc) = evaluate(model, validloader, val_criterion)
            
            if mc_auc > epoch_best_valid_score:
                epoch_best_valid_score = mc_auc
                save_dict = {'model_name': model_name,
                             'model_kwargs': model_kwargs,
                             'loss_kwargs': loss_kwargs,
                             'optim_kwargs': optim_kwargs,
                             'model_state_dict': model.state_dict()}
                save_path = f'ckpt/{model_name}_mcauc_{mc_auc:.3f}.pt'
                torch.save(save_dict, save_path)
                if mc_auc > global_best_score:
                    global_best_score = mc_auc
                    best_save_path = save_path
                
            logging.info(f'Epoch: {epoch:02}/{n_epochs} | '\
                         f'Train Loss: {train_loss:.3f} | Valid. MCAUC: {mc_auc:.3f}')
    return best_save_path, global_best_score


def load_model(ckpt_path):
    save_dict = torch.load(ckpt_path)
    if save_dict['model_name'] == 'gru':
        model = GRUTagging(batch_first=True, **save_dict['model_kwargs']).cuda()
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented.')
    model.load_state_dict(save_dict['model_state_dict'])
    return model



