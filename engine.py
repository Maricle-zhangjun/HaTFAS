import math
import sys
from typing import Iterable, Optional
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import logging

import numpy as np
from sklearn.metrics import roc_curve

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            
            loss_base, loss_logit_dist, loss_fea_dist = criterion(samples, outputs, targets)
            loss = loss_base + loss_logit_dist + loss_fea_dist

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            # print("Loss is {}, stopping training".format(loss_value))
            logging.error("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_base=loss_base.item())
        metric_logger.update(loss_logit_dist=loss_logit_dist.item())
        metric_logger.update(loss_fea_dist=loss_fea_dist.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, require_feat=False)
            loss = criterion(output, target)

        acc1, acc2 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def eval_hter(data_loader, model, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, require_feat=False)
        
        y_p = []
        for i in output:
            if i[0] > i[1]:
                y_p.append(0)
            else:
                y_p.append(1)

        y_t = target.cpu().numpy()
        
        for i in range(len(y_t)):
            if y_p[i] == 0 and y_t[i] == 0:
                TN += 1
            elif y_p[i] == 0 and y_t[i] == 1:
                FN += 1
            elif y_p[i] == 1 and y_t[i] == 1:
                TP += 1
            elif y_p[i] == 1 and y_t[i] == 0:
                FP += 1

    FAR = FP/(FP+TN)
    FRR = FN/(FN+TP)
    HTER = (FAR + FRR)/2

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return FAR, FRR, HTER

@torch.no_grad()
def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == 0))
    FN = np.sum((labels == 1) & (predict == 0))
    FP = np.sum((labels == 0) & (predict == 1))
    TP = np.sum((labels == 1) & (predict == 1))
    return TN, FN, FP, TP

@torch.no_grad()
def get_EER_states(probs, labels, thresholds):
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if(FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
        elif(FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list

@torch.no_grad()
def get_EER(data_loader, model, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    val_pro = []
    val_labels = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        for i in target.tolist():
            val_labels = np.append(val_labels, i)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, require_feat=False)
            results_soft = torch.nn.functional.softmax(output,dim=1).cpu().detach().numpy()

        for i in results_soft.tolist():
            val_pro = np.append(val_pro,i[1])   

    fpr, tpr, thresholds = roc_curve(val_labels, val_pro, pos_label=1)
    EER,thr,FRR_list,FAR_list =  get_EER_states(val_pro,val_labels,thresholds)  

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return EER, thr