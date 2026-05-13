# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from sklearn.metrics import f1_score, balanced_accuracy_score, multilabel_confusion_matrix, precision_score, recall_score
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image

def masked_ce_loss(logits_list, targets_list):
    """
    added for unknown labels in combined mul-lbl mul-cls training
    logits_list: list of tensors [batch, n_classes] for each task
    targets_list: list of tensors [batch] for each task, may contain -1
    """
    total_loss = 0.0
    n_tasks = len(logits_list)
    
    for logits, target in zip(logits_list, targets_list):
        mask = target >= 0                  # valid labels
        if mask.any():                      # compute loss only if valid labels exist
            total_loss += F.cross_entropy(logits[mask], target[mask])
    return total_loss / n_tasks

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    # debug
    count = 0
    for indices, samples, targets in metric_logger.log_every(data_loader, print_freq, header): #for handling IndexedDataset who return indices as well
        count += 1

        B = targets.size(0)
        N = len(samples)  # 4 crops
        logits_list = []
        targets = targets.to(device, non_blocking=True)
        for c in range(N):
            crop_batch = samples[c].to(device, non_blocking=True)  # (B, C, H, W)
   
            with amp_autocast():
                if args.model in ['vit_small','vit_base','resnet50','swin_s','swin_b']:
                    outputs = model(crop_batch)
                else:
                    outputs = model(
                        crop_batch,
                        if_random_cls_token_position=args.if_random_cls_token_position,
                        if_random_token_rank=args.if_random_token_rank
                    ) 
            logits_list.append(outputs)  #收集4个crop的prediction

        pos_idx = 1   # 你正类是第 1 类的情况下这样写

        pos_probs = [F.softmax(l, dim=1)[:, pos_idx] for l in logits_list]  # list of (B,)
        pos_stack = torch.stack(pos_probs, dim=0)    # (N, B)
        final_pos_prob = pos_stack.max(dim=0)[0]     # (B,)
        final_pos_prob = final_pos_prob.clamp(1e-6, 1-1e-6)
        final_pos_logit = torch.log(final_pos_prob / (1-final_pos_prob))  # (B,)

        loss = F.binary_cross_entropy_with_logits(final_pos_logit, targets.float().to(final_pos_logit.device))

        if args.if_nan2num: # default=false
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, args):
    if args.nb_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Test:'

    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0

    for batch_idx, samples, targets in metric_logger.log_every(data_loader, 10, header):
        B = targets.size(0)
        N = len(samples)
        logits_list = []
        targets = targets.to(device, non_blocking=True)

        for c in range(N):
            crop_batch = samples[c].to(device, non_blocking=True)
            with amp_autocast():
                logits = model(crop_batch)
                logits_list.append(logits)


        # ---- Voting ----
        if args.nb_classes == 2:
            pos_idx = 1
            pos_probs = [F.softmax(l, dim=1)[:, pos_idx] for l in logits_list]
            pos_stack = torch.stack(pos_probs, dim=0)
            final_pos_prob = pos_stack.max(dim=0)[0]
            final_pos_logit = torch.log(final_pos_prob / (1 - final_pos_prob))
            final_preds = (final_pos_prob > 0.5).long()
            loss = F.binary_cross_entropy_with_logits(final_pos_logit, targets.float())
        else:
            logits_stack = torch.stack(logits_list, dim=0)
            final_logits, _ = logits_stack.max(dim=0)
            final_preds = final_logits.argmax(dim=1)
            loss = criterion(final_logits, targets)


        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        acc1 = (final_preds == targets).float().sum() / batch_size * 100
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)

        y_true.append(targets.cpu().numpy())
        y_pred.append(final_preds.cpu().numpy())


    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if args.nb_classes == 2:
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1) * 100
    else:
        f1 = f1_score(y_true, y_pred, average='macro') * 100

    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100

    stats = {
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'loss': total_loss / len(data_loader.dataset),
        'acc1': metric_logger.acc1.global_avg  # 保留 acc1 key
    }

    return stats
