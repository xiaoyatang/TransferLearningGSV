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
        print('cosub is true.')  # 20251210  by xiaoya, for failure case 1
        # criterion = torch.nn.BCEWithLogitsLoss() 
        criterion = torch.nn.CrossEntropyLoss() # 20251210  by xiaoya, for failure case 1
        
    # debug
    # count = 0
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _, samples, targets in metric_logger.log_every(data_loader, print_freq, header): #for handling IndexedDataset who return indices as well
        # count += 1
        # if count > 20:
        #     break
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # print('target',targets.shape, targets)
        if mixup_fn is not None:
            # print('get into mixup', mixup_fn)
            samples, targets = mixup_fn(samples, targets)
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            targets = torch.cat((targets, targets), dim=0)   # 20251210  by xiaoya, for failure case 1
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
        #     outputs = model(samples) if args.model == 'vit_small' else model(
        #         samples,
        #         if_random_cls_token_position=args.if_random_cls_token_position,
        #         if_random_token_rank=args.if_random_token_rank
        #     )
            if args.model in ['vit_small','vit_base','resnet50','swin_s','swin_b']:
                outputs = model(samples)
            else:
                outputs = model(
                    samples,
                    if_random_cls_token_position=args.if_random_cls_token_position,
                    if_random_token_rank=args.if_random_token_rank
                )  # the two arguments are for 'vim', tuning the token positions
            # outputs = model(samples)

            # if not args.cosub:
            if args.cosub:
                if args.data_set == 'COMBINED':
                    # print('✅:Enter loss calculation for combined dataset.')
                    # added for multi-label multi-class training
                    # singleLane,crosswalk,green30,NSH,streetlight. Binary tasks 1,2,3,5 3-cls:4-th
                    logits_bin1 = outputs[:, 0:2]   # Binary task
                    logits_bin2 = outputs[:, 2:4]   # Binary task
                    logits_bin3 = outputs[:, 4:6]   # Binary task
                    logits_3cls = outputs[:, 6:9]   # 3-class task
                    logits_bin4 = outputs[:, 9:11]  # Binary task

                    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    # target_bin1 = targets[:, 0].clone()
                    # target_bin1[target_bin1 < 0] = -100
                    target_bin1 = targets[:, 0]
                    target_bin2 = targets[:, 1]
                    target_bin3 = targets[:, 2]
                    target_3cls = targets[:, 3]
                    target_bin4 = targets[:, 4]
                    # print('sliced target for 3-class label',target_3cls)

                    logits_list = [logits_bin1, logits_bin2, logits_bin3, logits_3cls, logits_bin4]
                    targets_list = [target_bin1, target_bin2, target_bin3, target_3cls, target_bin4]
                    loss = masked_ce_loss(logits_list, targets_list)
                    # print('total loss:',loss)
                    
                else:
                    loss = criterion(outputs, targets)
                    # loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        if args.if_nan2num:
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


# @torch.no_grad()
# def evaluate(data_loader, model, device, amp_autocast, args):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     if args.data_set == 'COMBINED':
#         all_preds = [[] for _ in range(5)]   # one list per task
#         all_targets = [[] for _ in range(5)]
#     else:
#         y_true = []
#         y_pred = []

#     # for visualization only
#     # visualized_count = 0 
#     # dataset = data_loader.dataset # wrap with image path
#     for batch_idx, images,target in enumerate(metric_logger.log_every(data_loader, 10, header)):
#     # for images, target in metric_logger.log_every(data_loader, 10, header):
#         # if len(batch) == 3:
#         #     _, images, target = batch  # IndexedDataset: index, image, label
#         # else:
#         #     images, target = batch

#         # images, target = batch  # wrap with image path
#         # _, images, target = batch  # wrap with image path

#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         with amp_autocast():
#             if args.attn_map:      # for visualization only
#                 # pos_indices = (target == 1).nonzero(as_tuple=True)[0]
#                 # if len(pos_indices) > 0 and visualized_count < 5: 
#                 #     attentions = model.module.get_last_selfattention(images) 
#                 #     for pos_idx in pos_indices:
#                 #         dataset_index = batch_idx * data_loader.batch_size + pos_idx.item()
#                 #         img_path, _ = dataset.samples[dataset_index]
#                 #         if visualized_count >= 3:
#                 #             import sys
#                 #             sys.exit(0)     
                        
#                 #         utils.visualize_attn(images[pos_idx:pos_idx+1],attentions[pos_idx:pos_idx+1],args)
#                 #         pred = model(images[pos_idx:pos_idx+1])
#                 #         print(
#                 #             'prediction', pred,
#                 #             'ground truth', target[pos_idx].item(),
#                 #             'path', img_path
#                 #         )
#                 #         visualized_count += 1
#                 continue
            
#             else:
#                 output = model(images)
#             if args.data_set == 'COMBINED':
#                 # --- Multi-label, multi-class evaluation ---
#                 logits_bin1 = output[:, 0:2]   # Binary
#                 logits_bin2 = output[:, 2:4]   # Binary
#                 logits_bin3 = output[:, 4:6]   # Binary
#                 logits_3cls = output[:, 6:9]   # 3-class
#                 logits_bin4 = output[:, 9:11]  # Binary

#                 # Slice targets
#                 target_bin1 = target[:, 0]
#                 target_bin2 = target[:, 1]
#                 target_bin3 = target[:, 2]
#                 target_3cls = target[:, 3]
#                 target_bin4 = target[:, 4]

#                 logits_list = [logits_bin1, logits_bin2, logits_bin3, logits_3cls, logits_bin4]
#                 targets_list = [target_bin1, target_bin2, target_bin3, target_3cls, target_bin4]

#                 # masked_ce_loss already ignores -1 targets
#                 loss = masked_ce_loss(logits_list, targets_list)

#                 # Predictions
#                 preds_list = [logits.argmax(dim=1) for logits in logits_list]
#                 # print('preds_list',preds_list)

#                 for i in range(len(preds_list)):                # accumulate per-task
#                     all_preds[i].append(preds_list[i].cpu().numpy())
#                     all_targets[i].append(targets_list[i].cpu().numpy())

#             else:
#                 loss = criterion(output, target)
#                 preds = output.argmax(dim=1)
#                 y_true.append(target.cpu().numpy())
#                 y_pred.append(preds.cpu().numpy())

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         if args.data_set != 'COMBINED':
#             acc1 = accuracy(output, target, topk=(1,))[0]
#             metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     # Initialize stats 
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#     if args.data_set == 'COMBINED': # For the combined training, balanced acc on each task and avg of acc/bal acc on all tasks are reported.
#         task_names = ["singleLane", "crosswalk", "green30", "NSH", "streetlight"]
#         task_bal_accs, task_accs = [], []

#         for i, name in enumerate(task_names):
#             y_t = np.concatenate(all_targets[i])
#             y_p = np.concatenate(all_preds[i])

#             # mask out ignored targets (-1)
#             mask = y_t != -1
#             if mask.sum() == 0:
#                 print(f"Task {name}: skipped (all ignored)")
#                 continue

#             y_t = y_t[mask]
#             y_p = y_p[mask]

#             bacc = balanced_accuracy_score(y_t, y_p) * 100 # Balanced Acc
#             task_bal_accs.append(bacc)
#             stats[f'{name}_balanced_acc'] = bacc

#             acc = (y_p == y_t).mean() * 100 # plain Balanced Acc
#             task_accs.append(acc)
#             stats[f'{name}_acc'] = acc
#             print(f"Task {name}: Acc = {acc:.2f}, BalancedAcc = {bacc:.2f}")

#         if task_bal_accs:
#             stats['avg_balanced_acc'] = np.mean(task_bal_accs)
#             print(f"Average BalancedAcc across 5 tasks = {stats['avg_balanced_acc']:.2f}")
#         if task_accs:
#             avg_acc = np.mean(task_accs)
#             metric_logger.meters['acc1'].update(avg_acc, n=1)  # acc1 = avg plain acc, for simplicity
#             stats['acc1'] = metric_logger.meters['acc1'].global_avg  
#             print(f"Average Acc across 5 tasks = {stats['acc1']:.2f}")

#     else:
#         # Flatten all collected labels
#         y_true = np.concatenate(y_true)
#         y_pred = np.concatenate(y_pred)

#         if args.data_set == 'STREET':
#             f1 = f1_score(y_true, y_pred, average='binary', pos_label=1) * 100
#             # f1 = f1_score(y_true, y_pred, average='macro') * 100
#         else:
#             f1 = f1_score(y_true, y_pred, average='macro') * 100
            
#         balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
#         stats['f1'] = f1
#         stats['balanced_accuracy'] = balanced_acc

#         print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} '
#             'F1 {f1:.2f} BalancedAcc {bacc:.2f}'
#             .format(top1=metric_logger.acc1, losses=metric_logger.loss,
#                     f1=f1, bacc=balanced_acc))

#         # Multicls confusion matrix
#         mcm = multilabel_confusion_matrix(y_true, y_pred)
#         n_classes = mcm.shape[0]

#         for i in range(n_classes):
#             tn, fp, fn, tp = mcm[i].ravel()
#             # Convert to native types to avoid JSON error
#             tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

#             stats[f'class_{i}_TP'] = tp
#             stats[f'class_{i}_FP'] = fp
#             stats[f'class_{i}_FN'] = fn
#             stats[f'class_{i}_TN'] = tn
#             stats[f'class_{i}_precision'] = float(precision)
#             stats[f'class_{i}_recall'] = float(recall)

#             print(f'Class {i}:')
#             print(f'  TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}')
#             print(f'  Precision = {precision:.3f}, Recall = {recall:.3f}')
        

#     return stats


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, args):
    # loss: for single-task (non-COMBINED) we use CE; for binary voting we compute BCE on final pos-logit
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # ensure meters used later exist
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Test:'

    model.eval()

    if args.data_set == 'COMBINED':
        all_preds = [[] for _ in range(5)]
        all_targets = [[] for _ in range(5)]
    else:
        y_true = []
        y_pred = []

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # support two possible dataloader tuple layouts:
        # - (images, target)
        # - (_, images, target) for IndexedDataset
        if len(batch) == 3:
            _, images, target = batch
        else:
            images, target = batch

        # If dataset returns 4 crops as a list: handle that below
        is_split4 = isinstance(images, (list, tuple)) and len(images) == 4

        # move targets
        target = target.to(device, non_blocking=True)

        logits_list = []
        with amp_autocast():
            if args.attn_map:
                # do visualization but still perform a forward pass for proper metrics
                # (you can keep existing visualize code; here we just avoid `continue`)
                # try visualizing only first few positives to avoid slowdowns
                try:
                    # if images is split-4, visualize first crop set
                    vis_img = images[0] if is_split4 else images
                    # ensure vis_img is on cpu & single sample for your visualize util
                    # (user's visualization code omitted here)
                    # utils.visualize_attn(...)  # optional
                    pass
                except Exception as e:
                    print("attn_map visualization skipped due to:", e)

            # Forward: if split-4 -> forward each crop; else forward single batch
            if is_split4:
                # each item in images is a tensor (B,C,H,W)
                for c in range(len(images)):
                    crop_batch = images[c].to(device, non_blocking=True)
                    logits = model(crop_batch)  # (B, C)
                    logits_list.append(logits)
            else:
                images = images.to(device, non_blocking=True)
                output = model(images)  # (B, C) or (B, total_heads) for COMBINED
                if args.data_set == 'COMBINED':
                    # we still need to process COMBINED from a single forward
                    logits_bin1 = output[:, 0:2]
                    logits_bin2 = output[:, 2:4]
                    logits_bin3 = output[:, 4:6]
                    logits_3cls = output[:, 6:9]
                    logits_bin4 = output[:, 9:11]
                    logits_list = [logits_bin1, logits_bin2, logits_bin3, logits_3cls, logits_bin4]

                else:
                    # single-task: treat output as single logits tensor
                    logits_list = [output]

        # --------- compute loss / predictions ----------
        if args.data_set == 'COMBINED':
            # compute masked loss (your masked_ce_loss expects lists)
            # targets slicing (already on device)
            target_bin1 = target[:, 0]
            target_bin2 = target[:, 1]
            target_bin3 = target[:, 2]
            target_3cls = target[:, 3]
            target_bin4 = target[:, 4]
            targets_list = [target_bin1, target_bin2, target_bin3, target_3cls, target_bin4]

            loss = masked_ce_loss(logits_list, targets_list)

            # predictions per task
            preds_list = [logits.argmax(dim=1).cpu().numpy() for logits in logits_list]
            for i in range(len(preds_list)):
                all_preds[i].append(preds_list[i])
                all_targets[i].append(targets_list[i].cpu().numpy())

            # For per-batch acc1 / loss logging, compute an average plain acc across tasks (ignoring -1)
            per_task_accs = []
            for i in range(len(all_preds)):
                # combine this batch preds/targets for task i
                # but in streaming code it's simpler to compute plain acc on this batch
                # get preds for this batch (already appended)
                pass  # we'll report aggregated metrics below

            # For metric_logger.acc1 we will set it later after aggregation
            batch_acc1 = 0.0

        else:
            # non-COMBINED: could be split4 or single
            if is_split4:
                # logits_list: list of N tensors shape (B, C)
                if args.nb_classes == 2:
                    pos_idx = 1
                    pos_probs = [F.softmax(l, dim=1)[:, pos_idx] for l in logits_list]  # list of (B,)
                    pos_stack = torch.stack(pos_probs, dim=0)  # (N, B)
                    final_pos_prob = pos_stack.max(dim=0)[0]  # (B,)
                    final_pos_logit = torch.log(final_pos_prob.clamp(1e-6,1-1e-6) / (1.0 - final_pos_prob.clamp(1e-6,1-1e-6)))
                    final_preds = (final_pos_prob > 0.5).long()
                    loss = F.binary_cross_entropy_with_logits(final_pos_logit, target.float())
                else:
                    # multiclass: max over logits
                    logits_stack = torch.stack(logits_list, dim=0)  # (N, B, C)
                    final_logits, _ = logits_stack.max(dim=0)  # (B, C)
                    loss = criterion(final_logits, target)
                    final_preds = final_logits.argmax(dim=1)
            else:
                # single forward
                output = logits_list[0]  # (B,C)
                if args.nb_classes == 2:
                    # binary using BCEWithLogits on logits[:,1] or use CrossEntropy
                    # Use CrossEntropy for stability (equivalent)
                    loss = criterion(output, target)  # CrossEntropy works for 2-class too
                    final_preds = output.argmax(dim=1)
                else:
                    loss = criterion(output, target)
                    final_preds = output.argmax(dim=1)

            # collect predictions/targets
            y_true.append(target.cpu().numpy())
            y_pred.append(final_preds.cpu().numpy())

        # update per-batch logging
        batch_size = target.size(0)
        metric_logger.update(loss=loss.item())
        if args.data_set != 'COMBINED':
            acc1 = accuracy(torch.from_numpy(np.concatenate([p for p in [final_preds.cpu().numpy()]]) ) .to(device), target, topk=(1,))[0] \
                   if False else (final_preds == target).float().sum() / batch_size * 100
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # end batch loop
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.data_set == 'COMBINED':
        # aggregate per-task
        task_names = ["singleLane", "crosswalk", "green30", "NSH", "streetlight"]
        task_bal_accs, task_accs = [], []
        for i, name in enumerate(task_names):
            if len(all_targets[i]) == 0:
                continue
            y_t = np.concatenate(all_targets[i])
            y_p = np.concatenate(all_preds[i])
            mask = y_t != -1
            if mask.sum() == 0:
                continue
            y_t = y_t[mask]; y_p = y_p[mask]
            bacc = balanced_accuracy_score(y_t, y_p) * 100
            acc = (y_p == y_t).mean() * 100
            stats[f'{name}_balanced_acc'] = bacc
            stats[f'{name}_acc'] = acc
            task_bal_accs.append(bacc); task_accs.append(acc)
            print(f"Task {name}: Acc = {acc:.2f}, BalancedAcc = {bacc:.2f}")

        if task_bal_accs:
            stats['avg_balanced_acc'] = np.mean(task_bal_accs)
        if task_accs:
            stats['acc1'] = np.mean(task_accs)
            metric_logger.meters['acc1'].update(stats['acc1'], n=1)
        return stats

    else:
        # flatten and compute f1 / balanced acc as before
        if len(y_true) == 0:
            # nothing evaluated
            stats['f1'] = 0.0
            stats['balanced_accuracy'] = 0.0
            return stats

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        if args.data_set == 'STREET':
            f1 = f1_score(y_true, y_pred, average='binary', pos_label=1) * 100
        else:
            f1 = f1_score(y_true, y_pred, average='macro') * 100
        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
        stats['f1'] = f1
        stats['balanced_accuracy'] = balanced_acc

        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} '
              'F1 {f1:.2f} BalancedAcc {bacc:.2f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss,
                      f1=f1, bacc=balanced_acc))

        # confusion matrices etc (keeps your code)
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        n_classes = mcm.shape[0]
        for i in range(n_classes):
            tn, fp, fn, tp = mcm[i].ravel()
            tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            stats[f'class_{i}_TP'] = tp
            stats[f'class_{i}_FP'] = fp
            stats[f'class_{i}_FN'] = fn
            stats[f'class_{i}_TN'] = tn
            stats[f'class_{i}_precision'] = float(precision)
            stats[f'class_{i}_recall'] = float(recall)
            print(f'Class {i}:  TP={tp}, FP={fp}, FN={fn}, TN={tn}  Precision={precision:.3f}, Recall={recall:.3f}')

        return stats
