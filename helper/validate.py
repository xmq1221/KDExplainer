import logging

import tqdm
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from .util import AverageMeter, accuracy


def validate(val_loader: DataLoader, model: Module, criterion: Module, device: torch.device):
    logger = logging.getLogger("validate")
    logger.info("Start validation")

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for x, target in tqdm.tqdm(val_loader):
            x = x.to(device)
            target = target.to(device)

            # compute output
            output = model(x)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.shape[0])
            top1.update(acc1[0], x.shape[0])
            top5.update(acc5[0], x.shape[0])

        logger.info("acc@1: %.4f acc@5: %.4f, loss: %.5f", top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


def validate_LTB(
    cfg: Dict[str, Any],
    val_loader: DataLoader,
    model: Module,
    criterion: Module,
    device: torch.device,
    num_classes: int,
    t: int,
    epoch: int,
    loss_method: str,
    stage: str    
    ):

    logger = logging.getLogger("validate")
    logger.info("Start validation")

    # if loss_method == 'nce':
    if cfg["model"]["task"] == 'mt':

        losses = [AverageMeter() for _ in range(num_classes)]
        top1 = [AverageMeter() for _ in range(num_classes)]
        top5 = [AverageMeter() for _ in range(num_classes)]

    # elif loss_method =='ce':
    elif cfg["model"]["task"] == 'mc':

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for x, target in tqdm.tqdm(val_loader):
            x = x.to(device)
            target = target.to(device)

            if stage == 's1':
                output = model(x, t/(epoch), True)
            elif stage == 's2':
                output = model(x, t/(epoch), False)

            # if loss_method == 'ce':
            if cfg["model"]["task"] == 'mc':

                loss = criterion(output, target.squeeze())
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
             
                # prec1 = accuracy(output, target.squeeze())
                losses.update(loss.item(), x.shape[0])
                top1.update(acc1[0], x.shape[0])
                top5.update(acc5[0], x.shape[0])   

                loss_avg = losses.avg
                top1_avg = top1.avg
                top5_avg = top5.avg

            # elif loss_method == 'nce':
            elif cfg["model"]["task"] == 'mt':

                loss = []
                acc1, acc5 = [], []

                for j in range(len(output)):
                    loss.append(criterion(output[j], target[:, j]))
                    acc1.append(accuracy(output[j], target[:, j], topk=(1, 1))[0])
                    acc5.append(accuracy(output[j], target[:, j], topk=(1, 1))[1])

                    losses[j].update(loss[j].item(), x.shape[0])
                    top1[j].update(acc1[j], x.shape[0])
                    top5[j].update(acc5[j], x.shape[0])

                losses_avg = [losses[k].avg for k in range(len(losses))]
                top1_avg = [top1[k].avg for k in range(len(top1))]
                top5_avg = [top5[k].avg for k in range(len(top5))]

                loss_avg = sum(losses_avg) / len(losses_avg)
                top1_avg = sum(top1_avg) / len(top1_avg)
                top5_avg = sum(top5_avg) / len(top5_avg)

                # loss = sum(loss)
        logger.info("acc@1: %.4f acc@5: %.4f, loss: %.5f", top1_avg, top5_avg, loss_avg)
        # logger.info("acc@1: %.4f acc@5: %.4f, loss: %.5f", top1.avg, top5.avg, losses.avg)

    # return top1.avg, top5.avg, losses.avg
    return top1_avg, top5_avg, loss_avg
    # return (loss_avg, prec1_avg)

