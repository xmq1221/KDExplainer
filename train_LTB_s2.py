import os
import argparse
from typing import Dict, Any
import copy
import logging

import yaml

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataset
# from models import get_model
from optim import get_optimizer

from helper.util import str2bool, get_logger, preserve_memory, adjust_learning_rate_stage2
from helper.util import make_deterministic
from helper.util import AverageMeter, accuracy
from helper.validate import validate_LTB

from LEARNTOBRANCH import LEARNTOBRANCH_Deep


def get_dataloader(cfg: Dict[str, Any]):
    # dataset
    dataset_cfg = cfg["dataset"]
    train_dataset = get_dataset(split="train", **dataset_cfg)
    val_dataset = get_dataset(split="val", **dataset_cfg)
    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg["validation"]["batch_size"],
        num_workers=cfg["validation"]["num_workers"],
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader, num_classes


def train_epoch(
    cfg: Dict[str, Any],
    epoch: int,
    train_loader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    tb_writer: SummaryWriter,
    device: torch.device,
    loss_method: str
):
    logger = logging.getLogger("train_epoch")
    logger.info("Start training one epoch...")

    # if loss_method == 'nce':
    if cfg["model"]["task"] == 'mt':
        losses = [AverageMeter() for _ in range(cfg["dataset"]["num_classes"])]
        top1 = [AverageMeter() for _ in range(cfg["dataset"]["num_classes"])]
        top5 = [AverageMeter() for _ in range(cfg["dataset"]["num_classes"])]

    # elif loss_method =='ce':
    elif cfg["model"]["task"] == 'mc':

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    model.train()

    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    for idx, (x, target) in enumerate(train_loader):
        __global_values__["it"] += 1

        x = x.to(device)
        target = target.to(device)

        # ===================forward=====================
        logit = model(x, int(cfg["training"]["t"])/(epoch), False)
        # logit = model(x)
        # if loss_method == 'ce':
        if cfg["model"]["task"] == 'mc':
            # print(logit.shape, target.shape)
            # return 0
            loss = criterion(logit, target.squeeze())
            acc1, acc5 = accuracy(logit, target.squeeze(), topk=(1, 5))
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
            print(len(logit), logit[0].shape, logit[1].shape)
            print(target.shape)
            return 0
            for j in range(len(logit)):
                print(logit[j].shape)
                print(logit)

                return 0

                loss.append(criterion(logit[j], target[:, j]))
                acc1.append(accuracy(logit[j], target[:, j], topk=(1, 1))[0])
                acc5.append(accuracy(logit[j], target[:, j], topk=(1, 1))[1])

                losses[j].update(loss[j].item(), x.shape[0])
                top1[j].update(acc1[j], x.shape[0])
                top5[j].update(acc5[j], x.shape[0])

            losses_avg = [losses[k].avg for k in range(len(losses))]
            top1_avg = [top1[k].avg for k in range(len(top1))]
            top5_avg = [top5[k].avg for k in range(len(top5))]

            loss_avg = sum(losses_avg) / len(losses_avg)
            top1_avg = sum(top1_avg) / len(top1_avg)
            top5_avg = sum(top5_avg) / len(top5_avg)

            loss = sum(loss)
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        tb_writer.add_scalars(
            main_tag="train/acc",
            tag_scalar_dict={
                "@1": top1_avg,
                "@5": top5_avg,
            },
            global_step=__global_values__["it"]
        )
        tb_writer.add_scalar("train/loss", loss_avg, global_step=__global_values__["it"])
        # if idx % cfg["training"]["print_iter_freq"] == 0:
        #     logger.info(
        #         "Epoch: %3d|%3d, idx: %d, total iter: %d, loss: %.5f, acc@1: %.4f, acc@5: %.4f",
        #         epoch, cfg["training"]["epochs"],
        #         idx, __global_values__["it"],
        #         losses.val, top1.val, top5.val
        #     )
        if idx % cfg["training"]["print_iter_freq"] == 0:
            logger.info(
                "Epoch: %3d|%3d, idx: %d, total iter: %d, loss: %.5f, acc@1: %.4f, acc@5: %.4f",
                epoch, cfg["training"]["epochs"],
                idx, __global_values__["it"],
                loss_avg, top1_avg, top5_avg
            )

    # return top1.avg, losses.avg
    return top1_avg, loss_avg


def train(
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    lr_scheduler: MultiStepLR,
    tb_writer: SummaryWriter,
    device: torch.device,
    ckpt_dir: str
):
    logger = logging.getLogger("train")
    logger.info("Start training...")

    best_acc = 0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # print(type(cfg["training"]["lr_global"]), type(cfg["training"]["epochs"]), type(cfg["dataset"]["num_classes"]))
        # sleep(0)
        # print(epoch, type(epoch))
        global_lr = adjust_learning_rate_stage2(
                    optimizer=optimizer,
                    epoch_current=epoch
                    )

        logger.info("Start training epoch: %d, current lr: %.6f", 
                    epoch, lr_scheduler.get_last_lr()[0])
        logger.info("current global_lr: %.6f",  global_lr)

        train_acc, train_loss = train_epoch(
            cfg=cfg,
            epoch=epoch,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            tb_writer=tb_writer,
            device=device,
            loss_method=cfg["model"]["loss_method"]
        )

        tb_writer.add_scalar("epoch/train_acc", train_acc, epoch)
        tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)

        val_acc, val_acc_top5, val_loss = validate_LTB(
            cfg=cfg,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            num_classes=cfg["dataset"]["num_classes"],
            t=cfg["training"]["t"],
            epoch=epoch,
            loss_method=cfg["model"]["loss_method"],
            stage='s2'
        )

        tb_writer.add_scalar("epoch/val_acc", val_acc, epoch)
        tb_writer.add_scalar("epoch/val_loss", val_loss, epoch)
        tb_writer.add_scalar("epoch/val_acc_top5", val_acc_top5, epoch)

        logger.info(
            "Epoch: %04d | %04d, acc: %.4f, loss: %.5f, val_acc: %.4f, val_acc_top5: %.4f, val_loss: %.5f",
            epoch, cfg["training"]["epochs"],
            train_acc, train_loss,
            val_acc, val_acc_top5, val_loss
        )

        lr_scheduler.step()

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "acc": val_acc,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
        
        # regular saving
        if epoch % cfg["training"]["save_ep_freq"] == 0:
            logger.info("Saving epoch %d checkpoint...", epoch)

            save_file = os.path.join(ckpt_dir, "epoch_{}.pth".format(epoch))
            torch.save(state, save_file)

        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_ep = epoch

            save_file = os.path.join(ckpt_dir, "best.pth")
            logger.info("Saving the best model with acc: %.4f", best_acc)
            torch.save(state, save_file)
        logger.info("Epoch: %04d | %04d, best acc: %.4f,", epoch, cfg["training"]["epochs"], best_acc)

    logger.info("Final best accuracy: %.5f, at epoch: %d", best_acc, best_ep)


def main(
    cfg_filepath: str,
    file_name_cfg: str,
    logdir: str,
    gpu_preserve: bool = False,
    debug: bool = False
):
    with open(cfg_filepath) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if debug:
        cfg["training"]["num_workers"] = 0
        cfg["validation"]["num_workers"] = 0

    seed = cfg["training"]["seed"]

    ckpt_dir = os.path.join(logdir, "ckpt")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    formatter = (
        cfg["model"]["name"],
        cfg["dataset"]["name"],
        args.stage,
        cfg["dataset"]["loss_method"]
    )
    writer = SummaryWriter(
        log_dir=os.path.join(
            logdir,
            "tf-logs",
            file_name_cfg.format(*formatter)
        ),
        flush_secs=1
    )

    train_log_dir = os.path.join(logdir, "train-logs")
    os.makedirs(train_log_dir, exist_ok=True)
    logger = get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            train_log_dir,
            "training-" + file_name_cfg.format(*formatter) + ".log"
        )
    )
    logger.info("Start running with config: \n{}".format(yaml.dump(cfg)))

    # set seed
    make_deterministic(seed)
    logger.info("Set seed : {}".format(seed))

    if gpu_preserve:
        logger.info("Preserving memory...")
        preserve_memory(args.preserve_percent)
        logger.info("Preserving memory done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataloaders
    logger.info("Loading datasets...")
    train_loader, val_loader, num_classes = get_dataloader(cfg)

    logger.info("num_classes: {}".format(num_classes))

    # get models
    logger.info("Loading model {}...".format(cfg["model"]["name"]))

    model = LEARNTOBRANCH_Deep(
        dataset=cfg["dataset"]["name"], 
        num_attributes=cfg["dataset"]["num_classes"],
        loss_method= cfg["model"]["loss_method"])

    branch_params_list = list(map(id, model.branch_2.parameters())) + list(map(id, model.branch_3.parameters())) + \
                         list(map(id, model.branch_4.parameters()))
    global_params = filter(lambda p: id(p) not in branch_params_list, model.parameters())

    logger.info(model)

    # get loss modules
    criterion = CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # optimizer
    # optimizer = torch.optim.Adam(params,
    #                             weight_decay=float(cfg["training"]["optimizer"]["weight_decay"]))
    # # optimizer = get_optimizer(model.parameters(), cfg["training"]["optimizer"])
    optimizer = torch.optim.SGD(
                params=global_params,
                lr=cfg["training"]["lr_stage2"],
                weight_decay=cfg["training"]["optimizer"]["weight_decay_stage2"],
                momentum=cfg["training"]["optimizer"]["momentum"])

    checkpoint = torch.load(cfg["model"]["pretrained"])
    model.load_state_dict(checkpoint['model'])
    logger.info("=> loaded checkpoint '{}'".format(cfg["model"]["pretrained"]))
    model._initialize_weights()

    lr_scheduler = None
    lr_scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=cfg["training"]["lr_decay_epochs"],
        gamma=cfg["training"]["lr_decay_rate"]
    )

    # print(optimizer)
    logger.info(optimizer)

    train(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        tb_writer=writer,
        device=device,
        ckpt_dir=ckpt_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--file_name_cfg", type=str)
    parser.add_argument("--stage", type=str)
    parser.add_argument("--gpu_preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--preserve_percent", type=float, default=0.95)
    args = parser.parse_args()

    __global_values__ = dict(it=0)
    main(
        cfg_filepath=args.config,
        file_name_cfg=args.file_name_cfg,
        logdir=args.logdir,
        gpu_preserve=args.gpu_preserve,
        debug=args.debug
    )
