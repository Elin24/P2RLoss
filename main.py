# -*- coding: utf-8 -*-

import sys
print(sys.executable)

import os
import time
import random
import argparse
import datetime
import torch
import torch.nn.functional as tF
from torch import optim
from torch.optim.lr_scheduler import StepLR
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from datasets import build_loader
from losses import build_loss
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, plot_curve, set_seed
STAGE_1, STAGE_2 = 0, 1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_args_parser():
    parser = argparse.ArgumentParser('Counting Everything training and evaluation script', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--label', type=float, help='percent of label data')
    parser.add_argument('--protocol', type=str, help='data-splitting protocol path')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main_worker(config):
    data_loader_train, data_loader_val = build_loader(config.DATA, mode='train'), build_loader(config.DATA, mode='test')

    logger.info(f"Creating model with:{config.MODEL.NAME}")
    student, teacher = build_model(config.MODEL)
    student.cuda(); teacher.cuda()
    
    criterion, test_criterion = build_loss(config.MODEL)
    criterion.cuda(); test_criterion.cuda()

    param_dicts = [
        {
            "params": [p for n, p in student.named_parameters() if "encoders" not in n and p.requires_grad]
        }, {
            "params": [p for n, p in student.named_parameters() if "encoders" in n and p.requires_grad],
            "lr": config.TRAIN.BACKBONE_LR,
        },
    ]

    optimizer = optim.Adam(param_dicts, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    

    n_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    lr_scheduler = StepLR(optimizer, step_size=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS, gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    max_accuracy = [1e6] * 3

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, [teacher, student], optimizer, lr_scheduler, logger)
        # mae, mse, loss = validate(config, data_loader_val, student, test_criterion)
        # max_accuracy = (mae, mse, loss)
        # logger.info(f"Accuracy of the network on the test images: {mae:.2f} | {mse:.2f}")
        if config.EVAL_MODE:
            return

    global STAGE_1, STAGE_2
    # STAGE_1 = max(25, int(5 / config.DATA.LABEL_PERCENT))
    STAGE_1 = 25 # config.TRAIN.EPOCHS
    STAGE_2 = STAGE_1 * 2


    logger.info(f"Start training: [STAGE_1: {STAGE_1}] [STAGE_2: {STAGE_2}]")
    start_time = time.time()
    epostack, maestack, msestack, lossstack = [], [], [], []
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        
        train_one_epoch(config, [teacher, student], criterion, data_loader_train, optimizer, epoch)

        if epoch == STAGE_2 - 1:
            save_checkpoint(config, "stage2", [teacher, student], max_accuracy, logger)

        # if lr_scheduler is not None: lr_scheduler.step()
        
        if epoch > 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            mae, mse, loss = validate(config, data_loader_val, student, test_criterion)
            epostack.append(epoch)
            maestack.append(mae)
            msestack.append(mse)
            lossstack.append(loss)
            plot_curve('mae', epostack, maestack, os.path.join('exp', config.TAG, 'train.log', 'mae_curve.png'))
            plot_curve('mse', epostack, msestack, os.path.join('exp', config.TAG, 'train.log', 'mse_curve.png'))
            plot_curve('loss', epostack, lossstack, os.path.join('exp', config.TAG, 'train.log', 'loss_curve.png'))

            logger.info(f"Accuracy of the network on the test images: {loss:.6f}")

            if mae * 4 + mse < max_accuracy[0] * 4 + max_accuracy[1]:
                save_checkpoint(config, "best", [teacher, student], max_accuracy, logger)
                max_accuracy = (mae, mse, loss)
            logger.info(f'Min total MAE|MSE|Loss: {max_accuracy[0]:.6f} | {max_accuracy[1]:.2f} | {max_accuracy[2] * 1e5:.2f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch):
    teacher, student = model
    student.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (limg, lseq, lid, uimg, umasks, uid) in enumerate(data_loader):
        limg = limg.cuda(non_blocking=True)
        lseq = [d.cuda(non_blocking=True) for d in lseq]

        lden = student(limg)
        down_rate = limg.size(-1) // lden.size(-1)
        loss = criterion(lden, lseq, down_rate)

        if epoch >= STAGE_2:
            wa_img = uimg[:, :3].cuda(non_blocking=True)
            sa_img = uimg[:, 3:].cuda(non_blocking=True)
            cut_img_mask = umasks.cuda(non_blocking=True)
            cut_den_mask = tF.avg_pool2d(cut_img_mask, kernel_size=down_rate, stride=down_rate) >= 0.5

            sa_img_mask = sa_img * cut_img_mask

            with torch.inference_mode():
                tu_den = teacher(wa_img)

                tu_mask, tu_seq2, = [], []
                for tden in tu_den:
                    tseq = torch.nonzero((tden >= 0).squeeze())

                    peaks = tden.squeeze()[tseq[:, 0], tseq[:, 1]]
                    threshold = 0.8472978603872036 # sigmoid(0.8472978603872036) = 0.7
                    tu_mask.append(peaks > threshold)

                    tseq2 = tseq * down_rate + (down_rate - 1) / 2
                    tu_seq2.append(tseq2)

            weight = min(max((epoch - STAGE_2) * 0.01, 0), 2)
            su_den = student(sa_img_mask)    
            semi_loss = criterion(su_den, tu_seq2, down_rate, tu_mask, crop_den_masks=cut_den_mask)
            loss = (loss + weight * semi_loss) / (1 + weight)

        optimizer.zero_grad()
        loss.backward()

        grad_norm = get_grad_norm(student.parameters())
        optimizer.step()

        # # momentum update
        momentum = 0.998
        for para_t, para_s in zip(teacher.parameters(), student.parameters()):
            para_t.data.copy_(para_t.data * momentum + para_s.data * (1.0 - momentum))

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), limg.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr*1e5:.3f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val :.3f} ({loss_meter.avg :.3f})\t'
                f'grad_norm {norm_meter.val  :.3f} ({norm_meter.avg :.3f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    mae_meter = AverageMeter()
    mse_meter = AverageMeter()

    end = time.time()
    for idx, (images, dotseq, imgid) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        dotseq = [d.cuda(non_blocking=True) for d in dotseq]
        cnt = torch.tensor([d.size(0) for d in dotseq]).float().cuda()
        bsize = images.size(0)
        # compute output
        with torch.no_grad():
            output = model(images)
            # outnum = output.sum(dim=(1,2,3)) / config.MODEL.FACTOR
            loss = criterion(output, dotseq, images.size(-1) // output.size(-1)).item()
            outnum = (output > 0).sum(dim=(1, 2, 3))
        diff = torch.abs(outnum - cnt)
        mae, mse = diff.mean(), (diff ** 2).mean()

        loss_meter.update(loss, bsize)
        mae_meter.update(mae.item(), bsize)
        mse_meter.update(mse.item(), bsize)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'Loss {loss_meter.val:.6f} ({loss_meter.avg:.6f})  '
                f'MAE {mae_meter.val:.3f} ({mae_meter.avg:.3f})  '
                f'MSE {mse_meter.val ** 0.5:.3f} ({mse_meter.avg ** 0.5:.3f})  '
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * MAE {mae_meter.avg:.3f} MSE {mse_meter.avg ** 0.5:.3f}')
    return mae_meter.avg, mse_meter.avg ** 0.5, loss_meter.avg

if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)
    _, config = get_args_parser()

    
    torch.cuda.set_device('cuda:0')
    set_seed(config.SEED)

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main_worker(config)
