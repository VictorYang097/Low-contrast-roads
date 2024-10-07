"""
Training codes on our LC-Roads dataset
"""
import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tqdm import tqdm
from datetime import datetime

from lib.LCMorph.LCMorph import LCMorph

from utils.utils import LR_Scheduler
from utils.loss import dice_bce_loss_func
from utils.metrics import Evaluator


# training function
def train(train_loader, model, optimizer, scheduler, epoch):
    model.train()
    evaluator.reset()
    train_loss = 0.0

    if not os.path.exists("run/Low_Contrast/{}".format(opt.model_name)):
        os.makedirs("run/Low_Contrast/{}".format(opt.model_name))
    file = open("run/Low_Contrast/{}/train_log.txt".format(opt.model_name), "a")
    for i, sample in enumerate(tqdm(train_loader)):
        scheduler(optimizer, i, epoch)
        optimizer.zero_grad()
        # ---- data prepare ----
        images = sample['image']
        gts = sample['label']

        images = images.cuda()
        gts = gts.cuda()

        # ---- forward ----
        pred_coarse, pred_refine = model(images)

        # ---- loss function ----
        gts = torch.unsqueeze(gts, 1)
        loss = (dice_bce_loss_func(pred_coarse, gts) + dice_bce_loss_func(pred_refine, gts)) / 2

        # ---- backward ----
        loss.backward()

        optimizer.step()

        # ---- recording loss ----
        preds = pred_refine.data.cpu().numpy()
        labels = gts.cpu().numpy()
        preds[preds >= 0.3] = 1
        preds[preds < 0.3] = 0

        evaluator.add_batch(labels.astype(int), preds.astype(int))
        train_loss += loss.item()

        if (i+1) % 20 == 0 or (i+1) == train_batch:
            ACC = evaluator.Pixel_Accuracy()
            ACC_class = evaluator.Pixel_Accuracy_Class()
            mIOU = evaluator.Mean_Intersection_over_Union()
            IOU = evaluator.Intersection_over_Union()
            Precision = evaluator.Pixel_Precision()
            Recall = evaluator.Pixel_Recall()
            F1 = evaluator.Pixel_F1()

            train_result = '{} Epoch [{:03d}/{:03d}], Train Batch [{:04d}/{:04d}], Loss {:.4f}, Acc {}, Acc_class {}, mIOU {}, IOU {}, Precision {}, Recall {}, F1 {}'.format(
                    datetime.now(), epoch, opt.epoch, i+1, train_batch, train_loss/(i+1),
                    ACC, ACC_class, mIOU,
                    IOU, Precision, Recall, F1
                )
            file.write(train_result + '\n')
            print(train_result)

            # scheduler.get_lr()

    file.close()

    iou = evaluator.Intersection_over_Union()
    if not os.path.exists("run/Low_Contrast/{}/ckpt/".format(opt.model_name)):
        os.makedirs("run/Low_Contrast/{}/ckpt/".format(opt.model_name))
    save_path = 'run/Low_Contrast/{}/ckpt/'.format(opt.model_name)

    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict(),  # multi GPUs
        # 'state_dict': model.state_dict(),  # single GPU
        'optimizer': optimizer.state_dict(),
        'train_iou': iou
    }, save_path + '{}.pth.tar'.format(opt.model_name))
    print('[Saving Snapshot:]', save_path + '{}.pth.tar'.format(opt.model_name))


# validation function
def valid(valid_loader, model, epoch, best):
    model.eval()
    evaluator.reset()
    valid_loss = 0.0

    if not os.path.exists("run/Low_Contrast/{}".format(opt.model_name)):
        os.makedirs("run/Low_Contrast/{}".format(opt.model_name))
    file = open("run/Low_Contrast/{}/valid_log.txt".format(opt.model_name), "a")

    for i, sample in enumerate(tqdm(valid_loader)):
        # ---- data prepare ----
        images = sample['image']
        gts = sample['label']

        images = images.cuda()
        gts = gts.cuda()

        # ---- forward ----
        with torch.no_grad():
            pred_coarse, pred_refine = model(images)

        # ---- loss function ----
        gts = torch.unsqueeze(gts, 1)
        loss = (dice_bce_loss_func(pred_coarse, gts) + dice_bce_loss_func(pred_refine, gts)) / 2

        # ---- recording loss ----
        preds = pred_refine.data.cpu().numpy()
        labels = gts.cpu().numpy()
        preds[preds >= 0.3] = 1
        preds[preds < 0.3] = 0

        evaluator.add_batch(labels.astype(int), preds.astype(int))
        valid_loss += loss.item()

        if (i+1) % 5 == 0 or (i+1) == valid_batch:
            ACC = evaluator.Pixel_Accuracy()
            ACC_class = evaluator.Pixel_Accuracy_Class()
            mIOU = evaluator.Mean_Intersection_over_Union()
            IOU = evaluator.Intersection_over_Union()
            Precision = evaluator.Pixel_Precision()
            Recall = evaluator.Pixel_Recall()
            F1 = evaluator.Pixel_F1()

            valid_result = '{} Epoch [{:03d}/{:03d}], Valid Batch [{:04d}/{:04d}], Loss {:.4f}, Acc {}, Acc_class {}, mIOU {}, IOU {}, Precision {}, Recall {}, F1 {}'.format(
                    datetime.now(), epoch, opt.epoch, i+1, valid_batch, valid_loss/(i+1),
                    ACC, ACC_class, mIOU,
                    IOU, Precision, Recall, F1
                )
            file.write(valid_result + '\n')
            print(valid_result)

    file.close()

    iou = evaluator.Intersection_over_Union()

    if not os.path.exists("run/Low_Contrast/{}/ckpt/".format(opt.model_name)):
        os.makedirs("run/Low_Contrast/{}/ckpt/".format(opt.model_name))
    save_path = 'run/Low_Contrast/{}/ckpt/'.format(opt.model_name)

    if iou > best:
        torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),  # multi GPUs
            # 'state_dict': model.state_dict(), # single GPU
            'optimizer': optimizer.state_dict(),
            'iou': iou
        }, save_path + '{}_best.pth.tar'.format(opt.model_name))
        return iou
    return best


if __name__ == '__main__':
    models = {'LCMorph': LCMorph}

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=80, help='epoch number')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='batch size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    # optimizer params
    parser.add_argument('--lr', type=float,
                        default=0.01, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='poly',  # learning rate decay
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--model_name', type=str,
                        default='LCMorph')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--base_size', type=int,
                        default=512, help='base image size')
    parser.add_argument('--crop_size', type=int,
                        default=512, help='crop image size')
    parser.add_argument('--start_epoch', type=int,
                        default=1, help='train from this epoch')
    parser.add_argument('--no_improve', type=int,
                        default=8, help='After a few epochs, if IOU doesn\'t improve, stop training')
    parser.add_argument('--resume', type=bool,
                        default=False, help='whether to continue training')

    opt = parser.parse_args()

    # ---- random seed ----
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # ---- build datasets ----
    train_ds = lowDataSet (opt, base_dir='/home/jovyan/work/LC-Roads/', split='train')
    valid_ds = lowDataSet (opt, base_dir='/home/jovyan/work/LC-Roads/', split='valid')

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=opt.batch_size, shuffle=False, num_workers=opt.batch_size)
    train_batch = len(train_loader)
    valid_batch = len(valid_loader)

    evaluator = Evaluator(num_class=2)  # evaluation metrics

    # ---- build models ----
    network = models[opt.model_name]

    model = network().cuda()

    model = DataParallel(model, device_ids=[0, 1, 2, 3])

    torch.cuda.manual_seed_all(opt.seed)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=opt.lr)
    scheduler = LR_Scheduler(opt.lr_scheduler, opt.lr, opt.epoch, train_batch)

    # Continue training
    if opt.resume:
        checkpoint = torch.load('run/Low_Contrast/{}/ckpt/{}.pth.tar'.format(opt.model_name, opt.model_name))
        checkpoint_best = torch.load('run/Low_Contrast/{}/ckpt/{}_best.pth.tar'.format(opt.model_name, opt.model_name))

        model.module.load_state_dict(checkpoint['state_dict'])  # multi GPUs
        # model.load_state_dict(checkpoint['state_dict'])  # single GPU
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_IOU = checkpoint_best['iou']

        no_improve = 0

        del checkpoint
        del checkpoint_best

        print("Continue to Train {} From Epoch {}".format(opt.model_name, start_epoch))
    # Training from the scratch
    else:
        start_epoch = 1
        best_IOU = 0.0

        no_improve = 0

        print("Start to Train {}".format(opt.model_name))

    # ---- Training & Validating ----
    for epoch in range(start_epoch, opt.epoch+1):

        print('----------------------Training-----------------------')
        train(train_loader, model, optimizer, scheduler, epoch)

        print('----------------------Validating-----------------------')
        current_IOU = valid(valid_loader, model, epoch, best_IOU)

        if current_IOU > best_IOU:
            best_IOU = current_IOU
            no_improve = 0
        else:
            no_improve += 1

        print("Best IOU is: {}".format(best_IOU))
        file = open("run/Low_Contrast/{}/valid_log.txt".format(opt.model_name), "a")
        file.write("Best IOU is: {}".format(best_IOU) + "\n")
        file.close()

        # Early stopping
        if no_improve >= opt.no_improve:
            print("Early stopping with best IOU: {} ".format(best_IOU))
            file = open("run/Low_Contrast/{}/valid_log.txt".format(opt.model_name), "a")
            file.write("Early stopping with best IOU: {} ".format(best_IOU) + "\n")
            file.close()

            break
