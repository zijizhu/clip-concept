import os
import sys
import json
import torch
import logging
import argparse
from torch import nn
from tqdm import tqdm
from typing import Callable
from datetime import datetime
from torch.utils.data import DataLoader

from data.cub.cub_dataset import (
    CUBDataset,
    get_transforms_resnet101
)
from apn_debug import resnet_proto_IoU, Loss_fn, get_middle_graph


@torch.no_grad()
def compute_corrects(preds: torch.Tensor, class_ids: torch.Tensor):
    return torch.sum(torch.argmax(preds.data, dim=-1) == class_ids.data).item()


@torch.no_grad()
def val_epoch(model: nn.Module, attribute_seen, acc_fn: nn.Module | Callable, dataloader: DataLoader,
              dataset_size: int, epoch: int, batch_size: int, device: torch.device, logger: logging.Logger):

    running_corrects = 0

    for batch in tqdm(dataloader):
        batch_input, batch_target = batch['pixel_values'], batch['class_ids']
        batch_target.to(device)
        model.zero_grad()
        # map target labels
        input_v = torch.autograd.Variable(batch_input)
        label_v = torch.autograd.Variable(batch_target)
        input_v = input_v.to(device)
        label_v = label_v.to(device)
        output, pre_attri, attention, pre_class = model(input_v, attribute_seen)

        running_corrects += acc_fn(output, batch_target)

    epoch_acc = running_corrects / dataset_size
    logger.info(f'EPOCH {epoch} Val Acc: {epoch_acc:.4f}')


def main():

    class Args(argparse.Namespace):
        dataset='CUB'
        root='./'
        image_root='/Users/zhijiezhu/Developer/Research/APN-ZSL/datasets/'
        matdataset=True
        image_embedding='res101'
        class_embedding='att'
        preprocessing=True
        standardization=False
        ol=False
        validation=False
        batch_size=64
        nepoch=30
        classifier_lr=1e-06
        beta1=0.5
        cuda=True
        pretrain_classifier=''
        manualSeed=4896
        resnet_path='./pretrained_models/resnet101_c.pth.tar'
        train_id=0
        pretrained=None
        image_type='test_unseen_loc'
        pretrain_epoch=4
        pretrain_lr=0.0001
        all=False
        gzsl=False
        additional_loss=True
        xe=1.0
        attri=0.01
        regular=5e-06
        l_xe=1.0
        l_attri=0.1
        l_regular=0.04
        cpt=1e-09
        alibrated_stacking=0.7
        save_att=False
        IoU_scale=1.4
        IoU_thr=0.5
        resize_WH=False
        out_of_edge=False
        max_area_center=False
        KNOW_BIRD_BB=False
        train_mode='distributed'
        n_batch=300
        ways=8
        shots=3
        transform_complex=False
        awa_finetune=False
        use_group=True
        avg_pool=False
        only_evaluate=False
        resume=False
        dataroot='./data'
        checkpointroot='./checkpoint'
    
    opt = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #################
    # Setup logging #
    #################

    log_path = os.path.join('logs', f'{datetime.now().strftime("%Y-%m-%d_%H-%M")}_apn_debug.log')
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger(__name__)

    #################################
    # Setup datasets and transforms #
    #################################

    train_transforms, test_transforms = get_transforms_resnet101()

    dataset_train = CUBDataset(
        os.path.join('datasets', 'CUB'),
        num_attrs=312,
        split='train',
        transforms=train_transforms
    )
    dataset_val = CUBDataset(
        os.path.join('datasets', 'CUB'),
        num_attrs=312,
        split='val',
        transforms=test_transforms
    )
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8
    )

    ##############################
    # Load models and optimizers #
    ##############################
    attribute_seen = dataset_train.attribute_vectors_pt.T.to(device)
    model = resnet_proto_IoU(opt)

    loss_log = {'ave_loss': 0, 'l_xe_final': 0, 'l_attri_final': 0, 'l_regular_final': 0,
                        'l_xe_layer': 0, 'l_attri_layer': 0, 'l_regular_layer': 0, 'l_cpt': 0}

    #################
    # Training loop #
    #################
    parts = ['head', 'belly', 'breast', 'belly', 'wing', 'tail', 'leg', 'others']
    group_dic = json.load(open(os.path.join(opt.root, 'datasets', 'attri_groups_8.json')))
    sub_group_dic = json.load(open(os.path.join(opt.root, 'datasets', 'attri_groups_8_layer.json')))

    reg_weight = {'final': {'xe': opt.xe, 'attri': opt.attri, 'regular': opt.regular},
                  'layer4': {'l_xe': opt.l_xe, 'attri': opt.l_attri, 'regular': opt.l_regular, 'cpt': opt.cpt}}  # l denotes layer
    layer_name = model.extract[0]
    middle_graph = get_middle_graph(reg_weight[layer_name]['cpt'], model)

    criterion = nn.CrossEntropyLoss()
    criterion_regre = nn.MSELoss()

    print('Train and test...')
    for epoch in range(opt.nepoch):
        # print("training")
        model.to(device)
        model.train()
        current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
        realtrain = epoch > opt.pretrain_epoch
        if epoch <= opt.pretrain_epoch:   # pretrain ALE for the first several epoches
            optimizer = torch.optim.Adam(params=[model.prototype_vectors[layer_name], model.ALE_vector],
                                    lr=opt.pretrain_lr, betas=(opt.beta1, 0.999))
        else:
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=current_lr, betas=(opt.beta1, 0.999))
        # loss for print
        loss_log = {'ave_loss': 0, 'l_xe_final': 0, 'l_attri_final': 0, 'l_regular_final': 0,
                    'l_xe_layer': 0, 'l_attri_layer': 0, 'l_regular_layer': 0, 'l_cpt': 0}

        num_batches = len(dataloader_train)
        for batch in tqdm(dataloader_train):
            batch_input, batch_target = batch['pixel_values'], batch['class_ids']
            model.zero_grad()
            # map target labels
            input_v = torch.autograd.Variable(batch_input)
            label_v = torch.autograd.Variable(batch_target)
            input_v = input_v.to(device)
            label_v = label_v.to(device)
            output, pre_attri, attention, pre_class = model(input_v, attribute_seen)
            label_a = attribute_seen[:, label_v].t()

            loss = Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model,
                            output, pre_attri, attention, pre_class, label_a, label_v,
                            realtrain, middle_graph, parts, group_dic, sub_group_dic)
            loss_log['ave_loss'] += loss.item()
            loss.backward()
            optimizer.step()
        # print('\nLoss log: {}'.format({key: loss_log[key] / batch for key in loss_log}))
        print('\n[Epoch %d, Batch %5d] Train loss: %.3f '
                % (epoch+1, num_batches, loss_log['ave_loss'] / num_batches))
        val_epoch(model=model, attribute_seen=attribute_seen, acc_fn=compute_corrects, dataloader=dataloader_val,
                  dataset_size=len(dataset_val), batch_size=opt.batch_size,
                  device=device, epoch=epoch, logger=logger)

    logger.info('DONE!')


if __name__ == '__main__':
    main()