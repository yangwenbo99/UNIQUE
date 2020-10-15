#!/bin/python3

'''
This file is to plot a graph with the following setting.

1. We first select an image x_0
2. We then add some pertubation to the image to get x_1 (its type shall
   configurable in the future, but we set it to be random or loaded from file
   currently)
3. Next, we plot f(x) for all x on the segment x_0 to x_1
4. Finally, we optionally save the pertuabation for future work

Example:
    python plot1.py --train '' --network lfc --ranking True --fidelity True --std_modeling True --std_loss '' --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --fixvar --max_epochs2 12  --batch_size=16 --batch_size2=16 --ckpt_path=checkpoints_many/lfc  -x /data_partition/yang/fyp/adv_1/IQA_database_syn/databaserelease2/jp2k/img4.bmp --pertubation_length 0.01

    python plot1.py --train '' --network lfc --ranking True --fidelity True --std_modeling True --std_loss '' --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --fixvar --max_epochs2 12  --batch_size=16 --batch_size2=16 --ckpt_path=checkpoints_many/lfc_lip  -x /data_partition/yang/fyp/adv_1/IQA_database_syn/databaserelease2/jp2k/img4.bmp --pertubation_length 0.01


    python plot1.py --train '' --network lfc --ranking True --fidelity True --std_modeling True --std_loss '' --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --fixvar --max_epochs2 12  --batch_size=16 --batch_size2=16 --ckpt_path=checkpoints_many/lfc_nom  -x /data_partition/yang/fyp/adv_1/IQA_database_syn/databaserelease2/jp2k/img4.bmp --pertubation_length 0.01  --force_normalization

    python plot1.py --train '' --network lfc_relu --ranking True --fidelity True --std_modeling True --std_loss '' --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --fixvar --max_epochs2 12  --batch_size=16 --batch_size2=16 --ckpt_path=checkpoints_many/lfc_relu_nom  -x /data_partition/yang/fyp/adv_1/IQA_database_syn/databaserelease2/jp2k/img4.bmp --pertubation_length 0.01  --force_normalization

    python plot1.py --train '' --network lfc_relu --ranking True --fidelity True --std_modeling True --std_loss '' --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --fixvar --max_epochs2 12  --batch_size=16 --batch_size2=16 --ckpt_path=checkpoints_many/lfc_relu_nom_lip  -x /data_partition/yang/fyp/adv_1/IQA_database_syn/databaserelease2/jp2k/img4.bmp --pertubation_length 0.01  --force_normalization
'''

import argparse
import TrainModel
import scipy.io as sio
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-x', '--img', type=str, help='the base image')
    parser.add_argument('-p', '--pertubation', type=str, default='',
            help='the pertubation of the image, will be randomly generated if not presented')
    parser.add_argument('--pertubation_length', type=float, default=0.01,
            help='the length of the pertubataion, if random generation is nessesary')
    parser.add_argument('-s', '--save_pertubation', type=str, default='',
            help='whether the pertubation should be saved')

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument('--get_scores', type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    # parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--seed", type=int, default=19901116)

    parser.add_argument("--backbone", type=str, default='resnet34')
    parser.add_argument("--fc", type=bool, default=True)
    parser.add_argument('--scnn_root', type=str, default='saved_weights/scnn.pkl')

    parser.add_argument("--network", type=str, default="basecnn",
            help='basecnn or dbcnn or lfc')

    parser.add_argument("--representation", type=str, default="BCNN")

    parser.add_argument("--ranking", type=bool, default=True,
            help='True for learning-to-rank False for regular regression')

    parser.add_argument("--fidelity", type=bool, default=True,
            help='True for fidelity loss False for regular ranknet with CE loss')

    parser.add_argument("--std_modeling", type=bool,
                        default=True)  # True for modeling std False for not
    parser.add_argument("--std_loss", type=bool, default=True)
    parser.add_argument("--fixvar", action='store_true') #+
    parser.add_argument("--force_normalization", action='store_true')
    parser.add_argument("--lipschitz", action='store_true')
    parser.add_argument("--margin", type=float, default=0.025)

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--trainset", type=str, default="./IQA_database/")
    parser.add_argument("--live_set", type=str, default="./IQA_database/databaserelease2/")
    parser.add_argument("--csiq_set", type=str, default="./IQA_database/CSIQ/")
    parser.add_argument("--tid2013_set", type=str, default="./IQA_database/TID2013/")
    parser.add_argument("--bid_set", type=str, default="./IQA_database/BID/")
    #parser.add_argument("--cid_set", type=str, default="./IQA_database/CID2013_camera/")
    parser.add_argument("--clive_set", type=str, default="./IQA_database/ChallengeDB_release/")
    parser.add_argument("--koniq10k_set", type=str, default="./IQA_database/koniq-10k/")
    parser.add_argument("--kadid10k_set", type=str, default="./IQA_database/kadid10k/")

    parser.add_argument("--eval_live", type=bool, default=True)
    parser.add_argument("--eval_csiq", type=bool, default=True)
    parser.add_argument("--eval_tid2013", type=bool, default=False)
    parser.add_argument("--eval_kadid10k", type=bool, default=True)
    parser.add_argument("--eval_bid", type=bool, default=True)
    parser.add_argument("--eval_clive", type=bool, default=True)
    parser.add_argument("--eval_koniq10k", type=bool, default=True)

    parser.add_argument("--split_modeling", type=bool, default=False)

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument("--train_txt", type=str, default='train.txt') # train.txt | train_synthetic.txt | train_authentic.txt | train_sub2.txt | train_score.txt

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size2", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_epochs2", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=3)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)

    parser.add_argument("--verbose", action='store_true')

    config = parser.parse_args()
    config.to_test = []

    return config


def main(config):
    t = TrainModel.Trainer(config)
    # checking compatability
    if config.fixvar and not config.network.startswith('lfc'):
        raise NotImplementedError()
    if str(config.backbone).startswith('lfc') and not config.std_modeling:
        raise NotImplementedError()


    model = t.model
    pil_img = Image.open(config.img)
    # pil_img = pil_img.reshape((1,) + tuple(pil_img.shape))
    img = t.test_transform(pil_img).to(t.device)

    if config.pertubation:
        with open(config.pertubation, 'rb') as f:
            pertubation = torch.load(f)
    else:
        pertubation = torch.rand(img.shape) * config.pertubation_length
        pertubation = pertubation.to(t.device)

    img = img.unsqueeze(0)
    print(img.shape)

    if config.save_pertubation:
        with open(config.save_pertubation, 'wb') as f:
            torch.save(pertubation, f)

    should_normalize = not config.network.startswith('lfc') or config.force_normalization

    if should_normalize:
        normalization_transform = \
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
        pertubation = normalization_transform(pertubation)

    x = list(np.linspace(0, 1, 100))
    y = [t.predict_single_image(img + p * pertubation).detach().cpu().numpy() for p in x]
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    config = parse_config()
    main(config)
