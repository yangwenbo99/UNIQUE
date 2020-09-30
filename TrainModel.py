import os
import time
import scipy.stats
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn

from ImageDataset import ImageDataset
from dataset import AutomatedDataset

from BaseCNN import BaseCNN
from DBCNN import DBCNN
from MNL_Loss import Fidelity_Loss
from lfc_cnn import E2EUIQA
from utils import functor
from typing import Dict

#from E2euiqa import E2EUIQA
#from MNL_Loss import L2_Loss, Binary_Loss
#from Gdn import Gdn2d, Gdn1d

#! Rows to delete
#< Rows to notice
#+ Rows with important comments for adversarial training

from Transformers import AdaptiveResize

from lipschitz import NetworkLipschitzEnforcer


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)

        self.config = config

        if not config.network.startswith('lfc') or config.force_normalization:
            self.train_transform = transforms.Compose([
                #transforms.RandomRotation(3),
                AdaptiveResize(512),
                transforms.RandomCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
            self.test_transform = transforms.Compose([
                AdaptiveResize(768),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
        else:
            #< Note that the original transform of LFC is different
            self.train_transform = transforms.Compose([
                transforms.RandomRotation(3),
                transforms.RandomCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            self.test_transform = transforms.Compose([
                AdaptiveResize(768),
                transforms.ToTensor()
            ])



        self.train_batch_size = config.batch_size
        self.test_batch_size = 1

        self.ranking = config.ranking

        def get_filelist_name(dset, txt):
            return os.path.join(dset, 'splits2', str(config.split), txt)

        self.train_loader = self._build_automated_dataset(
                'train',
                get_filelist_name(config.trainset, config.train_txt),
                config.trainset,
                test=(not config.ranking),
                batch_size=self.train_batch_size,
                transform=self.train_transform,
                shuffle=True,
                num_workers=12
                )

        self.test_loaders = [
                self._build_automated_dataset(
                    'live',
                    get_filelist_name(config.live_set, 'live_test.txt'),
                    config.live_set),
                self._build_automated_dataset(
                    'csiq',
                    get_filelist_name(config.csiq_set, 'csiq_test.txt'),
                    config.csiq_set),
                #<self._build_automated_dataset(
                #<    'tid2013',
                #<    get_filelist_name(config.tid2013_set, 'tid_test.txt'),
                #<    config.tid2013_set),
                self._build_automated_dataset(
                    'kadid10k',
                    get_filelist_name(config.kadid10k_set, 'kadid10k_test.txt'),
                    config.kadid10k_set),
                self._build_automated_dataset(
                    'bid',
                    get_filelist_name(config.bid_set, 'bid_test.txt'),
                    config.bid_set),
                self._build_automated_dataset(
                    'clive',
                    get_filelist_name(config.clive_set, 'clive_test.txt'),
                    config.clive_set),
                self._build_automated_dataset(
                    'koniq10k',
                    get_filelist_name(config.koniq10k_set, 'koniq10k_test.txt'),
                    config.koniq10k_set),
                ]
        #! Copy-and-paste based programming eliminated!

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")

        # initialize the model
        if config.network == 'basecnn':
            self.model = BaseCNN(config)
            self.model = nn.DataParallel(self.model, device_ids=[0])
        elif config.network == 'dbcnn':
            self.model = DBCNN(config)
            self.model = nn.DataParallel(self.model).cuda()
        elif config.network.startswith('lfc'):
            self.model = E2EUIQA(config)
            # self.model = nn.DataParallel(self.model).cuda()
        else:
            raise NotImplementedError("Not supported network, need to be added!")
        self.model.to(self.device)
        self.model_name = type(self.model).__name__
        if self.config.verbose:
            print(self.model)

        # loss function
        if config.ranking:
            if config.fidelity:
                self.loss_fn = Fidelity_Loss()
                # self.loss_fn_state = FunctorWrap(lambda state: self.loss_fn(state.p, state.g.detach()))
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
                # self.loss_fn_state = FunctorWrap(lambda state: FunctorWrap(lambda state: self.loss_fn(state.p, state.g.detach())))
        else:
            self.loss_fn = nn.MSELoss()
        self.loss_fn.to(self.device)

        if self.config.std_modeling:
            self.std_loss_fn = nn.MarginRankingLoss(margin=self.config.margin)
            self.std_loss_fn.to(self.device)

        if self.config.network == 'lfc':
            self.network_parameter_correct = functor.FunctorWrap(
                    lambda i, model: model.gdn_param_proc())
            # i is the iteration number and model is the model
        else:
            self.network_parameter_correct = functor.Functor()
        if self.config.lipschitz:
            self.lipschitz_enforcer = NetworkLipschitzEnforcer.default_enforcer(self.model)
            def enforce_lip(i, model):
                if i < 0 or i % 20 == 0:
                    self.lipschitz_enforcer.correct()
            self.network_parameter_correct = \
                    self.network_parameter_correct.then(enforce_lip)


        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        #self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr': lr},
        #                             {'params': self.sensitivity, 'lr': 1e-3},
        #                             {'params': self.specificity, 'lr': 1e-3}]
        #                            )
        #self.optimizer = optim.Adam([{'params': self.model.backbone.parameters(), 'lr': lr},
        #                             {'params': self.model.fc.parameters(), 'lr': 10 * lr}],
        #                            lr=lr, weight_decay=5e-4
        #                            )
        #self.optimizer = optim.SGD([{'params': self.model.backbone.parameters(), 'lr': lr},
        #                             {'params': self.model.fc.parameters(), 'lr': lr}],
        #                            lr=lr, weight_decay=5e-4, momentum=0.9
        #                            )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=5e-4)

        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.train_std_loss = []
        self.test_results_srcc = {
                dsname: [] for dsname in config.to_test}
        self.test_results_plcc = {
                dsname: [] for dsname in config.to_test}
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)

    def _gen_automated_dataset(self):
        pass

    def _build_automated_dataset(
            self, name, file_list, img_dir,
            test=True,
            transform=None,
            batch_size=None,
            num_workers=1,
            shuffle=False):
        if transform is None:
            transform = self.test_transform
        if batch_size is None:
            batch_size = self.test_batch_size

        data = ImageDataset(
                csv_file=file_list,
                img_dir=img_dir,
                transform=transform,
                test=test)

        loader = DataLoader(data,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=num_workers)

        automated_dataset = AutomatedDataset(name, loader)
        return automated_dataset


    def fit(self):
        if self.ranking:
            for epoch in range(self.start_epoch, self.max_epochs):
                    _ = self._train_single_epoch(epoch)
                    self.scheduler.step()
        else:
            for epoch in range(self.start_epoch, self.max_epochs):
                    _ = self._train_single_epoch_regression(epoch)
                    self.scheduler.step()

    class TrainSingleEpochState:
        """
        x1, x2, g, gstd1, gstd2, yb
        y1_var, y2_var
        p
        """
        def __init__(self, x1, x2, g, gstd1, gstd2, yb):
            self.x1 = x1
            self.x2 = x2
            self.g  = g
            self.gstd1 = gstd1
            self.gstd2 = gstd2
            self.yb = yb


    class TrainSingleEpoch:
        """
        For std_modeling:
            one need to determine p:
                by whether split modeling
            Then, determine std_loss
        Otherwise, p shall be determined by y1 - y2

        The loss is consist of two component , the first part is by fidelyty,
        and the second part is by std_loss
        """
        pass

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        running_std_loss = 0 if epoch == 0 else self.train_std_loss[-1]
        loss_corrected = 0.0
        std_loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        #self.scheduler.step()
        for step, sample_batched in enumerate(self.train_loader, 0):

            if step < self.start_step:
                continue

            x1, x2, g, gstd1, gstd2, yb = sample_batched['I1'], sample_batched['I2'], sample_batched['y'], sample_batched['std1'], sample_batched['std2'], sample_batched['yb']
            state = self.TrainSingleEpochState(
                    x1=x1.to(self.device),
                    x2=x2.to(self.device),
                    g = g.view(-1, 1).to(self.device),
                    yb = yb.view(-1, 1).to(self.device),
                    gstd1 = gstd1.to(self.device),
                    gstd2 = gstd2.to(self.device))

            self.optimizer.zero_grad()
            if self.config.std_modeling:
                state.y1, state.y1_var = self.model(state.x1)
                state.y2, state.y2_var = self.model(state.x2)
                state.y_diff = state.y1 - state.y2

                if self.config.split_modeling:
                    state.p = torch.sigmoid(state.y_diff)
                else:
                    #+ y_var = y1_var * y1_var + y2_var * y2_var + 1e-8
                    #+ Paul modified here, for adversarial training
                    state.y_var = state.y1_var * state.y1_var + state.y2_var * state.y2_var + 1e-8
                    #+ eps = 2 for checkpoint 1
                    state.p = 0.5 * (1 + torch.erf(state.y_diff / torch.sqrt(2 * state.y_var.detach())))
                    #!print('p.shape', p.shape)
                    if state.p.isnan().any():
                        print('!!! NaN in p')

                state.std_label = torch.sign((gstd1 - gstd2))
                if self.config.fixvar:
                    self.std_loss = 0
                else:
                    self.std_loss = self.std_loss_fn(state.y1_var, state.y2_var, state.std_label.detach())
                #!print('vars:', y1_var, y2_var)#!
            else:
                state.y1 = self.model(state.x1)
                state.y2 = self.model(state.x2)
                state.y_diff = state.y1 - state.y2
                state.p = state.y_diff

            if self.config.fidelity:
                self.loss = self.loss_fn(state.p, state.g.detach())
            else:
                self.loss = self.loss_fn(state.p, state.yb.detach())
            if self.config.std_loss:
                #!print(self.loss, self.std_loss)  #!
                self.loss += self.std_loss
                #! self.loss = self.std_loss #!

            if self.loss.isnan().any():
                print('!!! NaN in loss')
            self.loss.backward()
            self.optimizer.step()
            self.network_parameter_correct(step, self.model)

            # statistics
            with torch.no_grad(): #+
                running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
                loss_corrected = running_loss / (1 - beta ** local_counter)

                if self.config.std_loss and not self.config.fixvar:
                    running_std_loss = beta * running_std_loss + (1 - beta) * self.std_loss.data.item()
                    std_loss_corrected = running_std_loss / (1 - beta ** local_counter)
                else:
                    std_loss_corrected = 0

                current_time = time.time()
                duration = current_time - start_time
                running_duration = beta * running_duration + (1 - beta) * duration
                duration_corrected = running_duration / (1 - beta ** local_counter)
                examples_per_sec = self.train_batch_size / duration_corrected
                format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] [Std Loss = %.4f] (%.1f samples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected, std_loss_corrected,
                                    examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.network_parameter_correct(-1, self.model)

        self.train_loss.append(loss_corrected)
        self.train_std_loss.append(std_loss_corrected)

        #if (epoch+1) % self.epochs_per_eval == 0:
        if (not self.config.fc) & ((epoch+1) % self.epochs_per_eval == 0):
            self.eval_and_record()

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'train_std_loss': self.train_std_loss,
                'test_results_srcc': self.test_results_srcc,
                'test_results_plcc': self.test_results_plcc,
            }, model_name)

        return self.loss.data.item()


    def _train_single_epoch_regression(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        #self.scheduler.step()
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        for step, sample_batched in enumerate(self.train_loader, 0):

            if step < self.start_step:
                continue

            x, g = sample_batched['I'], sample_batched['mos']
            g = Variable(g).view(-1, 1)
            x = x.to(self.device)
            g = g.to(self.device)

            self.optimizer.zero_grad()
            y = self.model(x)

            self.loss = self.loss_fn(y, g.float().detach())
            self.loss.backward()
            self.optimizer.step()
            if self.config.network == 'lfc':
                self.model.gdn_param_proc()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)

        #if (epoch+1) % self.epochs_per_eval == 0:
        if (not self.config.fc) & ((epoch+1) % self.epochs_per_eval == 0):
            # evaluate after every other epoch
            self.eval_and_record()

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'train_std_loss': 0,
                'test_results_srcc': self.test_results_srcc,
                'test_results_plcc': self.test_results_plcc,
            }, model_name)

        return self.loss.data.item()

    def eval(self):
        """
        return: two dict named srccs and plccs, whose keys are dtatasets tested
        """
        srccs = {}
        plccs = {}
        self.model.eval()

        if self.config.std_modeling:
            model = lambda x : self.model(x)[0]
        else:
            model = self.model

        for dset in self.test_loaders:
            if dset._name in self.config.to_test:
                srcc, plcc = dset.eval(self.device, model)
                srccs[dset._name] = srcc
                plccs[dset._name] = plcc

        return srccs, plccs

    def eval_and_record(self, save_to_state_dict=True):
        def get_out_str(scores: Dict[str, float], score_name: str):
            l = ['{} {} {:.4f}'.format(name.upper(), score_name, score)
                    for name, score in scores.items()]
            return '  '.join(l)

        test_results_srcc, test_results_plcc = self.eval()
        if save_to_state_dict:
            for dset_name in self.config.to_test:
                self.test_results_srcc[dset_name].append(test_results_srcc[dset_name])
                self.test_results_plcc[dset_name].append(test_results_plcc[dset_name])

        out_str = 'Testing: ' + get_out_str(test_results_srcc, 'SRCC')
        out_str2 = 'Testing: ' + get_out_str(test_results_plcc, 'PLCC')
        print(out_str)
        print(out_str2)



    def get_scores(self):
        all_mos = {}
        all_hat = {}
        all_std = {}
        all_pstd = {}
        self.model.eval()
        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        for step, sample_batched in enumerate(self.live_loader, 0):
            x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, var = self.model(x)
                q_std.append(std.data.numpy())
                q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        all_mos['live'] = q_mos
        all_hat['live'] = q_hat
        all_std['live'] = q_std
        all_pstd['live'] = q_pstd

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        for step, sample_batched in enumerate(self.csiq_loader, 0):
            x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, var = self.model(x)
                q_std.append(std.data.numpy())
                q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        all_mos['csiq'] = q_mos
        all_hat['csiq'] = q_hat
        all_std['csiq'] = q_std
        all_pstd['csiq'] = q_pstd

        '''
        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        for step, sample_batched in enumerate(self.tid2013_loader, 0):
            x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, var = self.model(x)
                q_std.append(std.data.numpy())
                q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        all_mos['tid2013'] = q_mos
        all_hat['tid2013'] = q_hat
        all_std['tid2013'] = q_std
        all_pstd['tid2013'] = q_pstd
        '''

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        for step, sample_batched in enumerate(self.kadid10k_loader, 0):
            x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, var = self.model(x)
                q_std.append(std.data.numpy())
                q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        all_mos['kadid10k'] = q_mos
        all_hat['kadid10k'] = q_hat
        all_std['kadid10k'] = q_std
        all_pstd['kadid10k'] = q_pstd

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        for step, sample_batched in enumerate(self.bid_loader, 0):
            x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, var = self.model(x)
                q_std.append(std.data.numpy())
                q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        all_mos['bid'] = q_mos
        all_hat['bid'] = q_hat
        all_std['bid'] = q_std
        all_pstd['bid'] = q_pstd

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        for step, sample_batched in enumerate(self.clive_loader, 0):
            x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, var = self.model(x)
                q_std.append(std.data.numpy())
                q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        all_mos['clive'] = q_mos
        all_hat['clive'] = q_hat
        all_std['clive'] = q_std
        all_pstd['clive'] = q_pstd

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        for step, sample_batched in enumerate(self.koniq10k_loader, 0):
            x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar, var = self.model(x)
                q_std.append(std.data.numpy())
                q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar = self.model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        all_mos['koniq10k'] = q_mos
        all_hat['koniq10k'] = q_hat
        all_std['koniq10k'] = q_std
        all_pstd['koniq10k'] = q_pstd

        return all_mos, all_hat, all_std, all_pstd

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            if self.config.verbose:
                print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.train_std_loss = checkpoint['train_std_loss']
            self.test_results_srcc = checkpoint['test_results_srcc']
            self.test_results_plcc = checkpoint['test_results_plcc']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            if self.config.verbose:
                print("[*] loaded checkpoint '{}' (epoch {})"
                      .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

