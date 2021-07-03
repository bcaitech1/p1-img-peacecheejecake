from ..utils import *
from ..data import *
from ..config import ConfigBranch
from ..modules import models, loss, functional as func
from . import lr_scheduler

import math
import json
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adamp import AdamP, SGDP
from madgrad import MADGRAD

import wandb

from typing import Iterable



##########################################
# TRAINER ################################
##########################################


class Trainer:
    def __init__(self, config: ConfigBranch):
        self.config = config
        self._setup()
 

    def __call__(self):
        self._setup()
        return self
        

    def _setup(self):
        self.device = self.config.system.device

        model, arg, kwarg = models.MODEL_DICT[self.config.model.arc]
        self.model = model(*arg, **kwarg).to(self.device)
        if self.config.teacher:
            self.teacher = self.config.teacher.to(self.device)
        else:
            self.teacher = None
        if self.config.model.state_path:
            self.load_state_dict_to_model()
            self.initial_state_dict = self.config.model.state_path
        else:
            self.initial_state_dict = self.model.state_dict()

        self.lr = self.config.train.lr.base
        self._update_optimizer()
        if self.config.train.lr.scheduler:
            self.scheduler = lr_scheduler.DICT[self.config.train.lr.scheduler]
        elif self.config.train.lr.scheduler_manual.active:
            self.scheduler = self.config.train.lr.scheduler
        else:
            self.scheduler = None
        
        if self.config.train.criterion in loss.torch_loss_dict:
            self.criterion = loss.torch_loss_dict[self.config.train.criterion]()
        elif self.config.train.criterion == 'ArcFaceLoss':
            out_layer_in_features = list(self.model.parameters())[-2].size(1)
            out_layer_out_features = list(self.model.parameters())[-2].size(0)
            self.criterion = loss.AngularPenaltySMLoss(out_layer_in_features, out_layer_out_features)
        elif self.config.train.criterion == 'FocalLoss':
            pass
        elif self.config.train.criterion == 'LabelSmoothingLoss':
            pass
        elif self.config.train.criterion == 'F1Loss':
            pass
        else:
            raise ValueError
        
        self.logger = self.config.train.logger
        self.log_files = [os.path.join(self.config.path.logs, self.model.name + '_train.csv'),
                          os.path.join(self.config.path.logs, self.model.name + '_valid.csv')]

        valid_log_epochs = [logs[0] for logs in self.logger.valid]
        valid_log_state_dicts = [logs[-1] for logs in self.logger.valid]

        if self.config.model.state_path in valid_log_state_dicts:
            start_epoch = valid_log_epochs[valid_log_state_dicts.index(self.config.model.state_path)]
            self.logger.recover(start_epoch)
            self.logger.save_to_csv()
            self.epochs = start_epoch
        else:
            self.epochs = self.logger(self.log_files)

        self._load_data_loader()

    
    def _update_optimizer(self):
        optim_name = self.config.train.optimizer.name.lower()

        if optim_name == 'adam':
            self.optimizer = optim.Adam(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                weight_decay=self.config.train.weight_decay
            )
        elif optim_name == 'sgd':
            self.optimizer = optim.SGD(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                momentum=self.config.train.momentum,
                weight_decay=self.config.train.weight_decay,
                nesterov=self.config.train.nesterov
            )
        elif optim_name == 'adamp':
            self.optimizer = AdamP(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                betas=self.config.train.betas, 
                weight_decay=self.config.train.weight_decay
            )
        elif optim_name == 'sgdp':
            self.optimizer = SGDP(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                weight_decay=self.config.train.weight_decay,
                momentum=self.config.train.momentum,
                nesterov=self.config.train.nesterov
            )
        elif optim_name == 'madgrad':
            self.optimizer = MADGRAD(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                weight_decay=self.config.train.weight_decay,
                momentum=self.config.train.momentum
            )
        else:
            raise NameError("Register proper optimizer if needed")



    def _data_loader(self, dataset, shuffle):
        return DataLoader(
            dataset, 
            batch_size=self.config.data.batch_size,
            shuffle=shuffle, 
            num_workers=self.config.system.num_workers, 
            pin_memory=True
        )


    def _load_data_loader(self):
        all_train_data = glob(os.path.join(self.config.path.train, '*.jpg'))
        all_test_data = glob(os.path.join(self.config.path.test, '*.jpg'))
        if self.config.path.valid:
            all_valid_data = glob(os.path.join(self.config.path.valid, '*.jpg'))

        if self.config.data.upscaling and self.config.data.train_cutmix:
            raise ValueError

        if self.config.path.valid:
            if self.config.data.train_cutmix == 'random':
                if self.config.model.classifying =='all':
                    train_set = RandomCutMixTrainDataset(
                        data=all_train_data,
                        kernel_size=256,
                        num_classes=18
                    )
                elif self.config.model.classifying =='age':
                    train_set = RandomCutMixTrainDataset(
                        data=all_train_data,
                        kernel_size=256,
                        num_classes=3
                    )
            elif self.config.data.train_cutmix == 'center':
                if self.config.model.classifying =='all':
                    train_set = CenterCutMixTrainDataset(
                        data=all_train_data,
                        kernel_size=256,
                        num_classes=18
                    )
                elif self.config.model.classifying =='age':
                    train_set = CenterCutMixTrainDataset(
                        data=all_train_data,
                        kernel_size=256,
                        num_classes=3
                    )
            elif self.config.data.upscaling:
                if self.config.model.classifying =='all':
                    train_set = UpscaledDataset(
                        data=all_train_data
                    )
                elif self.config.model.classifying =='age':
                    train_set = UpscaledAgeDataset(
                        data=all_train_data
                    )
            else:
                if self.config.model.classifying =='all':
                    train_set = BasicDataset(
                        data=all_train_data,
                        labeled=True
                    )
                elif self.config.model.classifying == 'age':
                    train_set = AgeDataset(
                        data=all_train_data,
                        labeled=True
                    )
            valid_set = BasicDataset(
                data=all_valid_data,
                labeled=True
            )
            test_set = BasicDataset(
                data=all_test_data,
                labeled=False
            )

            self.num_classes = train_set.num_classes

                
        else:
            if self.config.data.upscaling:
                if self.config.model.classifying == 'all':
                    train_data, valid_data = train_valid_raw_split(all_train_data)
                    train_set = UpscaledDataset(
                        train_data,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize,
                        augments=self.config.data.custom_augment
                    )
                    valid_set = BasicDataset(
                        valid_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = BasicDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                elif self.config.model.classifying == 'age':
                    train_data, valid_data = train_valid_raw_split(all_train_data)
                    train_set = UpscaledAgeDataset(
                        train_data,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize,
                        augments=self.config.data.custom_augment
                    )
                    valid_set = AgeDataset(
                        valid_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = AgeDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                elif self.config.model.classifying == 'gender':
                    train_data, valid_data = train_valid_raw_split(all_train_data)
                    train_set = UpscaledGenderDataset(
                        train_data,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize,
                        augments=self.config.data.custom_augment
                    )
                    valid_set = GenderDataset(
                        valid_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = GenderDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                elif self.config.model.classifying == 'mask':
                    train_data, valid_data = train_valid_raw_split(all_train_data)
                    train_set = UpscaledMaskDataset(
                        train_data,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize,
                        augments=self.config.data.custom_augment
                    )
                    valid_set = MaskDataset(
                        valid_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = MaskDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                else:
                    raise NameError("Classifying mode should be one of (all, age, gender, mask).")

                self.num_classes = train_set.num_classes

            elif self.config.data.train_cutmix:
                if self.config.model.classifying == 'all':
                    train_data, valid_data = train_valid_raw_split(all_train_data)
                    if self.config.data.train_cutmix == 'random':
                        train_set = RandomCutMixTrainDataset(
                            train_data,
                            kernel_size=256,
                            num_classes=18
                        )
                    elif self.config.data.train_cutmix == 'center':
                        train_set = CenterCutMixTrainDataset(
                            train_data,
                            kernel_size=256,
                            num_classes=18
                        )
                    else:
                        raise NameError

                    valid_set = BasicDataset(
                        valid_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = BasicDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                elif self.config.model.classifying == 'age':
                    train_data, valid_data = train_valid_raw_split(all_train_data)
                    if self.config.data.train_cutmix == 'random':
                        train_set = RandomCutMixTrainDataset(
                            train_data,
                            kernel_size=256,
                            num_classes=3
                        )
                    elif self.config.data.train_cutmix == 'center':
                        train_set = CenterCutMixTrainDataset(
                            train_data,
                            kernel_size=256,
                            num_classes=3
                        )
                    else:
                        raise NameError
                        
                    valid_set = AgeDataset(
                        valid_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = AgeDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                elif self.config.model.classifying == 'gender':
                    train_data, valid_data = train_valid_raw_split(all_train_data)
                    if self.config.data.train_cutmix == 'random':
                        train_set = RandomCutMixTrainDataset(
                            train_data,
                            kernel_size=256,
                            num_classes=3
                        )
                    elif self.config.data.train_cutmix == 'center':
                        train_set = CenterCutMixTrainDataset(
                            train_data,
                            kernel_size=256,
                            num_classes=3
                        )
                    else:
                        raise NameError
    
                    valid_set = GenderDataset(
                        valid_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = GenderDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                elif self.config.model.classifying == 'mask':
                    train_data, valid_data = train_valid_raw_split(all_train_data)
                    if self.config.data.train_cutmix == 'random':
                        train_set = RandomCutMixTrainDataset(
                            train_data,
                            kernel_size=256,
                            num_classes=3
                        )
                    elif self.config.data.train_cutmix == 'center':
                        train_set = CenterCutMixTrainDataset(
                            train_data,
                            kernel_size=256,
                            num_classes=3
                        )
                    else:
                        raise NameError
    
                    valid_set = MaskDataset(
                        valid_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = MaskDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                else:
                    raise NameError("Classifying mode should be one of (all, age, gender, mask).")

                self.num_classes = train_set.num_classes

            else:
                if self.config.model.classifying == 'all':
                    train_set_before_split = BasicDataset(
                        all_train_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = BasicDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                elif self.config.model.classifying == 'age':
                    if self.config.data.soft_age:
                        train_set_before_split = NormalAgeDataset(
                            all_train_data,
                            upscaled=self.config.data.upscaling
                        )
                        test_set = AgeDataset(
                            all_test_data,
                            labeled=False, 
                            preprocess=False,
                            crop_size=False,
                            resize=False
                        )
                    else:
                        train_set_before_split = AgeDataset(
                            all_train_data,
                            labeled=True,
                            preprocess=self.config.data.preprocess,
                            crop_size=self.config.data.crop_size,
                            resize=self.config.data.resize
                        )
                        test_set = AgeDataset(
                            all_test_data,
                            labeled=False, 
                            preprocess=self.config.data.preprocess,
                            crop_size=self.config.data.crop_size,
                            resize=self.config.data.resize
                        )
                elif self.config.model.classifying == 'gender':
                    train_set_before_split = GenderDataset(
                        all_train_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = GenderDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                elif self.config.model.classifying == 'mask':
                    train_set_before_split = MaskDataset(
                        all_train_data,
                        labeled=True,
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                    test_set = MaskDataset(
                        all_test_data,
                        labeled=False, 
                        preprocess=self.config.data.preprocess,
                        crop_size=self.config.data.crop_size,
                        resize=self.config.data.resize
                    )
                else:
                    raise NameError("Classifying mode should be one of (all, age, gender, mask).")
                
                self.num_classes = train_set_before_split.num_classes

                train_set, valid_set = train_valid_split(
                    train_set_before_split,
                    valid_ratio=self.config.data.valid_ratio,
                    shuffle=False
                )

        
        self.train_loader = self._data_loader(train_set, True)
        self.valid_loader = self._data_loader(valid_set, False)
        self.test_loader = self._data_loader(test_set, False)


    def load_state_dict_to_model(self):
        if self.config.model.state_path:
            if os.path.exists(self.config.model.state_path):
                self.model.load_state_dict(torch.load(
                    self.config.model.state_path,
                    map_location=self.config.system.device
                ))
                print(f"Loaded state dict to model.")
            else:
                print(f"WARNING: failed to load state dict - could not find the path.")


    def train_and_save(self):
        print(f"[INFO]")
        print(f"model={self.model.name}")
        print(f"device={self.device}({torch.cuda.get_device_name(self.device)})")
        print(f"data size=({len(self.train_loader.dataset)} + {len(self.valid_loader.dataset)}), \
batch size={self.config.data.batch_size}")
        if self.config.data.upscaling:
            print(f"upaugs={list(map(lambda x: x.__class__.__name__, self.train_loader.dataset.augments))}")
        print(f"optimizer{self.config.train.optimizer}")
        print(f"epochs={self.config.train.num_epochs}, lr={self.lr}, weight_decay={self.config.train.weight_decay}", end="")
        if self.config.train.optimizer.name.lower() in ('adamp'):
            print(f", betas={self.config.train.betas}")
        elif self.config.train.optimizer.name.lower() in ('sgd', 'sgdp'):
            print(f", momentum={self.config.train.momentum}, nesterov={self.config.train.nesterov}")
        else:
            print()

        print()
        print("Start of traning.")
        
        
        def save_with_name(postfix):
            # config_mode="json" with function to convert ConfigTree to JSON
            try:
                # print(self.model.name) # debug
                saving_path = self.save(self.model.name, config_mode=None, postfix=postfix)
            except AttributeError:
                print("WARNING: model has no name.")
                saving_path = self.save(config_mode=None, postfix=postfix)
            
            return saving_path


        saving_postfix = "upscaled" if self.config.data.upscaling else ""
        saving_path = None

        if self.epochs > self.config.train.valid_min:
            _, last_valid_acc = self.valid()
        else:
            last_valid_acc = self.logger.valid[-1][2] if self.logger.valid else 0


        self.model = nn.DataParallel(self.model)

        for _ in range(self.config.train.num_epochs):
            # early-stop if learning rate is under the lower limit
            if self.scheduler and self.config.lr.scheduler \
            and self.lr < self.config.train.lr.low_limit:
                print(f">>>>> Learning rate: under the lower limit.")
                self.lr *= self.scheduler.divider
                break

            if self.config.train.valid_min == 0:
                last_valid_loss, last_valid_acc = self.valid()

            self.epochs += 1
            self.train_one_epoch()
            
            # naive k-fold validation
            if self.config.train.shuffle_period > 0 \
            and self.epochs % self.config.train.shuffle_period == 0:
                try:
                    if not saved:
                        save_with_name(saving_postfix)
                except NameError:
                    save_with_name(saving_postfix)
                print(">>>>> SAVED CHECKPOINT. RESTART TRAINING.")
                self.train_loader, self.valid_loader = self._train_valid_shuffle()
                self.model.load_state_dict(self.initial_state_dict)
                continue

            valided = False
            saved   = False
            tested  = False
            plotted = False

            # valid
            if self.config.train.valid_period > 0 \
            and self.epochs >= self.config.train.valid_min \
            and self.epochs % self.config.train.valid_period == 0:
                last_valid_loss, last_valid_acc = self.valid()
                valided = True
            
            # save
            if self.config.train.save_period > 0 \
            and self.epochs >= self.config.train.save_min \
            and self.epochs % self.config.train.save_period == 0 \
            and last_valid_acc >= self.config.train.save_min_acc:
                saving_path = save_with_name(saving_postfix)
                self.logger.valid[-1][-1] = saving_path
                saved = True
                 
            # test
            if self.config.train.test_period > 0 \
            and self.epochs >= self.config.train.test_min \
            and self.epochs % self.config.train.test_period == 0 \
            and last_valid_acc >= self.config.train.test_min_acc:
                self.infer_test_and_save()
                tested = True

            # plot loss & acc
            if self.config.train.plot_period > 0 \
            and self.config.train.plot_period > 0 \
            and self.epochs % self.config.train.plot_period == 0 \
            and last_valid_acc >= self.config.train.plot_min_acc:
                self.logger.plot()
                plotted = True
            
            # log & learning rate
            if valided:
                # log valid (to include saved model - if saved state dict)
                self.logger.log_last_valid_to_csv()

        if not valided:
            self.valid()
        if not saved:
            saving_path = save_with_name(saving_postfix)
            self.logger.valid[-1][-1] = saving_path
        if not tested and last_valid_acc >= self.config.train.test_min_acc:
            self.infer_test_and_save()
        if not plotted and self.epochs > 1 and last_valid_acc >= self.config.train.plot_min_acc:
            self.logger.plot()

        print()
        print(f"End of training. (end_lr={self.lr})")
        print()

        return saving_path

        
    def train_one_epoch(self, add_loader=None):
        print(f"[Epoch {self.epochs:03d}]", end="")
        
        self.model.train()
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        starter.record()
        
        data_loader = add_loader if add_loader else self.train_loader
        epoch_time = train_loss = 0
        
        total = 0
        correct = 0
        confusion_matrix = func.ConfusionMatrix(self.num_classes)
        
        for batch, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            targets = targets.to(self.device)

            if self.teacher:
                teacher_outputs = self.teacher(inputs)
                loss = self.criterion(outputs, teacher_outputs, targets)
            else:
                loss = self.criterion(outputs, targets)

            loss.backward()

            self.optimizer.step()
            if self.config.train.lr.scheduler:
                self.scheduler.step(loss)
            elif self.scheduler:
                few = self.scheduler.averaging_few
                last_few_losses = [logs[1] for logs in self.logger.valid[-few:]]
                last_last_few_losses = [logs[1] for logs in self.logger.valid[-few-1:-1]]
                if last_last_few_losses and  np.mean(last_last_few_losses) < np.mean(last_few_losses):
                    old_lr = self.lr
                    self.lr /= self.scheduler.divider
                    self.config.train.lr.base = self.lr
                    print(f">>>>> Learning rate: {old_lr} -> {self.lr}")


            _, predictions = torch.max(outputs, dim=1)  #
            if self.config.model.classifying == 'age' and self.config.data.soft_age:
                _, targets = torch.max(targets, dim=1)
            
            predictions = predictions.detach()
            targets = targets.detach()
            
            train_loss += loss.item()
            total += targets.shape[0]
            correct += predictions.eq(targets).sum().item()
            confusion_matrix(targets, predictions)

            ender.record()
            torch.cuda.synchronize()
            
            batch_time = starter.elapsed_time(ender) - epoch_time
            epoch_time += batch_time
            
            print(f'\r[Epoch {self.epochs:03d}] (Batch #{batch:03d})  \
Loss: {train_loss / (batch + 1):.5f},  Acc: {correct / total * 100:.3f}  \
({time_str(epoch_time)})', end='')


        train_loss /= (batch + 1)
        accuracy  = correct / total
        f1_score = confusion_matrix.f1_score()

        self.logger.log_train(self.epochs, train_loss, accuracy)
        self.logger.log_last_train_to_csv()
    
        print(f'\r[Epoch {self.epochs:03d}]  \
Loss: {train_loss:.5f},  Acc: {accuracy * 100:.3f},  F1 Score: {f1_score:.5f}  \
({time_str(epoch_time)})')

        
    def valid(self, log: bool=True):
        valid_num = (self.epochs - self.config.train.valid_min + 1)\
         // self.config.train.valid_period if self.epochs > 0 else self.epochs
        print(f"[Valid {valid_num:03d}] ", end=" ")
        
        self.model.eval()
        
        valid_loss = 0
        correct = total = 0
        confusion_matrix = func.ConfusionMatrix(self.num_classes)

        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(self.valid_loader):
                if self.config.train.valid_iters > 0 and batch >= self.config.train.valid_iters:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                if self.teacher:
                    teacher_outputs = self.teacher(inputs)
                    loss = self.criterion(outputs, teacher_outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)
                
                valid_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                if self.config.model.classifying == 'age' and self.config.data.soft_age:
                    _, targets = torch.max(targets, dim=1)

                predictions = predictions.detach() 
                targets = targets.detach()

                assert torch.all(predictions < self.num_classes), \
                    f"{torch.sum(predictions >= self.num_classes).item()}/{self.num_classes}: out of bound"

                total += targets.shape[0]
                correct += predictions.eq(targets).sum().item()
                confusion_matrix(targets, predictions)

                print(f'\r[Valid {valid_num:03d}] (Batch #{batch:03d})  \
Loss: {valid_loss / (batch + 1):.5f},  Acc: {correct / total * 100:.3f}', end='')


        valid_loss /= batch + 1
        accuracy = correct / total
        f1_score = confusion_matrix.f1_score()
        
        if log:
            self.logger.log_valid(self.epochs, valid_loss, accuracy, None)

        print(f'\r[Valid {valid_num:03d}]  \
Loss: {valid_loss:.5f},  Acc: {accuracy * 100:.3f},  F1 Score: {f1_score:.5f}')

        return valid_loss, accuracy
        
        
    def save(self, name="", postfix=None, config_mode=None):
        file_name = f'{filename_from_datetime(datetime.today())}_{name}_{postfix if postfix else ""}'
        
        # save model parameter
        model_output = os.path.join(self.config.path.models, file_name + ".pth")
        torch.save(self.model.state_dict(), model_output)
        print(f"Saved model: {model_output}")

        # save configurations
        if config_mode == "json":
            config_output = os.path.join(self.config.path.configs, file_name + ".json")
            with open(config_output, "w") as f:
                json.dump(self.config, f)

        return model_output


    def infer(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()

        self.model.eval()
        indices = [path.split('/')[-1] for path in glob(os.path.join(self.config.path.test, '*.jpg'))]
        result = pd.DataFrame(columns=['ans'], index=indices)
        result.index.name = 'ImageID'

        total_batch_num = len(self.test_loader.dataset) / self.config.data.batch_size
        total_batch_num = math.ceil(total_batch_num)
        
        with torch.no_grad():
            for i, (inputs, filenames) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                filenames = np.array(filenames)

                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1).detach()
                assert torch.all(predictions < self.num_classes), \
                    f"{torch.sum(predictions >= self.num_classes).item()}/{self.num_classes}: out of bound"

                result['ans'][filenames] = predictions.cpu().numpy()

                ender.record()
                torch.cuda.synchronize()
                infer_time = starter.elapsed_time(ender)
                
                print(f"\rEvaluating: {i}/{total_batch_num} ({time_str(infer_time)})", end="")

        print(f"\r{self.model.name}: End of evaluation ({time_str(infer_time)})")

        info_file = pd.read_csv(
            os.path.join(self.config.path.output, 'info.csv'),
            index_col='ImageID'
        )
        
        return result.loc[info_file.index]

                
    def infer_and_save_csv(self):
        result = self.infer()
        
        csv_name = f"{self.model.name}_{filename_from_datetime(datetime.today())}.csv"
        csv_path = os.path.join(self.config.path.output, csv_name)

        result.to_csv(csv_path)
        
        print(f"\rSaved result: {csv_path}")

        return csv_path


    def logits(self):
        test_size = len(self.test_loader.dataset)
        num_classes = self.num_classes
        batch_size = self.config.data.batch_size

        logits = torch.zeros((test_size, num_classes)).to(self.device)

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                outputs = F.log_softmax(outputs, dim=1)

                logits[batch_idx * batch_size: (batch_idx + 1) * batch_size] += outputs

        return logits


    def infer_with_simple_tta(self, augments: Iterable = None):
        if not augments: augments = SimpleTTA.augments

        self.model.eval()

        test_size = len(self.test_loader.dataset)
        batch_size = self.config.data.batch_size
        total_batch_num = math.ceil(test_size / batch_size)
        
        with torch.no_grad():
            logits_all = torch.zeros((test_size, self.num_classes), dtype=torch.float).to(self.device)
            for aug_idx, augment in enumerate(augments):
                test_loader = self._data_loader(
                    dataset=SimpleTTADataset(self.test_loader.dataset.data, augment),
                    shuffle=False
                )

                logits = torch.zeros((test_size, self.num_classes), dtype=torch.float).to(self.device)
                for batch_idx, (inputs, _) in enumerate(test_loader):
                    inputs = inputs.to(self.device)
                    outputs = F.log_softmax(self.model(inputs), dim=1)
                    logits[batch_idx * batch_size: (batch_idx + 1) * batch_size] = outputs
                    
                    print(f"\rEvaluating #{aug_idx + 1}/{len(augments)} ({batch_idx + 1}/{total_batch_num})", end="")

                logits_all += logits.detach()

        print()
        print(f"End of evaluation.")


        indices = [path.split('/')[-1] for path in self.test_loader.dataset.data]
        result = pd.DataFrame(columns=['ans'], index=indices)
        result.index.name = 'ImageID'
        result.ans = torch.argmax(logits_all, dim=1).cpu().numpy()

        info_file = pd.read_csv(
            os.path.join(self.config.path.output, 'info.csv'),
            index_col='ImageID'
        )
        result = result.loc[info_file.index]
        
        return result



def ensemble_and_infer_test(
    trainer: Trainer, 
    models: Iterable[models.BasicModel], 
    weighted: bool=False
):
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    starter.record()
    
    test_data = trainer.test_loader.dataset.data
    test_size = len(test_data)
    num_classes = trainer.test_loader.dataset.num_classes
    
    config = trainer.config
    device = config.system.device
    test_path = config.path.test
    save_path = config.path.output

    model_in_trainer = models.DICT[config.model.arc]

    logits = torch.zeros((test_size, num_classes)).to(device)
    
    
    for model_idx, model in enumerate(models):
        if model_idx > 0:
            print(f"\rEvaluating: {model_idx}/{len(models)} ({time_str(elapsed_time)})", end="")
        else:
            print(f"\rEvaluating: {model_idx}/{len(models)}", end="")

        config.model.arc = model.__class__.__name__
        logits += trainer.logits()

        ender.record()
        torch.cuda.synchronize()
        elapsed_time = starter.elapsed_time(ender)

    predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()

    filenames = [path.split('/')[-1] for path in test_data]
    result = pd.DataFrame(index=filenames, columns=['ans'])
    result.index.name = 'ImageID'
    result.ans = predictions
    
    info = pd.read_csv(os.path.join(test_path, '..', 'info.csv'), index_col='ImageID')
    result = result.loc[info.index]

    weighted_symbol = 'w' if weighted else 'x'
    csv_name = f"ensemble_{len(models)}{weighted_symbol}_{filename_from_datetime(datetime.today())}.csv"
    csv_path = os.path.join(save_path, csv_name)
        
    result.to_csv(csv_path)

    print(f"\rSaved result: {csv_path}")

    config.model.arc = model_in_trainer.__class__.__name__
