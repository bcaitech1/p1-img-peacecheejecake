from utils import *
from data import BasicDataset, train_valid_raw_split
from config import ConfigTree

import csv
import json
import pandas as pd

import torch
from torch.utils.data import DataLoader

from adamp import AdamP, SGDP

from torch.optim import __dict__ as torch_optim_dict
from torch.nn.modules.loss import __dict__ as torch_loss_dict


##########################################
# TRAINER ################################
##########################################


class Trainer():
    def __init__(self, config: ConfigTree):
        self.config = config
        self.configure()
        
        self.train_loader, self.valid_loader = self._train_valid_shuffle()

        test_set = BasicDataset(
            glob(os.path.join(self.config.path.test, '*.jpg')), 
            mode='test', 
            preprocess=True, 
            augment=False
        )
        self.test_loader = self._data_loader(test_set, False)

        self.criterion = torch_loss_dict[self.config.train.loss.criterion]()
        
        self.logger = config.train.logger
        self.log_files = [os.path.join(self.config.path.logs, self.model.name + '_train_loss.csv'),
                          os.path.join(self.config.path.logs, self.model.name + '_train_acc.csv'),
                          os.path.join(self.config.path.logs, self.model.name + '_valid_loss.csv'),
                          os.path.join(self.config.path.logs, self.model.name + '_valid_acc.csv')]
        
        self.epochs = 0
        self.load_logs()
        

    def _train_valid_shuffle(self):
#         train_set, valid_set = train_valid_split(
#             # dataset=AugDataset(self.train_data, num=0, affine_kernel=self.config.data.affine_kernel),
#             dataset=BasicDataset(self.config.path.train, labeled=True, preprocess=True, augment=False),
#             valid_ratio=self.config.data.valid_ratio, 
#             shuffle=True
#         )

        train_data, valid_data = train_valid_raw_split(
            self.config.path.train
        )
        train_set = BasicDataset(train_data, mode='train', preprocess=True, augment=False)
        valid_set = BasicDataset(valid_data, mode='valid', preprocess=True, augment=False)
        
        return self._data_loader(train_set, True), self._data_loader(valid_set, False)


    def _data_loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.config.data.batch_size,
                          shuffle=shuffle, num_workers=self.num_workers)


    def configure(self):
        self.device = self.config.system.device
        self.num_workers = self.config.system.num_workers
        
        self.model = self.config.model.model.to(self.device)
        params_backbone, params_classifier = [], []
        for module in self.model.children():
            if module == self.model.fc:
                params_classifier += list(module.parameters())
            else:
                params_backbone += list(module.parameters())
        
        if self.config.train.lr.backbone != self.config.train.lr.classifier:
            self.lr = {
                'backbone': self.config.train.lr.backbone,
                'classifier': self.config.train.lr.classifier
            }
            if self.config.train.optimizer.lower() == 'adamp':
                self.optimizer_backbone = AdamP(
                    [param for param in params_backbone if param.requires_grad],
                    lr=self.lr['backbone'],
                    betas=self.config.train.betas, 
                    weight_decay=self.config.train.weight_decay
                )
                self.optimizer_classifier = AdamP(
                    [param for param in params_classifier if param.requires_grad],
                    lr=self.lr['classifier'],
                    betas=self.config.train.betas, 
                    weight_decay=self.config.train.weight_decay
                )
            else:
                self.optimizer_backbone = torch_optim_dict[self.config.train.optimizer](
                    [param for param in params_backbone if param.requires_grad],
                    lr=self.config.train.lr.backbone,
                    weight_decay=self.config.train.weight_decay
                )
                self.optimizer_classifier = torch_optim_dict[self.config.train.optimizer](
                    [param for param in params_classifier if param.requires_grad],
                    lr=self.config.train.lr.classifier,
                    weight_decay=self.config.train.weight_decay
                )
        else:
            self.lr = self.config.train.lr.base
            if self.config.train.optimizer.lower() == 'adamp':
                self.optimizer = AdamP(
                    [param for param in self.model.parameters() if param.requires_grad],
                    lr=self.config.train.lr.base, 
                    betas=self.config.train.betas, 
                    weight_decay=self.config.train.weight_decay
                )
            else:
                self.optimizer = torch_optim_dict[self.config.train.optimizer](
                    [param for param in self.model.parameters() if param.requires_grad],
                    lr=self.config.train.lr.base,
                    weight_decay=self.config.train.weight_decay
                )
     

    def load_logs(self):
        dests = [self.logger.train['loss'], self.logger.train['acc'],
                 self.logger.valid['loss'], self.logger.valid['acc']]
        for i in range(4):
            if not os.path.exists(self.log_files[i]):
                continue

            with open(self.log_files[i], newline='') as csvfile:
                reader = csv.reader(csvfile)
                for epochs, value in reader:
                    dests[i].append((int(epochs), float(value)))

            if i == 0:
                self.epochs = int(epochs)


    def write_log(self, dest: str, *values):
        with open(dest, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(values)

        
    def train_and_save(self):
        self.configure()

        print(f"[INFO] device={self.device}({torch.cuda.get_device_name(self.device)}), \
model={self.model.name}, epochs={self.config.train.num_epochs}")
        print(f"       lr={self.lr}, batch size={self.config.data.batch_size}, \
optimizer={self.config.train.optimizer}")
        if self.config.train.optimizer.lower() == 'adamp':
            print(f"       betas={self.config.train.betas}")
        print()
        print(f"Start of traning.")


        def save_with_name():
            # config_mode="json" with function to convert ConfigTree to JSON
            try:
                self.save(self.model.name, config_mode=None)
            except AttributeError:
                self.save(config_mode=None)


        last_train_loss = 0
        last_train_score = 0
        last_valid_loss = 0
        last_valid_score = 0
        for _ in range(self.config.train.num_epochs):
            if self.config.train.valid_min == 0:
                valid_loss, valid_score = self.valid()
                last_valid_score = valid_score
            
            self.epochs += 1
            
            train_loss, train_score = self.train_one_epoch()
            last_train_loss, last_train_score = train_loss, train_score

            # k-fold validation
            if self.config.train.shuffle_period > 0 \
            and self.epochs % self.config.train.shuffle_period == 0:
                self.train_loader, self.valid_loader = self._train_valid_shuffle()

            if self.config.train.valid_period > 0 \
            and self.epochs >= self.config.train.valid_min \
            and self.epochs % self.config.train.valid_period == 0:
                valid_loss, valid_score = self.valid()
                last_valid_score = valid_score
            
            # save state
            if self.config.train.save_period > 0 \
            and self.epochs >= self.config.train.save_min \
            and self.epochs % self.config.train.save_period == 0 \
            and last_valid_score >= self.config.train.save_min_acc:
                save_with_name()
            
            # test
            if self.config.train.test_period > 0 \
            and self.epochs >= self.config.train.test_min \
            and self.epochs % self.config.train.test_period == 0 \
            and last_valid_score >= self.config.train.test_min_acc:
                self.infer_test_and_save()

            # plot loss & acc
            if self.config.train.plot_period > 0 \
            and self.config.train.plot_period > 0 \
            and self.epochs % self.config.train.plot_period == 0:
                self.logger.plot()
        
        self.valid()
        save_with_name()

        print()
        print("End of training.")
        print()
        
        
    def train_one_epoch(self, add_loader=None):
        print(f"[Epoch {self.epochs:03d}]", end="")
        
        self.model.train()
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        starter.record()
        
        data_loader = add_loader if add_loader else self.train_loader
        epoch_time = train_loss = 0
        correct = total = 0
        for batch, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            
            if self.config.train.lr.backbone != self.config.train.lr.classifier:
                self.optimizer_backbone.zero_grad()
                self.optimizer_classifier.zero_grad()
            else:
                self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            targets = targets.to(self.device)

            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.config.train.lr.backbone != self.config.train.lr.classifier:
                self.optimizer_backbone.step()
                self.optimizer_classifier.step()
            else:
                self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

            ender.record()
            torch.cuda.synchronize()
            
            batch_time = starter.elapsed_time(ender) - epoch_time
            epoch_time += batch_time
            
            print(f'\r[Epoch {self.epochs:03d}] (Batch #{batch:03d})  \
Loss: {train_loss / (batch + 1):.5f},  Acc: {correct / total * 100:.3f}  \
({time_str(epoch_time)})', end='')
            
        print()

        train_loss /= batch + 1
        accuracy  = correct / total
        
        self.logger.log_train(self.epochs, train_loss, accuracy)
        self.write_log(self.log_files[0], self.epochs, train_loss)
        self.write_log(self.log_files[1], self.epochs, accuracy)

        return train_loss, accuracy
        
        
    def valid(self):
        self.configure()

        valid_num = (self.epochs - self.config.train.valid_min + 1)\
         // self.config.train.valid_period if self.epochs > 0 else self.epochs
        print(f"[Valid {valid_num:03d}] ", end=" ")
        
        self.model.eval()
        
        valid_loss = 0
        correct = total = 0
        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(self.valid_loader):
                if self.config.train.valid_iters > 0 and batch >= self.config.train.valid_iters:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                valid_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)

                total += targets.shape[0]
                correct += predictions.eq(targets).sum().item()

                print(f'\r[Valid {valid_num:03d}] (Batch #{batch:03d})  \
Loss: {valid_loss / (batch + 1):.5f},  Acc: {correct / total * 100:.3f}', end='')

        print()
        valid_loss /= batch + 1
        accuracy = correct / total
        
        self.logger.log_valid(self.epochs, valid_loss, accuracy)
        self.write_log(self.log_files[2], self.epochs, valid_loss)
        self.write_log(self.log_files[3], self.epochs, accuracy)

        return valid_loss, accuracy
        
        
    def save(self, name='', postfix=None, config_mode=None):
        file_name = f'{datetime.today()}_{name}_{postfix if postfix else ""}'
        
        # save model parameter
        model_output = os.path.join(self.config.path.models, file_name + ".pth")
        torch.save(self.model.state_dict(), model_output)
        print(f"Saved model: {model_output}")

        # save configurations
        if config_mode == "json":
            config_output = os.path.join(self.config.path.configs, file_name + ".json")
            with open(config_output, "w") as f:
                json.dump(self.config, f)
                
                
    def infer_test_and_save(self):
        self.configure()
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()

        self.model.eval()
        indices = [path.split('/')[-1] for path in glob(os.path.join(self.config.path.test, '*.jpg'))]
        result = pd.DataFrame(columns=['ans'], index=indices)
        result.index.name = 'ImageID'
        
        with torch.no_grad():
            for i, (inputs, filenames) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                filenames = np.array(filenames)

                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1).cpu()

                result['ans'][filenames] = predictions

                ender.record()
                torch.cuda.synchronize()
                infer_time = starter.elapsed_time(ender)
                
                print(f"\rEvaluating: batch #{i} ({time_str(infer_time)})", end="")
        
        csv_name = f"{datetime.today()}.csv"
        csv_path = os.path.join(self.config.path.output, csv_name)
        result.to_csv(csv_path)
        
        print(f"\rSaved result: {csv_path}")