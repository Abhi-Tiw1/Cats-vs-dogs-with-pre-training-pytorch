# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:49:47 2023

@author: Abhishek
"""
import numpy as np
import torch

class Training():
    def __init__(self, criterion, optimizer, model, metric_fns, data_loader, val_data_loader, config):
        super.init(model, criterion,  metric_fns, optimizer, config)
        self.criterion = criterion
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.validation_run = True
        self.logged_values = []
        
    def _train_epoch(self, epoch):
        
        self.model.train()
        for batch_ix, (batch, target) in enumerate(self.data_loader):
            batch, target = batch.to(self.device), target.to(self.device)
            
            # reet gradiant
            self.optimizer.zero_grad()
            # get predictions
            pred = self.model(batch)
            # calculate loss
            loss = self.criterion(pred, target)
            # take backward step
            loss.backward()
            # update weights
            self.optimizer.step()
            # store loss and batch values
            
            # calculate output metrics
            metrics = [met(target, pred) for met in self.metric_fns]
            # store these values
            if batch_ix % self.log_step == 0:
                # print stuff
                print('Epoch: {}, Batch: {}. Loss: {}, Metrics: {}'.format(epoch, batch_ix, loss, metrics))
            
            # run single validation epoch
            if self.validation_run:
                self._validate_epoch(epoch)
            
        return self.logged_values

    def _validation_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                # ave loss, metric
        
        return 

    
    def train(self):
        if self.start_checkpoint:
            self._load_checkpoint()
        early_stop_count = 0
        for epoch in np.arange(self.start_epoch, self.epochs + 1 ):
            result = self._train_ep(epoch)
            best = False
            improved = check_improved()
            if improved:
                early_stop_count = 0
                self.mnt_best = self.mnt_metric
                best = True
            else:
                early_stop_count = early_stop_count + 1
            
            if early_stop_count >  self.early_stop_epochs:
                print('No improvement in loss --> Early stopping')
                break
            
            if epoch % self.save_period == 0:
                self._checkpoint_model(epoch, save_best = best)