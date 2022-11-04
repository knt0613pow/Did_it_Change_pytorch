import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker,visualize_activation


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (positive, negative) in enumerate(self.data_loader):
            """
            positive['t0'] : (batchsizie, 3, width, height)
            the other format is same shape of (N, 3, W, H)
            """
            p0 , p1 = positive['t0'].float().to(self.device), positive['t1'].float().to(self.device)
            n0 , n1 = negative['t0'].float().to(self.device), negative['t1'].float().to(self.device)

            self.optimizer.zero_grad()
            # output = self.model(data)
            #P0 ,P1 = self.model(P0).squeeze(), self.model(P1).squeeze()
            #N0, N1 = self.model(N0).squeeze(), self.model(N1).squeeze()
            # P0, P1, N0, N1 have same shape of (N, 2048, 1, 1) -> (N, 2048)
            
            P0 ,P1 = self.model(p0).view(1,-1), self.model(p1).view(1,-1)
            N0, N1 = self.model(n0).view(1,-1), self.model(n1).view(1,-1)
            #(2048, 1) -> (1,2048)
            
            loss = self.criterion(P0, P1, N0, N1)
            
            
            #loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            visualize_activation(self.model.model,p0)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(P0, P1, N0, N1))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                self.writer.add_image('pt0', make_grid(p0.cpu()))
                self.writer.add_image('pt1', make_grid(p1.cpu()))
                self.writer.add_image('nt0', make_grid(n0.cpu()))
                self.writer.add_image('nt1', make_grid(n1.cpu()))
                self.writer.add_image('activation_p0', make_grid(visualize_activation(self.model.model,p0).cpu()))
                self.writer.add_image('activation_p1', make_grid(visualize_activation(self.model.model,p1).cpu()))
                self.writer.add_image('activation_n0', make_grid(visualize_activation(self.model.model,n0).cpu()))
                self.writer.add_image('activation_n1', make_grid(visualize_activation(self.model.model,n1).cpu()))
                
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (positive, negative) in enumerate(self.valid_data_loader):
                P0 , P1 = positive['t0'].float().to(self.device), positive['t1'].float().to(self.device)
                N0 , N1 = negative['t0'].float().to(self.device), negative['t1'].float().to(self.device)
                P0 ,P1 = self.model(P0).view(1,-1), self.model(P1).view(1,-1)
                N0, N1 = self.model(N0).view(1,-1), self.model(N1).view(1,-1)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                loss = self.criterion(P0, P1, N0, N1)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(P0, P1, N0, N1))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
