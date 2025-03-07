import os.path as osp
import sys
import torch

from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x1, x2, y, i, metadata=None):
        raise NotImplementedError
    
    def seg_batch_update(self, x, y, i, metadata=None):
        raise NotImplementedError

    def seg_batch_update(self, x, y, i, metadata=None):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def check_tensor(self, data, is_label):
        if not is_label:
            return data if data.ndim <= 4 else data.squeeze()
        return data.long() if data.ndim <= 3 else data.squeeze().long()

    def infer_vis(self, dataloader, save=True, evaluate=False, slide=False, image_size=1024,
                  window_size=256, save_dir='./res', suffix='.tif'):
        """
        Infer and save results. (debugging)
        Note: Currently only batch_size=1 is supported.
        Weakly robust.
        'image_size' and 'window_size' work when slide is True.
        """
        import cv2
        import numpy as np

        self.model.eval()
        logs = {}
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for batch in iterator:
                if len(batch) == 5:
                    x1, x2, y, filename, metadata = batch
                else:
                    x1, x2, y, filename = batch
                    metadata = None

                assert y is not None or not evaluate, "When the label is None, the evaluation mode cannot be turned on."

                if y is not None:
                    x1, x2, y = self.check_tensor(x1, False), self.check_tensor(x2, False), \
                                self.check_tensor(y, True)
                    x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                    y_pred = self.model.forward(x1, x2, metadata)
                else:
                    x1, x2 = self.check_tensor(x1, False), self.check_tensor(x2, False)
                    x1, x2 = x1.float(), x2.float()
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    y_pred = self.model.forward(x1, x2, metadata)

                if evaluate:
                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                        metrics_meters[metric_fn.__class__.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)

                if save:
                    y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy().round()
                    y_pred = y_pred * 255
                    filename = filename[0].split('.')[0] + suffix

                    if slide:
                        inf_seg_maps = []
                        window_num = image_size // window_size
                        window_idx = [i for i in range(0, window_num ** 2 + 1, window_num)]
                        for row_idx in range(len(window_idx) - 1):
                            inf_seg_maps.append(np.concatenate([y_pred[i] for i in range(window_idx[row_idx],
                                                                                         window_idx[row_idx + 1])], axis=1))
                        inf_seg_maps = np.concatenate([row for row in inf_seg_maps], axis=0)
                        cv2.imwrite(osp.join(save_dir, filename), inf_seg_maps)
                    else:
                        # To be verified
                        cv2.imwrite(osp.join(save_dir, filename), y_pred)

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        filenames = []
        metrics_meters = {metric.__class__.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for i, batch in enumerate(iterator):
                if len(batch) == 5:
                    x1, x2, y, filename, metadata = batch
                else:
                    x1, x2, y, filename = batch
                    metadata = None
                x1, x2, y = self.check_tensor(x1, False), self.check_tensor(x2, False), \
                            self.check_tensor(y, True)
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                if i == len(dataloader) - 1:
                    i = -1
                loss, y_pred = self.batch_update(x1, x2, y, i, metadata)

                # update loss logs
                loss_value = loss.detach().cpu().numpy()
                loss_meter.add(loss_value)
                filenames.append(filename)
                # y_pred = torch.argmax(y_pred, dim=1)
                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                    metrics_meters[metric_fn.__class__.__name__].add(metric_value)
                

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        loss_logs = {self.loss.__name__: loss_meter.mean}
        logs.update(loss_logs)

        metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
        metrics_logs['filenames'] = filenames
        logs.update(metrics_logs)

        return logs

    def run_seg(self, dataloader):

        self.on_epoch_start()
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__class__.__name__: AverageValueMeter() for metric in self.metrics}


        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for i, batch in enumerate(iterator):
                if len(batch) == 4:
                    x, y, _, metadata = batch
                else:
                    x, y, _ = batch
                    metadata = None

                x, y = self.check_tensor(x, False), self.check_tensor(y, True)
                x, y = x.to(self.device), y.to(self.device)

                if i == len(dataloader) - 1:
                    i = -1
                loss, y_pred = self.seg_batch_update(x, y, i, metadata)
                if type(self.loss).__name__ != 'bce':
                    y_pred = torch.argmax(y_pred, dim=1)

                # update loss logs
                loss_value = loss.detach().cpu().numpy()
                loss_meter.add(loss_value)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                    metrics_meters[metric_fn.__class__.__name__].add(metric_value)
                

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        loss_logs = {type(self.loss).__name__: loss_meter.mean}
        logs.update(loss_logs)

        metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
        logs.update(metrics_logs)

        return logs

class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, grad_accum=1):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.optimizer.zero_grad()
        self.grad_accum = grad_accum

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x1, x2, y, i, metadata=None):
        
        prediction = self.model.forward(x1, x2, metadata)
        loss = self.loss(prediction, y)
        loss = loss / self.grad_accum
        loss.backward()
        if (i + 1) % self.grad_accum == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss, prediction
    
    def seg_batch_update(self, x, y, i, metadata=None):
        prediction = self.model.forward(x, metadata)
        if type(self.loss).__name__ == 'BCEWithLogitsLoss':
            prediction = prediction[:, 0]
            loss = self.loss(prediction, y.float())
        else:
            loss = self.loss(prediction, y)
        loss = loss / self.grad_accum
        loss.backward()
        if (i + 1) % self.grad_accum == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x1, x2, y, i, metadata=None):
        with torch.no_grad():
            prediction = self.model.forward(x1, x2, metadata)
            loss = self.loss(prediction, y)
        return loss, prediction

    def seg_batch_update(self, x, y, i, metadata=None):
        with torch.no_grad():
            prediction = self.model.forward(x, metadata)
            if type(self.loss).__name__ == 'BCEWithLogitsLoss':
                prediction = prediction[:, 0]
                loss = self.loss(prediction, y.float())
            else:
                loss = self.loss(prediction, y)
        return loss, prediction
