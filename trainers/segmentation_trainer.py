import  numpy as np
import torch
import torch_geometric
import utils.metrics.metric as module_metric
from base.base_trainer import BaseTrainer
from datasets.scannetlabelgraph_dataloader import ScanNetGraphDataLoader
from models.singleconvmeshnet import SingleConvMeshNet
from utils import MetricTracker
from utils.metrics.metrics_dcm import IoUDCM
from utils.metrics.confusionmatrix_dcm import ConfusionMatrixDCM
from utils.SemSegVisualizer import SemSegVisualizer
from utils import visualization_utils, unit_tests


class GraphSegmentationTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config):
        super().__init__(config)
        # init logger
        logger = config.get_logger('train')

        # setup GPU device if available, move model into configured device
        self._n_gpus = config['n_gpu']
        self.device, device_ids = self._prepare_device(self._n_gpus)

        # build model architecture, then print to console
        self.model = SingleConvMeshNet(**config['archs']['SingleConvMeshNet']['args'])

        logger.info(self.model)
        logger.info('Device: {}, Device IDs: {}'.format(self.device, device_ids))
        self.model = self.model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch_geometric.nn.DataParallel(self.model, device_ids=device_ids)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("Number of parameters: {}".format(count_parameters(self.model)))

        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        if self.config.resume is not None:
            self._resume_checkpoint(config.resume)

        # setup data_loader instances
        self.data_loader = ScanNetGraphDataLoader(config['data_loader']['args'], multi_gpu=self._n_gpus > 1)

        # get function handles of loss and metrics
        class_weights = self.data_loader.train_class_weights.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.data_loader.ignore_classes, weight=class_weights)
        self.metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

        self.config = config
        self.do_validation = self.config['trainer']['do_validation']
        self.num_cumulated_train_batches = self.config['data_loader']['args']['num_cumulated_train_batches']

        def get_instance(module, name, config, *args):
            return getattr(module, config[name]['type'])(*args, **config[name]['args'])
        self.lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', self.config, self.optimizer)

        metrics = ['loss', 'mIoU', 'mPrec', 'oPrec', 'oAccuracy']

        self.train_metrics = MetricTracker(*metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(*metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.iou_metrics = MetricTracker('IoUs', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _eval(self, mode):
        if mode == 'train':
            dataloader = self.data_loader.train_loader
        elif mode == 'val':
            dataloader = self.data_loader.val_loader
        else:
            # TODO: Implement eval for test set
            raise NotImplementedError

        if self.config['vis']:
            visualizer = SemSegVisualizer(self.data_loader, "visualizations/")
        self.model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                self.valid_metrics.reset()
                conf_matrix = ConfusionMatrixDCM(self.data_loader.num_classes)
                iou = IoUDCM(ignore_index=self.data_loader.ignore_classes)

                data = data.to(self.device)
                print('eval batch id, scene name', batch_idx, data.name)
                output = self.model(data)
                full_prediction = output[data.original_index_traces]
                loss = self.criterion(full_prediction, data.labels)

                conf_matrix.add(full_prediction, data.labels)
                self.valid_metrics.update('loss', loss.item(), write=False)
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(full_prediction, data.labels), write=False)

                metrics = iou.value(conf_matrix.value(normalized=False))
                self.valid_metrics.update('mIoU', metrics['mean_iou'], write=False)
                self.valid_metrics.update('mPrec', metrics['mean_precision'], write=False)
                self.valid_metrics.update('oPrec', metrics['overall_precision'], write=False)

                log = self.valid_metrics.result(write=False)
                # print logged informations to the screen
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

                if self.config['vis']:
                    argmax_predictions = torch.argmax(full_prediction, dim=1)
                    visualizer.visualize_result(data.name[0], argmax_predictions.cpu(), data.labels.cpu())

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        unit_tests.check_nan_in_model(self.model, logger=self.logger)

        self.model.train()
        self.train_metrics.reset()
        conf_matrix = ConfusionMatrixDCM(self.data_loader.num_classes)
        iou = IoUDCM(ignore_index=self.data_loader.ignore_classes)

        #with torch.profiler.profile(
        #        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #        schedule=torch.profiler.schedule(skip_first=1, wait=5, warmup=1, active=3, repeat=4),
        #        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler_test2'),
        #        record_shapes=True,
        #        with_stack=True
        #) as prof:
        #    with torch.profiler.record_function("model_training"):
        len_epoch = len(self.data_loader.train_loader)
        self.optimizer.zero_grad()
        for batch_idx, data in enumerate(self.data_loader.train_loader):
            if self._n_gpus > 1:
                print('multiple gpus')
                labels = torch.cat([d.labels for d in data]).to(self.device, non_blocking=True) # output.device
            else:
                print('single gpu')
                data = data.to(self.device)
                labels = data.labels
            names = [d.name for d in data] if isinstance(data, list) else data.name
            self.optimizer.zero_grad()
            output = self.model(data[0])
            loss = self.criterion(output, labels) / self.num_cumulated_train_batches
            loss.backward()
            loss = loss.item() * self.num_cumulated_train_batches

            if (batch_idx + 1) % self.num_cumulated_train_batches == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            #prof.step()


            self._print_batch_update('Train', epoch, batch_idx, len_epoch, loss, names)

            conf_matrix.add(output.detach(), labels.detach())
            self.train_metrics.update('loss', loss, write=False)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output.detach(), labels.detach()), write=False)

        metrics = iou.value(conf_matrix.value(normalized=False))
        self.train_metrics.update('mIoU', metrics['mean_iou'], write=False)
        self.train_metrics.update('mPrec', metrics['mean_precision'], write=False)
        self.train_metrics.update('oPrec', metrics['overall_precision'], write=False)
        self.train_metrics.update('oAccuracy', metrics['overall_accuracy'], write=False)

        self.writer.set_step(epoch - 1)

        # TODO: Fix this hack. Mode is tightly controlled by custom TensorBoard writer.
        original_mode = self.writer.get_mode()
        for index, class_iou in enumerate(metrics['iou']):
            if index != self.data_loader.ignore_classes:
                self.writer.set_mode('{}_{}'.format(self.data_loader.class_names[index], original_mode))
                self.iou_metrics.reset()
                self.iou_metrics.update('IoUs', class_iou, write=True)
        self.writer.set_mode(original_mode)

        log = self.train_metrics.result(write=True)

        # Draw visuals from here #
        visualization_utils.visualize_confusion_matrix(self.writer, conf_matrix, metrics['iou'], self.data_loader, epoch)

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
        assert self.data_loader.val_loader.batch_size == 1, 'ERROR: Validation step not implemented for batch size > 1'

        self.model.eval()
        self.valid_metrics.reset()
        conf_matrix = ConfusionMatrixDCM(self.data_loader.num_classes)
        iou = IoUDCM(ignore_index=self.data_loader.ignore_classes)

        with torch.no_grad():
            len_epoch = len(self.data_loader.val_loader)
            for batch_idx, data in enumerate(self.data_loader.val_loader):
                if self._n_gpus > 1:
                    # TODO: Backprop on batches of size > 1
                    labels = data[0].labels.to(self.device, non_blocking=True)
                else:
                    data = data.to(self.device)
                    labels = data.labels
                output = self.model(data)
                if self._n_gpus > 1:
                    # TODO: Generate full prediction on batches of size > 1
                    output = output[data[0].original_index_traces]
                else:
                    output = output[data.original_index_traces]
                loss = self.criterion(output, labels)

                names = data[0].name if isinstance(data, list) else data.name
                self._print_batch_update('Valid', epoch, batch_idx, len_epoch, loss.item(), names)

                conf_matrix.add(output.detach(), labels.detach())
                self.valid_metrics.update('loss', loss.item(), write=False)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output.detach(), labels.detach()), write=False)

        # TODO: Normalized?
        metrics = iou.value(conf_matrix.value(normalized=False))
        self.valid_metrics.update('mIoU', metrics['mean_iou'], write=False)
        self.valid_metrics.update('mPrec', metrics['mean_precision'], write=False)
        self.valid_metrics.update('oPrec', metrics['overall_precision'], write=False)
        self.valid_metrics.update('oAccuracy', metrics['overall_accuracy'], write=False)

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        self.writer.set_step(epoch - 1, 'valid')

        # TODO: Fix this hack. Mode is tightly controlled by custom TensorBoard writer.
        original_mode = self.writer.get_mode()
        for index, class_iou in enumerate(metrics['iou']):
            if index != self.data_loader.ignore_classes:
                self.writer.set_mode('{}_{}'.format(self.data_loader.class_names[index], original_mode))
                self.iou_metrics.reset()
                self.iou_metrics.update('IoUs', class_iou, write=True)
        self.writer.set_mode(original_mode)

        log = self.valid_metrics.result(write=True)

        # Draw visuals from here #
        visualization_utils.visualize_confusion_matrix(self.writer, conf_matrix, metrics['iou'], self.data_loader, epoch)

        return log

    def _print_batch_update(self, mode, epoch, batch_idx, len_epoch, loss, names):
        str = ':{} Epoch: {} {} I Loss: {:.6f} Names: {}'.format(mode, epoch,
                                                                self._progress(batch_idx, len_epoch),
                                                                loss, names)
        self.logger.debug(str)
