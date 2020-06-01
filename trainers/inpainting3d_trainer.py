import torch
import utils.metrics.metric as module_metric
from base.base_trainer import BaseTrainer
import datasets
from models import surfacetextureinpaintingnet
from utils import MetricTracker
from utils.ColorCompletionVisualizer import ColorCompletionVisualizer
from utils import model_utils, unit_tests, metrics
from utils.metrics import graph_metrics


class Inpainting3DTrainer(BaseTrainer):
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

        assert self._n_gpus == 1, 'Error: Trainer not implemented for multi-gpu training yet'

        self.models = {}

        # build model architecture, then print to console
        self.models['graph'] = surfacetextureinpaintingnet.define_G(**config['archs']['SurfaceTextureInpaintingNet']['args'],
                                                                    gpu_ids=[self.device]).to(self.device)

        logger.info('Device: {}, Device IDs: {}'.format(self.device, device_ids))
        for name, model in self.models.items():
            logger.info(model)
            self.models[name] = self.models[name].to(self.device)
            logger.info("Number of parameters in {}: {}".format(name, model_utils.count_parameters(model)))

        self.optimizers = {}

        # build optimizer
        for name, model in self.models.items():
            trainable_params = filter(lambda p: p.requires_grad, self.models[name].parameters())
            self.optimizers[name] = config.init_obj('optimizer', torch.optim, trainable_params)

        if self.config.resume is not None:
            self._resume_checkpoint(config.resume)

        # setup data_loader instances
        self.data_loader = config.init_obj_with_config('data_loader', datasets, multi_gpu=self._n_gpus > 1)

        # get function handles of loss and metrics
        self.criterion = torch.nn.L1Loss(reduction='none')
        self.metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

        self.config = config
        self.do_validation = self.config['trainer']['do_validation']
        self.num_cumulated_train_batches = self.config['data_loader']['args']['num_cumulated_train_batches']

        def get_instance(module, name, config, *args):
            return getattr(module, config[name]['type'])(*args, **config[name]['args'])

        self.lr_schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.lr_schedulers[name] = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', self.config,
                                                    self.optimizers[name])

        # TODO: This is a hack for now to make checkpointing work. Should work for all models/optimizers automatically
        self.optimizer = self.optimizers['graph']
        self.model = self.models['graph']

        self.use_mask_weighted_loss = self.config['trainer']['use_mask_weighted_loss']
        self.visualize_predictions = self.config['trainer']['visualize_predictions']
        self.visualize_samples = self.config['trainer']['visualize_samples']

        self.l1_metric = torch.nn.L1Loss()
        self.l2_metric = torch.nn.MSELoss()
        self.laplace_var_metric = graph_metrics.GraphLaplaceVariance().to(self.device)

        metrics = ['loss', 'l1', 'mse', 'graph_tv', 'graph_lap_var', 'psnr', 'psnr_mask_only', 'mem_allocated', 'mem_reserved']
        epoch_only_metrics = []
        epoch_metrics = metrics + epoch_only_metrics

        self.train_metrics = MetricTracker(*metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(*metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.epoch_train_metrics = MetricTracker(*epoch_metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.epoch_valid_metrics = MetricTracker(*epoch_metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _eval(self, mode):
        if mode == 'train':
            dataloader = self.data_loader.train_loader
        elif mode == 'valid':
            dataloader = self.data_loader.val_loader
        else:
            # TODO: Implement eval for test set
            raise NotImplementedError

        assert dataloader.batch_size == 1, 'ERROR: Batch size must be 1 in eval mode'

        if self.config['vis']:
            visualizer = ColorCompletionVisualizer(self.data_loader, "visualizations/")

        self.model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                data = data.to(self.device)

                # Graph
                output_graph = self._graph_forward(data, data.color)
                loss_graph = self.compute_loss(output_graph, data.color,
                                               weights=data.mask if self.use_mask_weighted_loss else None)
                loss_graph = loss_graph.item()
                self._update_metrics(output_graph, data.color, data.mask, data.edge_index, loss_graph, 'valid', write=False)
                self.logger.info('    {} {:15s}: {}'.format(data.name[0], 'loss', loss_graph))

                if self.config['vis']:
                    visualizer.visualize_result(data.name[0],
                                                (output_graph.detach() / 2.0 + 0.5).cpu(),
                                                (data.color.detach() / 2.0 + 0.5).cpu(),
                                                (data.mask.detach() > 0).cpu())
            log = self.valid_metrics.result(write=False)
            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

    def _graph_forward(self, data, color):
        output_graph = self.models['graph'](data)
        return torch.where((data.mask > 0).expand_as(color), output_graph, color)
        #return torch.lerp(color, output_graph, mask.float())

    def compute_loss(self, output, target, weights=None):
        loss = self.criterion(output, target)
        if weights is not None:
            loss *= torch.pow(0.99, weights.squeeze()).unsqueeze(1)

        return loss.mean()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for name, model in self.models.items():
            unit_tests.check_nan_in_model(model, logger=self.logger)
            model.train()

        self.train_metrics.reset()

        #torch.autograd.set_detect_anomaly(True)

        len_epoch = len(self.data_loader.train_loader)
        for name, optimizer in self.optimizers.items():
            optimizer.zero_grad(set_to_none=True)
        for batch_idx, data in enumerate(self.data_loader.train_loader):
            data = data.to(self.device)
            names = data.name

            self.writer.set_step((epoch - 1) * len_epoch + batch_idx)
            mem_allocated = torch.cuda.memory_allocated()
            mem_reserved = torch.cuda.memory_reserved()
            self._update_batch_epoch_metric('mem_allocated', mem_allocated, 'train')
            self._update_batch_epoch_metric('mem_reserved', mem_reserved, 'train')

            output_graph = self._graph_forward(data, data.color)
            #del data
            loss_graph = self.compute_loss(output_graph, data.color,
                                           weights=data.mask if self.use_mask_weighted_loss else None)
            loss_graph /= self.num_cumulated_train_batches
            loss_graph.backward()
            loss_graph = loss_graph.item() * self.num_cumulated_train_batches

            if (batch_idx + 1) % self.num_cumulated_train_batches == 0:
                for name, optimizer in self.optimizers.items():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            self._update_metrics(output_graph, data.color, data.mask, data.edge_index, loss_graph, 'train')
            del output_graph, data

            if batch_idx % self.config['trainer']['batches_per_log'] == 0:
                self._print_batch_update('Train', epoch, batch_idx, len_epoch, loss_graph, names)

        # Change logging type to epoch and log all results
        # WARNING: This sets the writer step to epoch mode until it is changed again
        self.writer.set_step(epoch - 1, 'epoch_train', quiet=True)
        log = self.train_metrics.result(write=True)

        # visualize last in the batch
        # TODO: Visualize predictions

        # TODO: Visualize samples

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        for _, lr_scheduler in self.lr_schedulers.items():
            if lr_scheduler is not None:
                lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        assert self.data_loader.val_loader.batch_size == 1, 'ERROR: Validation step not implemented for batch size > 1'

        for name, model in self.models.items():
            model.eval()

        self.valid_metrics.reset()

        with torch.no_grad():
            len_epoch = len(self.data_loader.val_loader)
            for batch_idx, data in enumerate(self.data_loader.val_loader):
                data = data.to(self.device)
                names = data.name

                self.writer.set_step((epoch - 1) * len_epoch + batch_idx, 'valid')
                mem_allocated = torch.cuda.memory_allocated()
                mem_reserved = torch.cuda.memory_reserved()
                self._update_batch_epoch_metric('mem_allocated', mem_allocated, 'valid')
                self._update_batch_epoch_metric('mem_reserved', mem_reserved, 'valid')

                output_graph = self._graph_forward(data, data.color)
                #del data
                loss_graph = self.compute_loss(output_graph, data.color,
                                               weights=data.mask if self.use_mask_weighted_loss else None)
                loss_graph = loss_graph.item()

                self._update_metrics(output_graph, data.color, data.mask, data.edge_index, loss_graph, 'valid')
                del output_graph, data

                if batch_idx % self.config['trainer']['batches_per_log'] == 0:
                    self._print_batch_update('Valid', epoch, batch_idx, len_epoch, loss_graph, names)

        # Change logging type to epoch and log all results
        # WARNING: This sets the writer step to epoch mode until it is changed again
        self.writer.set_step(epoch - 1, 'epoch_valid', quiet=True)
        log = self.valid_metrics.result(write=True)


        # visualize last in the batch
        # TODO: Visualize predictions

        # TODO: Visualize samples

        return log

    def _update_metrics(self, prediction, ground_truth, mask, edge_index, loss_value, mode, write=True):
        with torch.no_grad():
            l1_score = self.l1_metric(prediction, ground_truth).item()
            l2_score = self.l2_metric(prediction, ground_truth).item()
            graph_tv_score = metrics.graph_metrics.graph_total_variation(prediction, edge_index).item()
            graph_laplace_var_score = self.laplace_var_metric(prediction, edge_index).item()
            psnr_score = metrics.graph_metrics.psnr(prediction, ground_truth, data_range=2.0)
            psnr_masked_only_score = metrics.graph_metrics.psnr(prediction[mask.squeeze() > 0],
                                                                ground_truth[mask.squeeze() > 0],
                                                                data_range=2.0)

        self._update_batch_epoch_metric('loss', loss_value, mode, write)
        self._update_batch_epoch_metric('l1', l1_score, mode, write)
        self._update_batch_epoch_metric('mse', l2_score, mode, write)
        self._update_batch_epoch_metric('graph_tv', graph_tv_score, mode, write)
        self._update_batch_epoch_metric('graph_lap_var', graph_laplace_var_score, mode, write)
        self._update_batch_epoch_metric('psnr', psnr_score, mode, write)
        self._update_batch_epoch_metric('psnr_mask_only', psnr_masked_only_score, mode, write)

        # for met in self.metric_ftns:
        #    self.valid_metrics.update(met.__name__, met(output, color), write=False)

    def _update_batch_epoch_metric(self, name, score, type, write=True):
        if type == 'train':
            batch_metric = self.train_metrics
            epoch_metric = self.epoch_train_metrics
        elif type == 'valid':
            batch_metric = self.valid_metrics
            epoch_metric = self.epoch_valid_metrics
        else:
            raise NotImplementedError

        batch_metric.update(name, score, write=write)
        epoch_metric.update(name, score, write=False)

    def _print_batch_update(self, mode, epoch, batch_idx, len_epoch, loss1, names):
        str = ':{} Epoch: {} {} I Loss: {:.6f} Names: {}'.format(mode, epoch,
                                                                self._progress(batch_idx, len_epoch),
                                                                loss1, names)
        self.logger.debug(str)

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        path = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        self._state_save(epoch, path)
        self.logger.info("Saving checkpoint: {} ...".format(path))

    def _save_best(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        self._state_save(epoch, best_path)
        self.logger.info("Saving current best: model_best.pth ...")

    def _state_save(self, epoch, file_path):
        # arch = type(self.model).__name__

        archs = {}
        models = {}
        for name, model in self.models.items():
            models[name] = model.state_dict()
            archs[name] = type(model).__name__

        optimizers = {}
        for name, opt in self.optimizers.items():
            optimizers[name] = opt.state_dict()

        state = {
            'archs': archs,
            'epoch': epoch,
            'state_dicts': models,
            'optimizers': optimizers,
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        torch.save(state, file_path)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        for name, model in checkpoint['state_dicts'].items():
            arch = checkpoint['archs'][name]
            if arch not in self.config['archs'].keys():
                self.logger.warning(
                    "Warning: Architecture configuration given in config file is different from that of "
                    "checkpoint ({}). This may yield an exception while state_dict is being loaded.".format(arch))
            elif not self.config['archs'][arch]['enabled']:
                self.logger.warning(
                    "Warning: Architecture {} given in checkpoint is disabled in config file.".format(arch))
            else:
                self.models[name].load_state_dict(model)
                self.logger.info('Loaded {} model with architecture {}'.format(name, arch))

        for name, opt in checkpoint['optimizers'].items():
            # load optimizer state from checkpoint.
            # TODO: Check that optimizer instance exists in trainer.
            self.optimizers[name].load_state_dict(opt)
            self.logger.info('Loaded {} optimizer'.format(name))

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))