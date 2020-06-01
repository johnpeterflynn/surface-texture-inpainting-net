from collections import defaultdict
from datetime import datetime
import time
import lpips
import piq
import torch
from tqdm import tqdm
import utils.metrics.metric as module_metric
from base.base_trainer import BaseTrainer
import datasets
from models.losses.vgg16 import VGGLOSS
from models.losses import losses
from models import surfacetextureinpaintingnet
from utils import MetricTracker
from utils import model_utils, visualization_utils
from utils.metrics import fid_score_cumulative
from models import gan_networks


def unit_test_check_nan_in_model(model, logger):
    num_nan = model_utils.count_nan_parameters(model)#.item()
    #print('Num NaN:', num_nan)
    #input()
    if num_nan > 0:
        logger.error('Num NaN model params:', num_nan, '/', model_utils.count_parameters(model),
                     'Try turning on torch.autograd.set_detect_anomaly(True).')


class Inpainting2DTrainer(BaseTrainer):
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

        graphresnet_enabled = config['archs']['SurfaceTextureInpaintingNet']['enabled']
        graph_enabled = graphresnet_enabled

        #graph_enabled = config['archs']['SingleConvMeshNet']['enabled']
        classic_enabled = config['archs']['Resnet2D']['enabled']

        self.graph_enabled = graph_enabled
        self.conv2d_enabled = classic_enabled

        assert not (self.graph_enabled and self.conv2d_enabled), 'Support for running graph and 2d models simultaneously is disabled'

        # build model architecture, then print to console
        if graphresnet_enabled:
            self.models['graph'] = surfacetextureinpaintingnet.define_G(**config['archs']['SurfaceTextureInpaintingNet']['args'],
                                                                        gpu_ids=[self.device]).to(self.device)
        if classic_enabled:
            #self.models['2d'] = ImageConvCompletionNet(**config['archs']['UNet']['args'])
            #self.models['2d'] = UNet(**config['archs']['UNet']['args'])
            self.models['2d'] = surfacetextureinpaintingnet.define_G(**config['archs']['Resnet2D']['args'],
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
        self.criterion = torch.nn.L1Loss()
        self.metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

        # Init model for GAN loss
        self.use_gan = self.config['trainer']['use_gan']
        if self.use_gan:
            # TODO: NOTE: Chose 5 layers to be compatible with 128x128. Will this work well? How to choose it for
            #  larger images?
            netD = gan_networks.define_D(input_nc=1 + 3 + 3, ndf=64, netD='n_layers', n_layers_D=5, norm='instance', init_gain=0.02,
                                              gpu_ids = [self.device]).to(self.device)
            print('Num Discriminator Parameters:', model_utils.count_parameters(netD))
            optimizer_D = torch.optim.Adam(netD.parameters(), lr=config['optimizer']['args']['lr'],
                                                betas=(0.5, 0.999))
            self.models['discriminator'] = netD
            self.optimizers['discriminator'] = optimizer_D

            self.gan_loss_weight = self.config['trainer']['gan_loss_weight']
            self.gan_mode = self.config['trainer']['gan_mode']
            self.criterionGAN = gan_networks.GANLoss(gan_mode=self.gan_mode).to(self.device)


        # Init model for VGG loss
        self.use_vgg = self.config['trainer']['use_vgg']
        self.vgg_content_weight = self.config['trainer']['vgg_content_weight']
        self.vgg_style_weight = self.config['trainer']['vgg_style_weight']
        self.use_total_variation = self.config['trainer']['use_total_variation']
        self.total_variation_weight = self.config['trainer']['total_variation_weight']
        if self.use_vgg:
            self.criterionVGG = VGGLOSS().to(self.device, non_blocking=True)
        #self.criterion = VGGPerceptualLoss(resize=False).to(self.device, non_blocking=True)
        #self.criterionVGG = torch.nn.DataParallel(self.criterionVGG, device_ids)

        self.config = config
        self.do_validation = self.config['trainer']['do_validation']
        self.num_cumulated_train_batches = self.config['data_loader']['args']['num_cumulated_train_batches']

        def get_instance(module, name, config, *args):
            return getattr(module, config[name]['type'])(*args, **config[name]['args'])

        self.lr_schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.lr_schedulers[name] = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', self.config,
                                                    self.optimizers[name])

        self.visualize_predictions = self.config['trainer']['visualize_predictions']
        self.visualize_samples = self.config['trainer']['visualize_samples']

        self.epochs_per_fid = self.config['trainer']['epochs_per_fid']
        self.use_train_fid = self.config['trainer']['use_train_fid']
        self.use_val_fid = self.config['trainer']['use_val_fid']
        self.fid_p_train_id, self.fid_g_train_id = 'p_train', 'g_train'
        self.fid_p_valid_id, self.fid_g_valid_id = 'p_valid', 'g_valid'
        if self.use_train_fid or self.use_val_fid:
            self.fid = fid_score_cumulative.FIDScoreCumulative(device=self.device)

            def compute_fid_statistics(dataloader, id):
                self.fid.add_session(id, len(dataloader.dataset))
                for batch_idx, data in tqdm(enumerate(dataloader)):
                    data = data.to(self.device)
                    color = data.color.reshape((-1, self.data_loader.train_dataset.img_size,
                                                self.data_loader.train_dataset.img_size, 3))
                    color = color.permute(0, 3, 1, 2)
                    self.fid.add_activation(id, color)
                mg, sg = self.fid.calculate_activation_statistics(id)
                return mg, sg

            # Calculate train and validation statistics
            if self.use_val_fid:
                print('Computing validation ground truth statistics for FID score')
                self.mg_valid, self.sg_valid = compute_fid_statistics(self.data_loader.val_loader, self.fid_g_valid_id)

        self.lpips_metric = lpips.LPIPS(net='alex').to(self.device)
        self.l1_metric = torch.nn.L1Loss()
        self.l2_metric = torch.nn.MSELoss()

        metrics = ['loss', 'lpips', 'l1', 'mse', 'psnr']
        if self.use_gan:
            metrics += ['loss_G', 'loss_D_fake', 'loss_D_real', 'accuracy_D_fake', 'accuracy_D_real']
        if self.use_vgg:
            metrics += ['loss_vgg_content', 'loss_vgg_style']
        epoch_only_metrics = ['fid']
        epoch_metrics = metrics + epoch_only_metrics

        self.train_metrics = MetricTracker(*metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(*metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.epoch_train_metrics = MetricTracker(*epoch_metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.epoch_valid_metrics = MetricTracker(*epoch_metrics, *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _eval(self, mode):
        pass

    def _prepare_graph_prior(self, data):
        with torch.no_grad():
            color = data.x.clone()
            mask = data.mask.unsqueeze(1)
            data.x = torch.where(mask.expand_as(color), torch.zeros_like(color), color)
            prior = torch.cat([data.x, mask], dim=-1)

        return prior, data, color, mask

    def _prepare_2d_prior(self, data, img_size):
        with torch.no_grad():
            data.color = data.color.reshape((-1, img_size, img_size, 3)).permute(0, 3, 1, 2)
            data.mask = data.mask.reshape((-1, 1, img_size, img_size))
            data.x = data.x.reshape((data.color.shape[0], img_size, img_size, -1)).permute(0, 3, 1, 2)

        return data

    def _graph_forward(self, data, color):
        output_graph = self.models['graph'](data)
        return torch.where(data.mask.expand_as(color), output_graph, color)

    def _conv2d_forward(self, data, color):
        output = self.models['2d'](data.x)
        return torch.where(data.mask.expand_as(color), output, data.color)

    def _compute_graph_loss(self, output, color, backward=True, num_cum_batches=1):
        loss_dict = {}
        loss = self.criterion(output, color)
        with torch.no_grad():
            output = output.reshape((-1, self.data_loader.train_dataset.img_size,
                                     self.data_loader.train_dataset.img_size, 3))
            output = output.permute(0, 3, 1, 2)
            color = color.reshape((-1, self.data_loader.train_dataset.img_size,
                                   self.data_loader.train_dataset.img_size, 3))
            color = color.permute(0, 3, 1, 2)

        if self.use_vgg:
            loss_content, loss_style = self.criterionVGG(output, color)
            loss += self.vgg_content_weight * loss_content + self.vgg_style_weight * loss_style
            loss_content = loss_content.item()
            loss_style = loss_style.item()
            loss_dict['loss_vgg_content'] = loss_content
            loss_dict['loss_vgg_style'] = loss_style
            if self.use_total_variation:
                loss += losses.total_variation_loss(output, self.total_variation_weight)
        loss /= num_cum_batches
        if backward:
            loss.backward()
        loss_dict['loss'] = loss.item() * num_cum_batches
        return loss, loss_dict

    def _compute_2d_loss(self, output, color, backward=True, num_cum_batches=1):
        loss_dict = {}
        loss = self.criterion(output, color)
        if self.use_vgg:
            loss_content, loss_style = self.criterionVGG(output, color)
            loss += self.vgg_content_weight * loss_content + self.vgg_style_weight * loss_style
            loss_content = loss_content.item()
            loss_style = loss_style.item()
            loss_dict['loss_vgg_content'] = loss_content
            loss_dict['loss_vgg_style'] = loss_style
            if self.use_total_variation:
                loss += losses.total_variation_loss(output, self.total_variation_weight)
        loss /= num_cum_batches
        if backward:
            loss.backward()
        loss_dict['loss'] = loss.item() * num_cum_batches
        return loss, loss_dict

    def _backward_D(self, prior, fake, real):
        """Calculate GAN loss for the discriminator"""
        loss_dict = {}
        # Fake; stop backprop to the generator by detaching fake_B
        fake_input = torch.cat((prior, fake), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.models['discriminator'](fake_input.detach())
        loss_dict['accuracy_D_fake'] = torch.mean(1 - torch.sigmoid(pred_fake)).item()
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_dict['loss_D_fake'] = loss_D_fake.item()
        # Real
        real_input = torch.cat((prior, real), 1)
        pred_real = self.models['discriminator'](real_input)
        loss_dict['accuracy_D_real'] = torch.mean(torch.sigmoid(pred_real)).item()
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_dict['loss_D_real'] = loss_D_real.item()
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()

        return loss_dict

    def _backward_G(self, prior, fake, real):
        """Calculate GAN and other losses for the generator"""
        loss_total, loss_dict = self._compute_2d_loss(fake, real, backward=False)

        if self.use_gan:
            # First, G(A) should fake the discriminator
            fake_input = torch.cat((prior, fake), 1)
            pred_fake = self.models['discriminator'](fake_input)
            loss_G = self.criterionGAN(pred_fake, True)
            loss_dict['loss_G'] = loss_G.item()

            # combine loss and calculate gradients
            loss_total += self.gan_loss_weight * loss_G

        loss_total.backward()
        loss_dict['loss'] = loss_total.item()  # Override prev loss value
        return loss_dict

    # TODO: WARNING: CAN this interfere with loss functions with weights?
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for name, model in self.models.items():
            unit_test_check_nan_in_model(model, logger=self.logger)
            model.train()

        use_fid_this_epoch = self.use_train_fid and epoch % self.epochs_per_fid == 0
        if use_fid_this_epoch:
            self.fid.add_session(self.fid_p_train_id, len(self.data_loader.train_dataset))
            self.fid.add_session(self.fid_g_train_id, len(self.data_loader.train_dataset))

        self.train_metrics.reset()

        #synced_timers = SyncedTimer(num_drop=3)

        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(skip_first=1, wait=2, warmup=1, active=3, repeat=4),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./saved/overfit_graphs/{}_sage_batch32_nonorm'.format(datetime.now().strftime(r'%m%d_%H%M%S'))),
                record_shapes=True,
                with_stack=True
        ) as prof:
            with torch.profiler.record_function("model_training"):
                len_epoch = len(self.data_loader.train_loader)
                #for name, optimizer in self.optimizers.items():
                #    optimizer.zero_grad()
                for batch_idx, data in enumerate(self.data_loader.train_loader):
                    data = data.to(self.device)

                    # Graph
                    if self.graph_enabled:
                        self.optimizers['graph'].zero_grad()
                        #synced_timers.start('graph_fwd')
                        output = self._graph_forward(data, data.color)
                        #synced_timers.end('graph_fwd')
                        _, loss_dict = self._compute_graph_loss(output, data.color, backward=True)
                        self.optimizers['graph'].step()

                        with torch.no_grad():
                            data.color = data.color.reshape((-1, self.data_loader.train_dataset.img_size,
                                                             self.data_loader.train_dataset.img_size, 3))
                            data.color = data.color.permute(0, 3, 1, 2)
                            output = output.reshape((-1, self.data_loader.train_dataset.img_size,
                                                     self.data_loader.train_dataset.img_size, 3))
                            output = output.permute(0, 3, 1, 2)

                    # 2D
                    if self.conv2d_enabled:
                        data = self._prepare_2d_prior(data, self.data_loader.train_dataset.img_size)
                        #synced_timers.start('2d_fwd')
                        output = self._conv2d_forward(data, data.color)
                        #synced_timers.end('2d_fwd')
                        #loss_dict = self._compute_2d_loss(output, color, backward=True)

                        loss_dict = {}
                        if self.use_gan:
                            # TODO: Q: Shouldn't we turn off the generator's gradients here?
                            # Discriminator
                            self.set_requires_grad(self.models['discriminator'], True)
                            self.optimizers['discriminator'].zero_grad()
                            loss_dict_D = self._backward_D(prior=data.x, fake=output, real=data.color)
                            loss_dict.update(loss_dict_D)
                            self.optimizers['discriminator'].step()

                        # Generator
                        if self.use_gan:
                            self.set_requires_grad(self.models['discriminator'], False)
                        self.optimizers['2d'].zero_grad()
                        loss_dict_G = self._backward_G(prior=data.x, fake=output, real=data.color)
                        loss_dict.update(loss_dict_G)
                        self.optimizers['2d'].step()

                    #if (batch_idx + 1) % self.num_cumulated_train_batches == 0:
                    #    for name, optimizer in self.optimizers.items():
                    #        optimizer.step()
                    #        optimizer.zero_grad()
                    #prof.step()

                    with torch.no_grad():
                        lpips_score = self.lpips_metric(output, data.color).mean().item()
                        l1_score = self.l1_metric(output, data.color).item()
                        l2_score = self.l2_metric(output, data.color).item()
                        psnr_score = piq.psnr(output+1, data.color+1, data_range=2).item()

                        if use_fid_this_epoch:
                            self.fid.add_activation(self.fid_p_train_id, output)
                            self.fid.add_activation(self.fid_g_train_id, data.color)

                    self.writer.set_step((epoch - 1) * len_epoch + batch_idx)
                    for loss_name, loss_value in loss_dict.items():
                        self._update_batch_epoch_metric(loss_name, loss_value, 'train')
                    self._update_batch_epoch_metric('lpips', lpips_score, 'train')
                    self._update_batch_epoch_metric('l1', l1_score, 'train')
                    self._update_batch_epoch_metric('mse', l2_score, 'train')
                    self._update_batch_epoch_metric('psnr', psnr_score, 'train')
                    #for met in self.metric_ftns:
                    #    self.train_metrics.update(met.__name__, met(output, color), write=False)

                    if batch_idx % self.config['trainer']['batches_per_log'] == 0:
                        self._print_batch_update('Train', epoch, batch_idx, len_epoch, loss_dict['loss'], '<name>')

                #synced_timers.print(reset=True)

        #input('finished train')

        # Change logging type to epoch and log all results
        # WARNING: This sets the writer step to epoch mode until it is changed again
        self.writer.set_step(epoch - 1, 'epoch_train', quiet=True)
        log = self.train_metrics.result(write=True)

        if use_fid_this_epoch:
            self.logger.debug('Computing train ground truth FID statistics')
            mg, sg = self.fid.calculate_activation_statistics(self.fid_g_train_id)
            self.fid.remove_session(self.fid_g_train_id)
            self.logger.debug('Computing train prediction FID statistics')
            mp_g, sp_g = self.fid.calculate_activation_statistics(self.fid_p_train_id)
            self.fid.remove_session(self.fid_p_train_id)
            score = self.fid.calculate_frechet_distance(mp_g, sp_g, mg, sg)
            self.epoch_train_metrics.update('fid', score, write=True)

        # visualize last in the batch
        if self.visualize_predictions:
            visualization_utils.visualize_tensor(self.writer, 'predicted', output.cpu(), normalize=True, range=(-1, 1))
            visualization_utils.visualize_tensor(self.writer, 'ground_truth', data.color.cpu(), normalize=True, range=(-1, 1))
            #visualization_utils.visualize_tensor(self.writer, 'mask', data.mask.float().cpu(), normalize=True, range=(-0.5, 0.5))

        if self.visualize_samples:
            self._visualize_select_data(self.data_loader.sample_train_loader,
                                        self.data_loader.train_dataset.img_size, train=True)

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

        use_fid_this_epoch = self.use_val_fid and epoch % self.epochs_per_fid == 0
        if use_fid_this_epoch:
            self.fid.add_session(self.fid_p_valid_id, len(self.data_loader.val_dataset))

        self.valid_metrics.reset()

        with torch.no_grad():
            len_epoch = len(self.data_loader.val_loader)
            for batch_idx, data in enumerate(self.data_loader.val_loader):
                data = data.to(self.device)

                # Graph
                if self.graph_enabled:
                    output = self._graph_forward(data, data.color)
                    _, loss_dict = self._compute_graph_loss(output, data.color, backward=False)

                    data.color = data.color.reshape((-1, self.data_loader.train_dataset.img_size,
                                                     self.data_loader.train_dataset.img_size, 3))
                    data.color = data.color.permute(0, 3, 1, 2)
                    output = output.reshape((-1, self.data_loader.train_dataset.img_size,
                                             self.data_loader.train_dataset.img_size, 3))
                    output = output.permute(0, 3, 1, 2)

                # 2D
                if self.conv2d_enabled:
                    data = self._prepare_2d_prior(data, self.data_loader.val_dataset.img_size)
                    output = self._conv2d_forward(data, data.color)
                    _, loss_dict = self._compute_2d_loss(output, data.color, backward=False)

                with torch.no_grad():
                    lpips_score = self.lpips_metric(output, data.color).mean().item()
                    l1_score = self.l1_metric(output, data.color).item()
                    l2_score = self.l2_metric(output, data.color).item()
                    psnr_score = piq.psnr(output+1, data.color+1, data_range=2).item()

                if use_fid_this_epoch:
                    # For now, val ground truth statistics are calculated once at the start of training
                    self.fid.add_activation(self.fid_p_valid_id, output)

                self.writer.set_step((epoch - 1) * len_epoch + batch_idx, 'valid')
                for loss_name, loss_value in loss_dict.items():
                    self._update_batch_epoch_metric(loss_name, loss_value, 'valid')
                self._update_batch_epoch_metric('lpips', lpips_score, 'valid')
                self._update_batch_epoch_metric('l1', l1_score, 'valid')
                self._update_batch_epoch_metric('mse', l2_score, 'valid')
                self._update_batch_epoch_metric('psnr', psnr_score, 'valid')
                #for met in self.metric_ftns:
                #    self.valid_metrics.update(met.__name__, met(output, color), write=False)

                if batch_idx % self.config['trainer']['batches_per_log'] == 0:
                    self._print_batch_update('Valid', epoch, batch_idx, len_epoch, loss_dict['loss'], '<name>')

        # Change logging type to epoch and log all results
        # WARNING: This sets the writer step to epoch mode until it is changed again
        self.writer.set_step(epoch - 1, 'epoch_valid', quiet=True)
        log = self.valid_metrics.result(write=True)

        if use_fid_this_epoch:
            self.logger.debug('Computing val prediction FID statistics')
            mp_g_valid, sp_g_valid = self.fid.calculate_activation_statistics(self.fid_p_valid_id)
            self.fid.remove_session(self.fid_p_valid_id)
            score = self.fid.calculate_frechet_distance(mp_g_valid, sp_g_valid, self.mg_valid, self.sg_valid)
            self.epoch_valid_metrics.update('fid', score, write=True)

        # visualize last in the batch
        if self.visualize_predictions:
            visualization_utils.visualize_tensor(self.writer, 'predicted', output.cpu(), normalize=True, range=(-1, 1))
            visualization_utils.visualize_tensor(self.writer, 'ground_truth', data.color.cpu(), normalize=True, range=(-1, 1))

        if self.visualize_samples:
            self._visualize_select_data(self.data_loader.sample_val_loader,
                                        self.data_loader.val_dataset.img_size, train=False)

        return log

    def _visualize_select_data(self, dataloader, img_size, train=False):
        #for name, model in self.models.items():
        #    if train:
        #        model.train()
        #    else:
        #        model.eval()

        disp_prior = []
        disp_mask = []
        disp_pred = []

        def _generate_disp_tensors():
            with torch.no_grad():
                for batch_idx, data in enumerate(dataloader):
                    data = data.to(self.device)

                    # Graph
                    if self.graph_enabled:
                        output = self._graph_forward(data, data.color)

                        data.x = torch.where(data.mask.expand_as(data.color), -1*torch.ones_like(data.color), data.color)
                        data.color = data.color.reshape((-1, img_size, img_size, 3)).permute(0, 3, 1, 2)
                        data.x = data.x.reshape((-1, img_size, img_size, 3)).permute(0, 3, 1, 2)
                        output = output.reshape((-1, img_size, img_size, 3))
                        output = output.permute(0, 3, 1, 2)
                        disp_pred.append(output)

                    # 2D
                    if self.conv2d_enabled:
                        data = self._prepare_2d_prior(data, img_size)
                        output = self._conv2d_forward(data, data.color)
                        disp_pred.append(output)

                    disp_prior.append(data.color)
                    disp_mask.append(data.x)

        for name, model in self.models.items():
            if train:
                model.train()
            else:
                model.eval()
        _generate_disp_tensors()

        disp_prior = torch.cat(disp_prior, dim=0).cpu()
        visualization_utils.visualize_tensor(self.writer, 'sample_prior', disp_prior, single=False, normalize=True, range=(-1, 1))
        disp_mask = torch.cat(disp_mask, dim=0).cpu()
        visualization_utils.visualize_tensor(self.writer, 'sample_mask', disp_mask, single=False, normalize=True, range=(-1, 1))
        disp_pred = torch.cat(disp_pred, dim=0).cpu()
        visualization_utils.visualize_tensor(self.writer, 'sample', disp_pred, single=False, normalize=True, range=(-1, 1))

    def _update_batch_epoch_metric(self, name, score, type):
        if type == 'train':
            batch_metric = self.train_metrics
            epoch_metric = self.epoch_train_metrics
        elif type == 'valid':
            batch_metric = self.valid_metrics
            epoch_metric = self.epoch_valid_metrics
        else:
            raise NotImplementedError

        batch_metric.update(name, score, write=True)
        epoch_metric.update(name, score, write=False)

    def _print_batch_update(self, mode, epoch, batch_idx, len_epoch, loss, names):
        str = ':{} Epoch: {} {} I Loss: {:.6f} Names: {}'.format(mode, epoch,
                                                                self._progress(batch_idx, len_epoch),
                                                                loss, names)
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
        #arch = type(self.model).__name__

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
                self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                    "checkpoint ({}). This may yield an exception while state_dict is being loaded.".format(arch))
            elif not self.config['archs'][arch]['enabled']:
                self.logger.warning("Warning: Architecture {} given in checkpoint is disabled in config file.".format(arch))
            else:
                self.models[name].load_state_dict(model)
                self.logger.info('Loaded {} model with architecture {}'.format(name, arch))

        for name, opt in checkpoint['optimizers'].items():
            # load optimizer state from checkpoint.
            # TODO: Check that optimizer instance exists in trainer.
            self.optimizers[name].load_state_dict(opt)
            self.logger.info('Loaded {} optimizer'.format(name))

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
