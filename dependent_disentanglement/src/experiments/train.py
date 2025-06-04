import collections
import os
from datetime import datetime

import wandb
from pathlib import Path
import numpy as np
import torch

from src.datasets import get_dataloaders
from src.models import get_model
from src.losses import get_loss
from src.helpers import read_config, get_models_list, get_optimizer_scheduler


class TrainExperiment:

    experiment_name = "train"
    wandb_offline = False

    def __init__(
            self, 
            data_config, 
            loss_config, 
            model_config, 
            training_config,
            wandb_key, 
            device='cuda', 
            parallel=False,
            seed=42,
            continue_training=False,
        ) -> None:
         
        self.config_file = data_config+"-"+loss_config+"-"+model_config+"-"+training_config
        self.wandb_key = wandb_key
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.parallel = parallel
        self.seed = seed
        self.ini_epoch = self.load_last_epoch() if continue_training else 1

        root = Path(__file__).resolve().parents[2]
        self.cfg = {
            "data": read_config(os.path.join(root, "configs", "data", data_config)),
            "loss": read_config(os.path.join(root, "configs", "losses", loss_config)),
            "model": read_config(os.path.join(root, "configs", "models", model_config)),
            "training": read_config(os.path.join(root, "configs", "training", training_config))
        }

        self.results_dir = os.path.join(root, "results", self.config_file)
        self.model_dir = os.path.join(self.results_dir, "models")
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.preds_dir = os.path.join(self.results_dir, "predictions")
        Path(self.preds_dir).mkdir(parents=True, exist_ok=True)

        self.constraints_dir = os.path.join(root, "constraints")
        
        self.paired = 'paired' in self.cfg['training'] and self.cfg['training']['paired']

        self._set_all()


    #==========Setters==========
    def _set_wandb(self):
        now = datetime.now()
        wandb.login(key=self.wandb_key)
        self.wandb_run = wandb.init(
            dir=self.results_dir,
            config=self.cfg,
            project='dependent_disentanglement',
            group=self.experiment_name,
            mode="offline" if self.wandb_offline else "online",
            name="{experiment_name}_{config_file}_{seed}_{date}".format(
                experiment_name=self.experiment_name,
                config_file=self.config_file,
                seed=self.seed,
                date=now.strftime("%m-%d_%H:%M"),
            ),
        )

    def _set_loaders(self):
        self.cfg['data']['batch_size'] = self.cfg['training']['batch_size']
        self.train_loader, _ = get_dataloaders(
            device=self.device, shuffle=True, return_pairs=self.paired, **self.cfg['data'])

    def _set_model(self):
        self.cfg['model']['img_size'] = self.train_loader.dataset.img_size
        self.model = get_model(device=self.device, **self.cfg['model'])

    def _set_loss(self):
        self.loss_f = get_loss(
            device=self.device, n_data=len(self.train_loader.dataset), **self.cfg['loss'])

    def _set_optimizer(self):
        self.cfg['training']['optimizer']['batch_size'] = self.cfg['training']['batch_size']
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            params=self.model.parameters(),
            cfg_optimizer=self.cfg['training']['optimizer'], 
            cfg_scheduler=self.cfg['training']['scheduler'],
        )

    def _set_all(self):
        self._set_wandb()
        self._set_loaders()
        self._set_model()
        self._set_loss()
        self._set_optimizer()
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)


    #==========Train==========
    def _train_step(self, x):
        x = x.to(self.device)

        if self.loss_f.mode == 'post_forward':
            model_out = self.model(x)
            inputs = {'data': x, 'is_train': self.model.training, **model_out}
        elif self.loss_f.mode == 'pre_forward':
            inputs = {'data': x, 'model': self.model, 'is_train': self.model.training}
        elif self.loss_f.mode == 'optimizes_internally':
            inputs = {'data': x, 'model': self.model, 'optimizer': self.optimizer}

        loss_out = self.loss_f(**inputs)
        if not self.loss_f.mode=='optimizes_internally':
            self.optimizer.zero_grad()
            loss_out['loss'].backward()
            self.optimizer.step()
        return loss_out['to_log']
    

    def _train_step_paired(self, x, paired_x, shared_idcs=None):
        x = x.to(self.device)
        paired_x = paired_x.to(self.device)

        if self.loss_f.mode == 'post_forward':
            model_out = self.model(x)
            paired_model_out = self.model(paired_x)
            paired_model_out = {
                f'paired_{key}': item for key, item in paired_model_out.items()}
            inputs = {
                'data': x, 'paired_data': paired_x, 'shared_idcs': shared_idcs,
                'is_train': self.model.training, **model_out, **paired_model_out}
        elif self.loss_f.mode == 'pre_forward':
            inputs = {
                'data': x, 'paired_data': paired_x, 'shared_idcs': shared_idcs,
                'model': self.model, 'is_train': self.model.training}
        elif self.loss_f.mode == 'optimizes_internally':
            inputs = {'data': x, 'model': self.model, 'optimizer': self.optimizer}

        loss_out = self.loss_f(**inputs)
        if not self.loss_f.mode=='optimizes_internally':
            self.optimizer.zero_grad()
            loss_out['loss'].backward()
            self.optimizer.step()
        return loss_out['to_log']
    

    def train_epoch(self, epoch):
        self.model.train()
        losses_epoch = collections.defaultdict(list)
        for input_step in self.train_loader:
            if self.paired:
                losses_step = self._train_step_paired(
                    input_step[0][0],input_step[0][0])
            else:
                losses_step = self._train_step(input_step[0])
            for key, value in losses_step.items():
                losses_epoch[key].append(value)
        self.scheduler.step()
        if self.save:
            self.save_epoch(epoch)
        return {"losses_train/"+key: np.mean(item) for key, item in losses_epoch.items()}
    

    #==========Test==========
    def test_epoch(self, epoch):
        labels_preds = {'y': [], 'z': []}
        self.model.eval()
        for input_step in self.train_loader:
            if self.paired:
                x, y = input_step[0][0], input_step[1][0]
            else:
                x, y = input_step[0], input_step[1]
            output = self.model(x.to(self.device))
            labels_preds['y'].extend(y.cpu())
            labels_preds['z'].extend(output['stats_qzx'][:,:,0].cpu())
        labels_preds = {k: torch.stack(v) for k, v in labels_preds.items()}
        torch.save(labels_preds, os.path.join(self.preds_dir, "epoch_{}.pth".format(epoch)))

    #==========Run==========
    def run(self):
        for epoch in range(self.ini_epoch, self.cfg['training']['n_epochs']+1):
            self.save = epoch%self.cfg['training']['save_epochs']==0 or\
                epoch==self.cfg['training']['n_epochs']
            losses_train = self.train_epoch(epoch)
            if self.save:
                with torch.no_grad():
                    self.test_epoch(epoch)
            log_dict = {**losses_train}
            self.wandb_run.log(log_dict)
            print('\n'.join(["{}:\t{}".format(k, v) for k, v in log_dict.items()])+'\n')


    #==========Save and Load==========
    def save_epoch(self, epoch):
        epoch_path = os.path.join(self.model_dir, "epoch_"+str(epoch)+".pt")
        model_state_dict = self.model.state_dict()
        checkpoint = {
            'epoch':    epoch,
            'model':    model_state_dict,
        }
        torch.save(checkpoint, epoch_path)

    def load_epoch(self, epoch):
        previous_models = get_models_list(self.model_dir, 'epoch_')
        assert 'epoch_'+str(epoch)+'.pt' in previous_models, "Selected epoch is not available"
        checkpoint = torch.load(
            os.path.join(self.model_dir, 'epoch_'+str(epoch)+'.pt'),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model'])

    def load_last_epoch(self, restart=False):
        previous_models = get_models_list(self.model_dir, 'epoch_')
        if len(previous_models)>0 and not restart:
            checkpoint = torch.load(
                os.path.join(self.model_dir, previous_models[-1]),
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint['model'])
            return checkpoint['epoch']
        else:
            return 0
        