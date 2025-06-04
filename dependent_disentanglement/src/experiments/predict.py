import os

import yaml
import torch

from src.datasets import get_dataloaders
from src.helpers import read_config
from src.utils import get_all_metrics
from .train import TrainExperiment


class PredictExperiment(TrainExperiment):

    experiment_name = "predict"
    wandb_offline = True

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

    #==========Setters==========
    def _set_correlations(self):
        dataset = self.cfg['data']['dataset']
        if dataset in ["mpi3d_real", "mpi3d_real_complex"]:
            dataset = "mpi3d"
        self.correlations = {k: v  for k, v in read_config(
            os.path.join(self.constraints_dir, "correlations")
        ).items() if (k.split("_")[0]==dataset) or k=="base"}

    def _set_loaders(self):
        self.cfg['data']['batch_size'] = 2048
        self.train_loader, _ = get_dataloaders(
            device=self.device, shuffle=True, return_pairs=self.paired, num_workers=12,
            **self.cfg['data'])

    def _set_all(self):
        self._set_loaders()
        self._set_model()
        self.load_last_epoch()
        self.model.eval()
        self._set_correlations()
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)

    #==========Test==========
    def get_loaders_correlation(self, correlations_filepath):
        self.cfg['data']['batch_size'] = 2048
        loader, _ = get_dataloaders(
            device=self.device, shuffle=True, return_pairs=self.paired, num_workers=12,
            correlations_filepath=correlations_filepath, **self.cfg['data'])
        return loader

    def predict_epoch(self, correlation):
        correlations_filepath = "{}/correlations.yaml:{}".format(
            self.constraints_dir, correlation)
        if correlation=="base":
            loader = self.train_loader
        else:
            loader = self.get_loaders_correlation(correlations_filepath)
        labels_preds = {'y': [], 'z': []}
        for i, input_step in enumerate(loader):
            print("{} - {}".format(i, len(loader)))
            if self.paired:
                x, y = input_step[0][0], input_step[1][0]
            else:
                x, y = input_step[0], input_step[1]
            output = self.model(x.to(self.device))
            labels_preds['y'].extend(y.cpu())
            labels_preds['z'].extend(output['stats_qzx'][:,:,0].cpu())
        labels_preds = {k: torch.stack(v) for k, v in labels_preds.items()}
        all_metrics = {k: v.item() for k, v in get_all_metrics(labels_preds).items()}
        with open(os.path.join(self.preds_dir, "{}.yaml".format(correlation)), 'w') as f:
            yaml.dump(all_metrics, f)
        torch.save(labels_preds, os.path.join(self.preds_dir, "{}.pth".format(correlation)))

    #==========Run==========
    def run(self):
        for correlation in self.correlations:
            with torch.no_grad():
                print(correlation)
                self.predict_epoch(correlation)