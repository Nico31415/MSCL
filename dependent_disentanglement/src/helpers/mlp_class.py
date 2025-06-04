import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset
from torch.autograd import Variable

from src.helpers.training import get_optimizer, get_loader
from .metrics import calc_accuracy


class MLPClassifier(nn.Module):

    def __init__(
            self, 
            hidden_layer_sizes=None,
            device=torch.device('cpu'),
            parallel=True,
        ) -> None:
        super().__init__()
        self.hls = hidden_layer_sizes
        self.device = device
        self.parallel=parallel

    #==========Setters==========
    def _set_loss(self):
        self.loss_fn = SqueezeBCEWithLogitsLoss() \
            if self.n_classes==2 else nn.CrossEntropyLoss()

    def _set_classifier(self):
        output_dim = 1 if self.n_classes==2 else self.n_classes
        if self.hls is None:
            layers = [nn.Linear(self.input_dim, output_dim)]
        else:
            layers = [
                nn.Linear(self.input_dim, self.hls[0]),
                nn.BatchNorm1d(self.hls[0]),
                nn.ReLU()
            ]
            for i in range(len(self.hls)-1):
                layers.append(nn.Linear(self.hls[i], self.hls[i+1]))
                layers.append(nn.BatchNorm1d(self.hls[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hls[-1], output_dim))
        self.classifier = nn.Sequential(*layers)
        self.classifier.to(device=self.device)

    def _set_optimizer(self):
        self.optimizer = get_optimizer(
            params=self.classifier.parameters(),
            optimizer=self.solver,
            base_lr=self.learning_rate,
            base_batch_size=self.batch_size,
        )
    
    #==========Fit==========
    def _train_step(self, x: Tensor, y: Tensor) -> Tensor:
        outputs = self.classifier(x)
        loss = self.loss_fn(outputs, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(
            self, 
            x_train: Tensor, 
            y_train: Tensor,
            solver='adam',
            batch_size=256,
            learning_rate=0.001,
            n_epochs=20,
        ):
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.input_dim = x_train.shape[-1]
        self.n_classes = int(y_train.max()+1)
        self._set_loss()
        self._set_classifier()
        self._set_optimizer()
        if self.parallel:
            self.classifier = nn.DataParallel(self.classifier)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader, _ = get_loader(
            train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=0)
        self.classifier.train()
        for _ in range(self.n_epochs):
            for x, y in train_loader:
                self._train_step(*self._prepare_inputs(x, y))

    #==========Get Accuracy==========
    def get_results(self, x_test, y_test):
        test_dataset = TensorDataset(x_test, y_test)
        test_loader, _ = get_loader(test_dataset, shuffle=False, batch_size=1024)
        self.classifier.eval()
        outputs, ys = [], []
        for x, y in test_loader:
            x, y = self._prepare_inputs(x, y)
            ys.extend(y.cpu().detach())
            outputs.extend(self.classifier(x).cpu().detach())
        ys, outputs = torch.stack(ys), torch.stack(outputs)
        if self.n_classes==2:
            probs = ys.sum() / len(ys)
            probs_preds = torch.sigmoid(outputs)
            entropy = -(probs * torch.log(probs) + (1-probs) * torch.log(1-probs))
            cond_entropy = -(probs_preds * torch.log(probs_preds)).mean()
            cross_entropy = 0
        else:
            probs_class = torch.bincount(ys, minlength=self.n_classes) / len(ys)
            probs_preds = torch.softmax(outputs, dim=-1)
            entropy = -(probs_class * torch.log(probs_class)).sum()
            cond_entropy = -(probs_preds * torch.log(probs_preds)).sum(axis=1).mean()
            cross_entropy = -(torch.log(
                probs_preds[torch.arange(len(ys)), ys.long()])).mean()
        return {
            'accuracy':  calc_accuracy(outputs, y_test),
            'mi':       1 - cond_entropy / entropy,
            'ce':       cross_entropy,
        }
    
    #==========Miscellanea==========
    def _prepare_inputs(self, x, y):
        x = Variable(x).to(device=self.device, dtype=torch.float, non_blocking=True)
        if self.n_classes==2:
            y = y.to(device=self.device, non_blocking=True)
        else:
            y = y.to(device=self.device, non_blocking=True, dtype=torch.long)
        return x, y

    def parallelize(self):
        self.encoder1 = nn.DataParallel(self.encoder1)
        self.parallel = True



class SqueezeBCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input.squeeze(1), target)
    


def get_results_classifier(x_train, y_train, x_test, y_test, device, n_layers=2):
    x_dim, y_dim = x_train.shape[-1], int(y_train.max()+1)
    hidden_sizes = np.linspace(x_dim, y_dim, n_layers + 2, dtype=int)[1:-1].tolist()
    model = MLPClassifier(hidden_sizes, device=device)
    model.fit(x_train, y_train, n_epochs=10), 
    return model.get_results(x_test, y_test)