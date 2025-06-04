import numpy as np
import torch
from torch import Tensor

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def calc_accuracy(preds: Tensor, labels: Tensor):
    if preds.shape[-1]==1:
        accuracy = ((preds[:,0] >= 0.5).float() == labels).float().mean() * 100
    else:
        predicted_labels = torch.argmax(preds, dim=1)
        correct = (predicted_labels == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
    return accuracy


def get_results_classifier_sklearn(
        x_train: Tensor, 
        y_train: Tensor,
        x_test: Tensor, 
        y_test: Tensor,
        n_layers: int=2,
        max_iter=50,
        **kwargs
    ):
    x_train, y_train = x_train.numpy(), y_train.numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()
    n_classes = int(y_train.max()+1)
    if n_layers>0: 
        x_dim, y_dim = x_train.shape[-1], int(y_train.max()+1)
        hidden_sizes = tuple(np.linspace(x_dim, y_dim, n_layers+2, dtype=int)[1:-1])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = MLPClassifier(
            hidden_layer_sizes=hidden_sizes, 
            activation='relu', 
            solver='adam', 
            max_iter=max_iter, 
            batch_size=256,
            random_state=42
        )
    else:
        model = LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=max_iter
        )
    model.fit(x_train, y_train)
    class_pred = model.predict(x_test)
    y_pred = model.predict_proba(x_test)
    y_pred[y_pred == 0] = 1e-6 
    probs_class = np.bincount(y_train.astype(int), minlength=n_classes) / len(y_train)
    entropy = -(probs_class * np.log(probs_class)).sum()
    cond_entropy = -(y_pred * np.log(y_pred)).sum(axis=1).mean()
    del model
    return {
        'accuracy': accuracy_score(y_test, class_pred) * 100,
        'mi': 1 - cond_entropy / entropy,
    }