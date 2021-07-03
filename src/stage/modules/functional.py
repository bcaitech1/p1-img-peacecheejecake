# from scipy.stats import hmean

import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self._reset()

    def __call__(self, targets: torch.Tensor, predictions: torch.Tensor):
        assert targets.shape == predictions.shape
        
        input_size = targets.size(0)
        new_info = self._new_matrix()
        for input_idx in range(input_size):
            target = targets[input_idx].item()
            prediction = predictions[input_idx].item()
            if target == prediction:
                new_info[target, prediction] += 1
            else:
                new_info[target, prediction] += 1
                new_info[prediction, target] += 1

        self.matrix += new_info
                
    def __add__(self, value):
        if not isinstance(value, ConfusionMatrix):
            raise TypeError("Only ConfusionMatrix can be added to ConfusionMatrix.")

        self.matrix += value.matrix

        return self

    def _reset(self):
        self.matrix = self._new_matrix()

    def _new_matrix(self):
        return torch.zeros((self.num_classes, self.num_classes), dtype=torch.float)

    def recall(self):
        return (self.matrix / self.matrix.sum(dim=0)).diagonal().mean().item()

    def precision(self):
        return (self.matrix / self.matrix.sum(dim=1)).diagonal().mean().item()

    def f1_score(self):
        # return hmean([self.recall(), self.precision()])
        return 2 / (1 / self.recall() + 1 / self.precision())
