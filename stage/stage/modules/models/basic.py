import torch.nn as nn

from typing import Iterable



class BasicModel(nn.Module):
    def __init__(self, name='model'):
        super(BasicModel, self).__init__()
        self.name = name
        
        
    def init(self):
        self.__init__()
        return self


    def requires_grad(self, mode=True):
        for parameter in self.parameters():
            parameter.requires_grad = mode


    def requires_grad(self, parameters: Iterable[nn.Parameter], mode=True):
        for parameter in parameters:
            parameter.requires_grad = mode

    
    def forward(self, x):
        if x.dim() == 5:
            batch, crop, channel, height, width = x.size()
            x = x.view(-1, channel, height, width)
            outputs = self._forward_impl(x)
            outputs = outputs.view(batch, crop, -1).mean(1)
            return outputs

        return self._forward_impl(x)


    def __str__(self):
        return self.__class__.__name__



class BasicSequential(BasicModel):
    def __init__(self, name='seq'):
        super(BasicSequential, self).__init__(name)


    def _forward_impl(self, x):
        for child in self.children():
            x = child(x)
        
        return x




def freeze(model, freeze_backbone, freeze_cl):
    for name, param in model.named_parameters():
        if freeze_backbone and not ('fc' in name or 'classifier' in name):
            param.requires_grad = False
        elif not ('fc' in name or 'classifier' in name):
            param.requires_grad = True
        
        if freeze_cl and ('fc' in name or 'classifier' in name):
            param.requires_grad = False
        elif 'fc' in name or 'classifier' in name:
            param.requires_grad = True


