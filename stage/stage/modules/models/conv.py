from .basic import *
from .classifier import MultiheadClassifier

import torchvision.models.resnet as resnet
import efficientnet_pytorch



class PureResNet(BasicModel):
    resnet_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }

    resnet_block_layers = {
        'resnet18': (resnet.BasicBlock, [2, 2, 2, 2]),
        'resnet34': (resnet.BasicBlock, [3, 4, 6, 3]),
        'resnet50': (resnet.Bottleneck, [3, 4, 6, 3]),
        'resnet101': (resnet.Bottleneck, [3, 4, 23, 3]),
        'resnet152': (resnet.Bottleneck, [3, 8, 36, 3]),
        'resnetxt50_32x4d': (resnet.Bottleneck, [3, 4, 6, 3]),
        'resnetxt101_32x8d': (resnet.Bottleneck, [3, 4, 23, 3]),
        'wide_resnet50_2': (resnet.Bottleneck, [3, 4, 6, 3]),
        'wide_resnet101_2': (resnet.Bottleneck, [3, 4, 23, 3])
    }

    def __init__(self, name='resnet18', pretrained=True, num_classes=18, freezed_conv=False):
        super(PureResNet, self).__init__()
        resnet_ = resnet._resnet(name, *PureResNet.resnet_block_layers[name], pretrained, progress=True)
        # if pretrained:
        #     state_dict = load_state_dict_from_url(PureResNet.resnet_urls[name], progress=True)
        #     resnet_.load_state_dict(state_dict)
        self.backbone = BasicSequential('backbone')
        for name, child in resnet_.named_children():
            if 'fc' in name:
                fc_in_features = child.in_features
            else:
                self.backbone.add_module(name, child)
    
        if freezed_conv:
            self.backbone.requires_grad(self.backbone.parameters(), False)

        self.fc = nn.Linear(fc_in_features, num_classes, bias=True)

        self.name = name

        
    def _forward_impl(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)

        return x



class SimpleConvNet(BasicModel):
    def __init__(self, name='simple_conv'):
        super().__init__(name)
        self.organize()
        
    
    def organize(self):
        self.conv0 = nn.Conv2d(3, 32, (3, 3), 1, 1)
        self.conv1 = nn.Conv2d(32, 64, (3, 3), 2, 1)
        self.block0 = SimpleResidualBlock(32, 64)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), 2, 1)
        self.block1 = SimpleResidualBlock(64, 128) * 2
        self.conv3 = nn.Conv2d(128, 256, (3, 3), 2, 1)
        self.block2 = SimpleResidualBlock(128, 256) * 8
        self.conv4 = nn.Conv2d(256, 512, (3, 3), 2, 1)
        self.block3 = SimpleResidualBlock(256, 512) * 8
        self.conv5 = nn.Conv2d(512, 1024, (3, 3), 2, 1)
        self.block4 = SimpleResidualBlock(512, 1024) * 4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(1024, 18)
        
        self.leaky_relu = nn.LeakyReLU()


    def _forward_impl(self, x):
        x = self.conv0(x)
        x = self.leaky_relu(x)
        
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.block0(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        
        x = self.block1(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)

        x = self.block2(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.block3(x)
        
        x = self.conv5(x)
        x = self.leaky_relu(x)

        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(-1, self.fc.in_features)

        x = self.fc(x)
        
        return x


class SimpleResidualBlock(nn.Module):
    def __init__(
        self,
        conv1_filters: int,
        conv2_filters: int,
        repeat: int = 1
    ):
        super(SimpleResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(conv2_filters, conv1_filters, (1, 1), 1, 0)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, (3, 3), 1, 1)
        self.leaky_relu = nn.LeakyReLU()

    
    def forward(self, x):
        y = self.conv1(x)
        y = self.leaky_relu(y)
        y = self.conv2(y)
        y = self.leaky_relu(y)

        return x + y


    def __mul__(self, num: int):
        block = nn.Sequential()
        for layer_num in range(num):
            block.add_module(f"layer{layer_num}", self)
        
        return block




class EfficientNetWithMultihead(BasicModel):
    def __init__(self, arc: str, pretrained: bool = True, label_weight_for_output: int = 0.5, name = 'multihead_effnet'):
        super(EfficientNetWithMultihead, self).__init__()

        if pretrained:
            self.backbone = efficientnet_pytorch.EfficientNet.from_pretrained(arc)
        else:
            self.backbone = efficientnet_pytorch.EfficientNet.from_name(arc)

        self.classifier = MultiheadClassifier(
            in_features=self.backbone._fc.out_features, 
            label_weight=label_weight_for_output,
            out_features=18,
            name=f"multihead_{arc}"
        )

        self.model = nn.Sequential(
            self.backbone,
            nn.Dropout(),
            self.classifier
        )

        self.name = name


    def _forward_impl(self, x):
        x = self.model(x)
        return x

