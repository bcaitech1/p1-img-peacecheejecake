from stage1.modules.models.conv import EfficientNetWithMultihead
from .basic import BasicModel

MODEL_DICT = {
    "BasicModel": (BasicModel, [], {"name": "model"}),
    "EfficientNetWithMultihead": (EfficientNetWithMultihead, ['efficientnet-b0', False], {}),
    "EfficientNetWithMultihead-pretrained": (EfficientNetWithMultihead, ['efficientnet-b0', True], {}),
}