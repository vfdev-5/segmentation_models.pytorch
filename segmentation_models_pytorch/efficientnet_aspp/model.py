from torch import nn
from torch.nn import functional as F

from ..encoders import get_encoder
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from collections import OrderedDict


class EfficientnetASPP(nn.Module):

    def __init__(
            self,
            encoder_name='efficientnet-b2',
            encoder_weights='imagenet',
            classes=1):

        super().__init__()

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        out_shapes = encoder.out_shapes

        base_model = SimpleSegmentationModel

        backbone = encoder
        classifier = DeepLabHead(out_shapes[0], classes)
        aux_classifier = None

        self.model = base_model(backbone, classifier, aux_classifier)

    def forward(self, x):
        return self.model(x)


class SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features[0]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result