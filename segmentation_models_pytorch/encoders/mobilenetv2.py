from torchvision.models.mobilenet import MobileNetV2, model_urls


class MobileNetV2Encoder(MobileNetV2):
    """MobileNetV2 encoder based on torchvision's implementation

    Intermediate 5 outputs are based on Light-Weight RefineNet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.classifier

    def _forward(self, x):
        x = self.features[0](x)
        x0 = self.features[1](x)

        x1 = self.features[2:4](x0)
        x2 = self.features[4:7](x1)
        x3 = self.features[7:11](x2)
        x4 = self.features[11:14](x3)
        x5 = self.features[14:17](x4)
        x6 = self.features[17](x5)
        return x6, x5, x4, x3, x2, x1, x0

    def forward(self, x):
        x6, x5, x4, x3, x2, x1, x0 = self._forward(x)
        return [x6, x4, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('classifier.1.bias')
        state_dict.pop('classifier.1.weight')
        super().load_state_dict(state_dict, **kwargs)


class MobileNetV2EncoderExtendedOutput(MobileNetV2Encoder):
    """MobileNetV2 encoder based on torchvision's implementation

    Intermediate 6 outputs are based on Light-Weight RefineNet and
    can be used to implement Light-Weight RefineNet segmentation model.
    """
    def forward(self, x):
        x6, x5, x4, x3, x2, x1, x0 = self._forward(x)
        return [x6, x5, x4, x3, x2, x1]


def _get_pretrained_settings():
    pretrained_settings = {
        'imagenet': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'url': model_urls['mobilenet_v2'],
            'input_space': 'RGB',
            'input_range': [0, 1]
        }
    }
    return pretrained_settings


mobilenetv2_encoders = {
    'mobilenetv2': {
        'encoder': MobileNetV2Encoder,
        'pretrained_settings': _get_pretrained_settings(),
        'out_shapes': (320, 96, 32, 24, 16),
        'params': {},
    },
    'mobilenetv2_extended_output': {
        'encoder': MobileNetV2EncoderExtendedOutput,
        'pretrained_settings': _get_pretrained_settings(),
        'out_shapes': (320, 160, 96, 64, 32, 24),
        'params': {},
    },

}
