from .decoder import RefineNetDecoder, LightWeightRefineNetDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class RefineNet(EncoderDecoder):
    """RefineNet_ is a fully convolution neural network for image semantic segmentation:
    'RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation'.

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        output_upsampling_factor: upsampling factor used to interpolate the output in `RefineNetDecoder`. By default
            it is chosen as 4.0 according to the paper. If `ignore_stem_output` is False, value of
            `output_upsampling_factor` should be 2.0. To ignore final interpolation,
            set `output_upsampling_factor=None`.
        upsampling_config: upsampling configuration dictionary, e.g. `{'mode': 'nearest', 'align_corners': None}`
        ignore_stem_output: flag to ignore the first output feature map from network stem, with size equal
            `input_size / 2`. In the paper, this feature map is ignored.

    Returns:
        ``torch.nn.Module``: **RefineNet**

    Notes:
        Decoder uses improved chained residual pool architecture,
        see https://github.com/guosheng/refinenet#network-architecture-and-implementation

    .. _RefineNet:
        https://arxiv.org/abs/1611.06612

    """

    def __init__(
            self,
            encoder_name='resnet101',
            encoder_weights='imagenet',
            num_refine_channels=256,
            classes=1,
            activation='sigmoid',
            output_upsampling_factor=4.0,
            upsampling_config=None,
            ignore_stem_output=True):

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        out_shapes = encoder.out_shapes
        if ignore_stem_output:
            out_shapes = out_shapes[:-1]  # in paper, x0 output (size // 2) of resnet is ignored

        decoder = RefineNetDecoder(
            encoder_channels=out_shapes,
            num_refine_channels=num_refine_channels,
            final_channels=classes,
            output_upsampling_factor=output_upsampling_factor,
            upsampling_config=upsampling_config
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'refine-{}'.format(encoder_name)
        self.ignore_stem_output = ignore_stem_output

    def forward(self, x):
        x = self.encoder(x)
        if self.ignore_stem_output:
            x = x[:-1]
        x = self.decoder(x)
        return x


class LightWeightRefineNet(EncoderDecoder):
    """Light-Weight-RefineNet_ is a fully convolution neural network for image semantic segmentation:
    'Light-Weight RefineNet forReal-Time Semantic Segmentation'.

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        output_upsampling_factor: upsampling factor used to interpolate the output in `RefineNetDecoder`. By default
            it is chosen as 4.0 according to the paper. If `ignore_stem_output` is False, value of
            `output_upsampling_factor` should be 2.0. To ignore final interpolation,
            set `output_upsampling_factor=None`.
        upsampling_config: upsampling configuration dictionary, e.g. `{'mode': 'nearest', 'align_corners': None}`

    Returns:
        ``torch.nn.Module``: **LightWeightRefineNet**

    .. _Light-Weight-RefineNet:
        https://arxiv.org/abs/1810.03272

    """

    def __init__(
            self,
            encoder_name='mobilenetv2_extended_output',
            encoder_weights='imagenet',
            num_refine_channels=256,
            classes=1,
            activation='sigmoid',
            output_upsampling_factor=4.0,
            upsampling_config=None):

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        out_shapes = encoder.out_shapes

        decoder = LightWeightRefineNetDecoder(
            encoder_channels=out_shapes,
            num_refine_channels=num_refine_channels,
            final_channels=classes,
            output_upsampling_factor=output_upsampling_factor,
            upsampling_config=upsampling_config
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'lwrefine-{}'.format(encoder_name)
