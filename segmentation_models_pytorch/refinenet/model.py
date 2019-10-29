from .decoder import RefineNetDecoder
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
    Returns:
        ``torch.nn.Module``: **RefineNet**

    Notes:
        Decoder uses improved chained residual pool architecture, see details_.

    .. _RefineNet:
        https://arxiv.org/abs/1611.06612

    .. _details:
        https://github.com/guosheng/refinenet#network-architecture-and-implementation

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            num_refine_channels=256,
            classes=1,
            activation='sigmoid',
            output_upsampling_factor=None,
            ignore_stem_output=True
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        out_shapes = encoder.out_shapes
        if ignore_stem_output:
            out_shapes = out_shapes[:-1]  # in paper, x0 output (size // 2) of resnet is ignored

        if output_upsampling_factor is None:
            output_upsampling_factor = 4.0 if ignore_stem_output else 2.0

        decoder = RefineNetDecoder(
            encoder_channels=out_shapes,
            num_refine_channels=num_refine_channels,
            final_channels=classes,
            output_upsampling_factor=output_upsampling_factor
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
