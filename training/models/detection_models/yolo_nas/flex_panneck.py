from typing import Union, List, Tuple

from omegaconf import DictConfig
from torch import Tensor
import torch.nn as nn

from super_gradients.common.registry import register_detection_module
from super_gradients.modules.detection_modules import BaseDetectionModule
from super_gradients.training.utils.utils import HpmStruct
import super_gradients.common.factories.detection_modules_factory as det_factory


@register_detection_module()
class FlexYoloNASPANNeckWithC2(BaseDetectionModule):
    """
    A PAN (path aggregation network) neck with 4 stages (2 up-sampling and 2 down-sampling stages)
    where the up-sampling stages include a higher resolution skip
    Returns outputs of neck stage 2, stage 3, stage 4
    """

    def __init__(
        self,
        in_channels: List[int],
        up_necks: List[Union[str, HpmStruct, DictConfig]],
        down_necks: List[Union[str, HpmStruct, DictConfig]],
    ):
        """
        Initialize the PAN neck
        :param in_channels: Input channels of the 4 feature maps from the backbone
        """
        super().__init__(in_channels)
        factory = det_factory.DetectionModulesFactory()

        # Up necks init
        self.up_necks = nn.ModuleList()
        for i, up_nack in enumerate(up_necks):
            neck_in_channels = [
                in_channels[-1] if i == 0 else self.up_necks[i - 1].out_channels[1],  # to upsample
                in_channels[-(2 + i)],  # same resolution skip
                in_channels[-(2 + i + 1)],  # to downsample
            ]
            self.up_necks.append(factory.get(factory.insert_module_param(up_nack, "in_channels", neck_in_channels)))

        # Down necks init
        self.down_necks = nn.ModuleList()
        for i, down_neck in enumerate(down_necks):
            neck_in_channels = [
                self.up_necks[-1].out_channels[1] if i == 0 else self.down_necks[i - 1].out_channels,  # to downsample
                self.up_necks[-(i + 1)].out_channels[0],
            ]
            self.down_necks.append(factory.get(factory.insert_module_param(down_neck, "in_channels", neck_in_channels)))

        self._out_channels = [self.up_necks[-1].out_channels[1], *[_down_neck.out_channels for _down_neck in self.down_necks]]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        # Up necks forward
        x_inters = []
        x = inputs[-1]
        for i, up_neck in enumerate(self.up_necks):
            x_inter, x = up_neck([x, inputs[-(2 + i)], inputs[-(2 + i + 1)]])
            x_inters.append(x_inter)

        # Up necks forward
        outputs: List[Tensor] = [x]
        for i, down_neck in enumerate(self.down_necks):
            x = down_neck([x, x_inters[-(i + 1)]])
            outputs.append(x)

        return tuple(outputs)
