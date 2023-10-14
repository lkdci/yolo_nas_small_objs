import copy
from typing import Union

from omegaconf import DictConfig

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry import register_model
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils import get_param
from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.models.detection_models.yolo_nas.yolo_nas_variants import YoloNAS

logger = get_logger(__name__)


@register_model("yolo_nas_s_s4")
class YoloNAS_S_S4(YoloNAS):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_s_s4_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> PPYoloEPostPredictionCallback:
        return PPYoloEPostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @property
    def num_classes(self):
        return self.heads.num_classes
