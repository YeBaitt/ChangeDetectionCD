# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
from torch import Tensor

from opencd.registry import MODELS
from .siamencoder_decoder import SiamEncoderDecoder


@MODELS.register_module()
class DIEncoderDecoder(SiamEncoderDecoder):
    """Dual Input Encoder Decoder segmentors.

    DIEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        x = self.backbone(img_from, img_to)  # x是一个tuple，含有四元素，分别是维度为
        #  (batch, 128, 128, 128)、(batch, 256, 64, 64)、(batch, 512, 32, 32)、(batch, 1024, 16, 16)的tensor
        if self.with_neck:
            x = self.neck(x)  # FPN的每层通道数是相同的，一般是256，这样加入neck后extract_feat()的输出维度就变了
        return x
