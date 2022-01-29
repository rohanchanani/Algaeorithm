# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Layers package definition."""

from models.official.vision.beta.modeling.layers.box_sampler import BoxSampler
from models.official.vision.beta.modeling.layers.detection_generator import DetectionGenerator
from models.official.vision.beta.modeling.layers.detection_generator import MultilevelDetectionGenerator
from models.official.vision.beta.modeling.layers.mask_sampler import MaskSampler
from models.official.vision.beta.modeling.layers.nn_blocks import BottleneckBlock
from models.official.vision.beta.modeling.layers.nn_blocks import BottleneckResidualInner
from models.official.vision.beta.modeling.layers.nn_blocks import DepthwiseSeparableConvBlock
from models.official.vision.beta.modeling.layers.nn_blocks import InvertedBottleneckBlock
from models.official.vision.beta.modeling.layers.nn_blocks import ResidualBlock
from models.official.vision.beta.modeling.layers.nn_blocks import ResidualInner
from models.official.vision.beta.modeling.layers.nn_blocks import ReversibleLayer
from models.official.vision.beta.modeling.layers.nn_blocks_3d import BottleneckBlock3D
from models.official.vision.beta.modeling.layers.nn_blocks_3d import SelfGating
from models.official.vision.beta.modeling.layers.nn_layers import CausalConvMixin
from models.official.vision.beta.modeling.layers.nn_layers import Conv2D
from models.official.vision.beta.modeling.layers.nn_layers import Conv3D
from models.official.vision.beta.modeling.layers.nn_layers import DepthwiseConv2D
from models.official.vision.beta.modeling.layers.nn_layers import GlobalAveragePool3D
from models.official.vision.beta.modeling.layers.nn_layers import PositionalEncoding
from models.official.vision.beta.modeling.layers.nn_layers import Scale
from models.official.vision.beta.modeling.layers.nn_layers import SpatialAveragePool3D
from models.official.vision.beta.modeling.layers.nn_layers import SqueezeExcitation
from models.official.vision.beta.modeling.layers.nn_layers import StochasticDepth
from models.official.vision.beta.modeling.layers.nn_layers import TemporalSoftmaxPool
from models.official.vision.beta.modeling.layers.roi_aligner import MultilevelROIAligner
from models.official.vision.beta.modeling.layers.roi_generator import MultilevelROIGenerator
from models.official.vision.beta.modeling.layers.roi_sampler import ROISampler
