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
"""Backbones package definition."""

from models.official.vision.beta.modeling.backbones.efficientnet import EfficientNet
from models.official.vision.beta.modeling.backbones.mobiledet import MobileDet
from models.official.vision.beta.modeling.backbones.mobilenet import MobileNet
from models.official.vision.beta.modeling.backbones.resnet import ResNet
from models.official.vision.beta.modeling.backbones.resnet_3d import ResNet3D
from models.official.vision.beta.modeling.backbones.resnet_deeplab import DilatedResNet
from models.official.vision.beta.modeling.backbones.revnet import RevNet
from models.official.vision.beta.modeling.backbones.spinenet import SpineNet
from models.official.vision.beta.modeling.backbones.spinenet_mobile import SpineNetMobile
