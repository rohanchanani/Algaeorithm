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

"""All necessary imports for registration."""

# pylint: disable=unused-import
from models.official.common import registry_imports
from models.official.vision.beta.projects.simclr.configs import simclr
from models.official.vision.beta.projects.simclr.losses import contrastive_losses
from models.official.vision.beta.projects.simclr.modeling import simclr_model
from models.official.vision.beta.projects.simclr.tasks import simclr as simclr_task