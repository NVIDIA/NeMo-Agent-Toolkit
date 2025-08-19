# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import importlib
import importlib.abc
import importlib.util
import warnings


class CompatFinder(importlib.abc.MetaPathFinder):

    def __init__(self, alias_prefix, target_prefix):
        self.alias_prefix = alias_prefix
        self.target_prefix = target_prefix

    def find_spec(self, fullname, path, target=None):  # pylint: disable=unused-argument
        if fullname == self.alias_prefix or fullname.startswith(self.alias_prefix + "."):
            # Map aiq.something -> nat.something
            target_name = self.target_prefix + fullname[len(self.alias_prefix):]
            spec = importlib.util.find_spec(target_name)
            if spec is None:
                return None
            # Wrap the loader so it loads under the alias name
            return importlib.util.spec_from_loader(fullname, CompatLoader(fullname, target_name))
        return None


class CompatLoader(importlib.abc.Loader):

    def __init__(self, alias_name, target_name):
        self.alias_name = alias_name
        self.target_name = target_name

    def create_module(self, spec):
        # Reuse the actual module so there's only one instance
        target_module = importlib.import_module(self.target_name)
        sys.modules[self.alias_name] = target_module
        return target_module

    def exec_module(self, module):
        # Nothing to execute since the target is already loaded
        pass


# Register the compatibility finder
sys.meta_path.insert(0, CompatFinder("aiq", "nat"))

warnings.warn(
    "!!! The 'aiq' namespace is deprecated and will be removed in a future release. "
    "Please use the 'nat' namespace instead.",
    DeprecationWarning,
    stacklevel=2,
)
