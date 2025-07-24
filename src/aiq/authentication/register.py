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

# pylint: disable=unused-import
# flake8: noqa
# isort:skip_file

# Import any providers which need to be automatically registered here
#from .oauth2.auth_code_grant_config import oauth2_authorization_code_grant
#from .api_key.api_key_config import api_key

#from aiq.authentication.oauth2.register import oauth2_authorization_code_grant_client
#from aiq.authentication.api_key.register import api_key_client

from aiq.authentication.http_basic_auth import register
