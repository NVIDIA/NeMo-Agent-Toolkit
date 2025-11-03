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

import os

import pytest


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "restore_environ")
async def test_full_workflow(oauth2_client_credentials: dict[str, str]):
    import urllib

    from nat.runtime.loader import load_config
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_simple_auth.ip_lookup import WhoAmIConfig

    print("OAuth2 Client Credentials:", oauth2_client_credentials)

    # Even though we set this later on the config object, the yaml won't validate without these env vars set
    os.environ.update({
        "NAT_OAUTH_CLIENT_ID": oauth2_client_credentials["id"],
        "NAT_OAUTH_CLIENT_SECRET": oauth2_client_credentials["secret"],
    })

    config_file = locate_example_config(WhoAmIConfig)
    config = load_config(config_file)

    oauth_url = oauth2_client_credentials["url"]
    allowed_origins = config.general.front_end.cors.allow_origins
    for (i, url) in enumerate(allowed_origins):
        if urllib.parse.urlparse(url).port == 5001:
            allowed_origins[i] = oauth_url

    config.authentication['test_auth_provider'].authorization_url = f"{oauth_url}/oauth/authorize"
    config.authentication['test_auth_provider'].token_url = f"{oauth_url}/oauth/token"
    config.authentication['test_auth_provider'].client_id = oauth2_client_credentials["id"]
    config.authentication['test_auth_provider'].client_secret = oauth2_client_credentials["secret"]

    await run_workflow(config=config,
                       question="Who am I logged in as?",
                       expected_answer=oauth2_client_credentials["username"])
