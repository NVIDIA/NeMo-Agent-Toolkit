# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PiiTestConfig(FunctionBaseConfig, name="pii_redaction_test"):
    """Configuration for PII redaction test function."""
    test_email: str = "test@example.com"
    test_phone: str = "555-123-4567"
    test_credit_card: str = "4111-1111-1111-1111"
    test_ssn: str = "352-01-1142"
    custom_secret: str = "sk-12jfw23jfwicn34213"


@register_function(config_type=PiiTestConfig)
async def test_pii_redaction(config: PiiTestConfig, builder: Builder):
    """Test function that demonstrates Weave PII redaction capabilities."""
    # Create sample user data with PII
    user_data = {
        "name": "John Doe",
        "email": config.test_email,
        "phone": config.test_phone,
        "payment": {
            "credit_card": config.test_credit_card, "ssn": config.test_ssn
        },
        "custom_secret": config.custom_secret
    }

    async def process_user_data(query: str) -> str:
        """Process user data and return results (will be traced with all PII)."""
        return user_data

    description = "This is a simple function that returns John Doe's data with personally identifiable information."

    yield FunctionInfo.from_fn(process_user_data, description=description)
