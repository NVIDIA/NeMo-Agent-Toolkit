# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from dataclasses import dataclass


@dataclass
class LocalAuthenticationOptions:
    bearer_token: str | None
    env_id: str

    @classmethod
    def from_environment(cls) -> "LocalAuthenticationOptions":
        return cls(bearer_token=os.getenv("BEARER_TOKEN"), env_id=os.getenv("ENVIRONMENT_ID", "prod"))
