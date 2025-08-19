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

import base64
import json
import mimetypes
import sys

# Usage: python serialize_file.py <file_path>

if len(sys.argv) != 2:
    print("Usage: python serialize_file.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, "rb") as f:
    data = f.read()

payload = {
    "data": base64.b64encode(data).decode("ascii"),
    "content_type": mimetypes.guess_type(file_path)[0],
    "metadata": {},
}

with open(file_path + ".json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
