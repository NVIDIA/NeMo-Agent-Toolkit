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
'''
This is a temporary script that will be dropped once the eval dataset
download is enhanced to easily download the swebench datasets
'''
import json
import logging

import pandas as pd

# Define dataset paths
URL = 'hf://datasets/princeton-nlp/SWE-bench_Lite/'

DEV_DATASET = 'data/dev-00000-of-00001.parquet'
DEV_DATASET_DEST = './dev_dataset_lite.json'

TEST_DATASET = 'data/test-00000-of-00001.parquet'
TEST_DATASET_DEST = './test_dataset_lite.json'

SPLITS = {'dev': DEV_DATASET, 'test': TEST_DATASET}

# download subset
SUBSET_SIZE = 0

logger = logging.getLogger(__name__)


# Function to process a dataset: read, filter columns, extract first 5 rows, and write to JSON
def process_dataset(split_name, file_path, output_file, columns):
    try:
        # Read the dataset
        df = pd.read_parquet(URL + file_path)

        # Filter the specified columns
        df_filtered = df[columns] if columns else df

        # Extract subset if needed
        df_subset = df_filtered.head(SUBSET_SIZE) if SUBSET_SIZE else df_filtered

        # Write to JSON
        data = df_subset.to_dict(orient="records")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug("%s dataset written to '%s' with %s rows and selected columns.",
                     split_name,
                     output_file,
                     SUBSET_SIZE)
    except Exception as e:
        logger.exception("Error processing %s dataset: %s", split_name, e, exc_info=True)


# Columns to extract
columns_to_extract = ["repo", "instance_id", "base_commit", "problem_statement", "hints_text"]

# Process 'dev' dataset
process_dataset("dev", SPLITS["dev"], DEV_DATASET_DEST, None)

# Process 'test' dataset
process_dataset("test", SPLITS["test"], TEST_DATASET_DEST, None)
