# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -euo pipefail

# Usage: store_blobs.sh /path/to/dir db_name
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <db_user> <db_password> <local_directory> <bucket_name>"
    exit 1
fi

DB_USER="$1"
DB_PASS="$2"
DIR="$3"
DB=bucket_"$4"

# If input dir does not exist, exit
if [ ! -d "${DIR}" ]; then
    echo "Input directory ${DIR} does not exist"
    exit 1
fi

# Copy the dir to /tmp
cp -r "${DIR}" /tmp/
DIR=/tmp/$(basename "${DIR}")

MYSQL="mysql -u ${DB_USER} -p${DB_PASS}"

# Delete the database if it exists
${MYSQL} <<EOF
DROP DATABASE IF EXISTS \`${DB}\`;
EOF

# Create database and schema/tables
${MYSQL} <<EOF
CREATE DATABASE IF NOT EXISTS \`${DB}\` DEFAULT CHARACTER SET utf8mb4;
USE \`${DB}\`;

CREATE TABLE IF NOT EXISTS object_meta (
  id INT AUTO_INCREMENT PRIMARY KEY,
  path VARCHAR(768) NOT NULL UNIQUE,
  size BIGINT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS object_data (
  id INT PRIMARY KEY,
  data LONGBLOB NOT NULL,
  FOREIGN KEY (id) REFERENCES object_meta(id) ON DELETE CASCADE
) ENGINE=InnoDB ROW_FORMAT=DYNAMIC;
EOF

echo "Finished creating database and schema/tables"

echo "Starting to process files..."
# Process each file
find "$DIR" -type f | while IFS= read -r filepath; do
  relpath="${filepath#$DIR/}"
  filepath_abs=$(realpath "$filepath")

  echo "Processing file: ${relpath}"
  fsize=$(stat -c%s "$filepath")
  # Serialize the file
  python serialize_file.py "$filepath"
  serialized_file="${filepath}.pkl"

  ${MYSQL[@]} <<EOF
USE \`$DB\`;
START TRANSACTION;

INSERT INTO object_meta (path, size)
VALUES ('${relpath}', ${fsize})
ON DUPLICATE KEY UPDATE size=VALUES(size), created_at=CURRENT_TIMESTAMP;

SET @obj_id := (SELECT id FROM object_meta WHERE path='${relpath}' FOR UPDATE);

REPLACE INTO object_data (id, data)
VALUES (
  @obj_id,
  LOAD_FILE('${serialized_file}')
);

COMMIT;
EOF

  echo "Stored '${relpath}', ${fsize} bytes."
done