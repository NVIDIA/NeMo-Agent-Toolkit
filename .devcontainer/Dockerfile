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



ARG BASE
ARG PYTHON_PACKAGE_MANAGER

FROM node:22 as node

FROM ${BASE} as base

# ===== install common packages ==================================================================

RUN <<EOF

set -x

apt-get update;

rm -rf /tmp/*;
rm -rf /var/tmp/*;
rm -rf /var/cache/apt/*;
rm -rf /var/lib/apt/lists/*;

EOF

# # ===== install node + camouflage ================================================================

# COPY --from=node /usr/local/bin/node /usr/local/bin/node
# COPY --from=node /usr/local/include/node /usr/local/include/node
# COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
# COPY --from=node /opt/yarn-v*/bin/* /usr/local/bin/
# COPY --from=node /opt/yarn-v*/lib/* /usr/local/lib/

# RUN \
# <<EOF

# bash -c 'echo -e "
# fund=false\n
# audit=false\n
# save-prefix=\n
# --omit=optional\n
# save-exact=true\n
# package-lock=false\n
# update-notifier=false\n
# scripts-prepend-node-path=true\n
# registry=https://registry.npmjs.org/\n
# " | tee /root/.npmrc >/dev/null'

# ln -s /usr/local/bin/node /usr/local/bin/nodejs
# ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
# ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# echo "node version: $(node --version)"
# echo " npm version: $(npm --version)"
# echo "yarn version: $(yarn --version)"

# npm install -g camouflage-server@0.15

# EOF


FROM base as uv-base

FROM ${PYTHON_PACKAGE_MANAGER}-base

ENV PROJECT_MANIFEST_YML="/home/coder/dev/ai-query-engine/manifest.yaml"
ENV PATH="${PATH}:/home/coder/dev/ai-query-engine/.devcontainer/bin"

ARG CUDA
ENV CUDAARCHS="RAPIDS"
ENV CUDA_VERSION="${CUDA_VERSION:-${CUDA}}"

ARG RAPIDS
ENV RAPIDS=${RAPIDS}

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV AWS_ROLE_ARN="arn:aws:iam::279114543810:role/nv-gha-token-sccache-devs"
ENV HISTFILE="/home/coder/.cache/._bash_history"
