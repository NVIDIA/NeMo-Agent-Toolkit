#!/bin/bash

set -e -o pipefail

if [[ -z "$NAT_CONFIG_FILE" ]]; then
  echo "NAT_CONFIG_FILE not set" >&2
  exit 1
fi

export NVIDIA_API_KEY=$(aws secretsmanager get-secret-value --secret-id 'nvidia-api-credentials' \
                     --region $AWS_DEFAULT_REGION --query SecretString --output text | jq -r '.NVIDIA_API_KEY')

exec opentelemetry-instrument nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0