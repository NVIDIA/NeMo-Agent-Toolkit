import json

import boto3

client = boto3.client('bedrock-agentcore', region_name='<AWS_REGION>')
payload = json.dumps({"inputs": "How do I use the Strands Agents API?"})

response = client.invoke_agent_runtime(
    agentRuntimeArn='arn:aws:bedrock-agentcore:<AWS_REGION>:<AWS_ACCOUNT_ID>:runtime/<AGENT_RUNTIME_ID>',
    #    runtimeSessionId='<RUNTIME_SESSION_ID>',  # Must be 33+ chars
    payload=payload,
    qualifier="DEFAULT"  # Optional
)
response_body = response['response'].read()
response_data = json.loads(response_body)
print("Agent Response:", response_data)
print("Agent Response:", response_data)
