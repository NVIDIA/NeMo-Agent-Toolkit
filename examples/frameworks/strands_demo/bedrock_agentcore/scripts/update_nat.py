import boto3

client = boto3.client('bedrock-agentcore-control', region_name='<AWS_REGION>')

response = client.update_agent_runtime(
    agentRuntimeId='<AGENT_RUNTIME_ID>',
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': '<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/strands-demo:latest'
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn='<IAM_AGENTCORE_ROLE>')

print("Agent Runtime created successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")
