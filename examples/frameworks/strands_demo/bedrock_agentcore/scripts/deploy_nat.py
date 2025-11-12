import boto3

client = boto3.client('bedrock-agentcore-control', region_name='<AWS_REGION>')

response = client.create_agent_runtime(
    agentRuntimeName='strands_demo',
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': '<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/strands-demo:latest'
            #                                                   '<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/nat-test-repo:latest'
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    #                                       roleArn='<IAM_AGENTCORE_ROLE>')
    roleArn='<IAM_AGENTCORE_ROLE>')

print(f"Agent Runtime created successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")
