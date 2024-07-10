import boto3

from botocore.exceptions import ClientError
import os
import json

prompt = "translate the text \"That was genuinely mildly impressive.\" to Hebrew"
access_key = os.getenv('ACCESS_KEY')
secret_key = os.getenv('SECRET_KEY')
brt = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key )
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
native_request = {"messages":
                      [{"role":"user","content":
                          [{"type":"text",
                            "text": prompt}]}],
                  "anthropic_version":"bedrock-2023-05-31",
                  "max_tokens":2000,
                  "temperature":1,
                  "top_k":250,
                  "top_p":0.999,
                  "stop_sequences":[]}

request = json.dumps(native_request)
try:
    # Invoke the model with the request.
    response = brt.invoke_model(modelId=model_id, body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Decode the response body.
model_response = json.loads(response["body"].read())
# Extract and print the response text.
response_text = model_response["content"][0]["text"]
print(response_text)
