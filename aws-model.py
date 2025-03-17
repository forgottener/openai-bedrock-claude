"""
Copyright 2024 AWS Claude API Proxy Service Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import boto3

bedrock = boto3.client('bedrock', region_name='us-east-1')

response = bedrock.list_foundation_models(byProvider="anthropic")

for summary in response["modelSummaries"]:
	print(summary["modelId"])