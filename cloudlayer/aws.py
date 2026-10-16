"""AWS adapter — Lab 5 Task 4: upload, download and invoke only, to prove the seam.

SDK:  pip install boto3 (not in requirements.txt: only the swap demonstration needs it)

Written against real AWS. Nothing in this file knows about LocalStack: boto3 honours the
standard AWS_ENDPOINT_URL variable, so pointing the same code at the emulator the
demonstration runs on (no AWS account was available; see README, Lab 5, Task 4) is a line in
the environment, not a branch in the code. Delete that line and this talks to AWS.

Configuration, read here and nowhere else:
  BLOB_URI     s3://<bucket>/<prefix>
  AWS_REGION   e.g. ap-southeast-1. Not REGION: that one holds an Azure region name
               ("eastasia"), and boto3 would build an endpoint from it — a leak, written up.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3://bucket/key, got {uri!r} — is BLOB_URI still another "
                         "provider's address?")
    return parsed.netloc, parsed.path.lstrip("/")


class AwsAdapter(CloudAdapter):
    def _client(self, service: str) -> Any:
        import boto3

        return boto3.client(service, region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))

    def upload(self, local_path: str, key: str) -> str:
        bucket, prefix = _parse_s3_uri(self.cfg.blob_uri.rstrip("/"))
        object_key = f"{prefix}/{key}".lstrip("/")
        self._client("s3").upload_file(local_path, bucket, object_key)
        return f"s3://{bucket}/{object_key}"

    def download(self, uri: str, local_path: str) -> None:
        bucket, key = _parse_s3_uri(uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client("s3").download_file(bucket, key, local_path)

    def push_image(self, local_tag: str) -> str:
        raise NotImplementedError("not part of the Lab 5 swap: upload, download, invoke only")

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the serving function once and return what the Azure adapter returns.

        `endpoint` is a Lambda function name or ARN. On AWS proper the model would sit behind a
        SageMaker endpoint (sagemaker-runtime.invoke_endpoint); SageMaker is not in the
        emulator's free tier, so the same scoring code runs as a Lambda function. Either way
        the caller sees the dict below.

        The leak this method exists to absorb: Lambda answers HTTP 200 even when the function
        failed — the failure is a FunctionError field beside the payload — and the service's
        own status, headers and body come back wrapped one level deeper. The Azure adapter got
        the status from HTTP itself.
        """
        response = self._client("lambda").invoke(
            FunctionName=endpoint, InvocationType="RequestResponse",
            Payload=json.dumps({"body": json.dumps(payload)}).encode(),
        )
        envelope = json.loads(response["Payload"].read() or b"{}")
        if response.get("FunctionError"):
            raise RuntimeError(f"function error: {envelope.get('errorMessage', envelope)}")
        status = int(envelope.get("statusCode", 500))
        body = json.loads(envelope.get("body") or "{}")
        if status >= 400:
            raise RuntimeError(f"HTTP {status} from {endpoint}: {body}")
        body["model_version_header"] = (envelope.get("headers") or {}).get("x-model-version", "")
        return body
