"""Provision the Lab 5 Task 4 demonstration on the second provider: one bucket, one function.

Not one of the three swapped methods — the handout swaps upload, download and invoke and
explicitly not the whole system — but something has to exist for invoke to call. Kept apart
from aws.py so the adapter stays exactly what a real AWS account would run.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlparse


def provision(blob_uri: str, function_name: str, code_dir: Path, model_version: str) -> str:
    """Create the bucket in BLOB_URI and a python3.12 function whose code is `code_dir`.

    The code is mounted, not uploaded: the emulator's "hot-reload" bucket maps a host
    directory into the function. On AWS this would be a zip in S3 or a container image in ECR
    — a real difference between the two, and noted in the write-up.
    """
    import boto3
    from botocore.exceptions import ClientError

    region = os.environ.get("AWS_REGION", "ap-southeast-1")
    s3 = boto3.client("s3", region_name=region)
    bucket = urlparse(blob_uri).netloc
    try:
        s3.create_bucket(Bucket=bucket,
                         CreateBucketConfiguration={"LocationConstraint": region})
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise

    fn = boto3.client("lambda", region_name=region)
    try:
        fn.delete_function(FunctionName=function_name)
    except ClientError:
        pass
    fn.create_function(
        FunctionName=function_name, Runtime="python3.12",
        Role="arn:aws:iam::000000000000:role/itcs355-serving",
        Handler="cloudlayer.lambda_handler.handler",
        Code={"S3Bucket": "hot-reload", "S3Key": str(code_dir.resolve())},
        Timeout=120, MemorySize=1024,
        Environment={"Variables": {"MODEL_PATH": "/var/task/model.joblib",
                                   "MODEL_VERSION": model_version}},
        Tags={"course": "itcs355", "lab": "5"},
    )
    for _ in range(60):
        if fn.get_function(FunctionName=function_name)["Configuration"]["State"] == "Active":
            return function_name
        time.sleep(2)
    raise RuntimeError(f"function {function_name} never became Active")
