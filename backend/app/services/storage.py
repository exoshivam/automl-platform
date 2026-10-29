"""S3 (MinIO in dev). Every key starts with org/{organization_id}/ (NFR-SEC3); presigned URLs live 15 min max."""

import uuid
from functools import lru_cache

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.settings import get_settings


@lru_cache
def client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        region_name=s.s3_region,
        aws_access_key_id=s.s3_access_key or None,
        aws_secret_access_key=s.s3_secret_key or None,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
    )


def bucket() -> str:
    return get_settings().s3_bucket


def ensure_bucket() -> None:
    try:
        client().head_bucket(Bucket=bucket())
    except ClientError:
        client().create_bucket(Bucket=bucket())


# ---- key layout

def org_prefix(org_id: uuid.UUID) -> str:
    return f"org/{org_id}/"


def upload_key(org_id, upload_id, index: int) -> str:
    return f"{org_prefix(org_id)}uploads/{upload_id}/{index:06d}"


def object_key(org_id, dataset_id, sha256: str, ext: str = "") -> str:
    """Content-addressed per dataset: the same bytes uploaded twice are stored once (DAT-4)."""
    return f"{org_prefix(org_id)}datasets/{dataset_id}/objects/{sha256[:2]}/{sha256}{ext}"


def thumb_key(org_id, dataset_id, sha256: str) -> str:
    return f"{org_prefix(org_id)}datasets/{dataset_id}/thumbs/{sha256}.webp"


def table_key(org_id, dataset_id, sha256: str) -> str:
    return f"{org_prefix(org_id)}datasets/{dataset_id}/tables/{sha256}.parquet"


def manifest_key(org_id, dataset_id, version_no: int) -> str:
    return f"{org_prefix(org_id)}datasets/{dataset_id}/versions/{version_no}/manifest.parquet"


# ---- operations

def presign_put(key: str, expires: int | None = None) -> str:
    return client().generate_presigned_url("put_object", Params={"Bucket": bucket(), "Key": key},
                                           ExpiresIn=expires or get_settings().presign_ttl_seconds)


def presign_part(key: str, s3_upload_id: str, part_number: int, expires: int | None = None) -> str:
    return client().generate_presigned_url(
        "upload_part", Params={"Bucket": bucket(), "Key": key, "UploadId": s3_upload_id, "PartNumber": part_number},
        ExpiresIn=expires or get_settings().presign_ttl_seconds)


def presign_get(key: str, expires: int | None = None) -> str:
    return client().generate_presigned_url("get_object", Params={"Bucket": bucket(), "Key": key},
                                           ExpiresIn=expires or get_settings().presign_ttl_seconds)


def create_multipart(key: str) -> str:
    return client().create_multipart_upload(Bucket=bucket(), Key=key)["UploadId"]


def list_parts(key: str, s3_upload_id: str) -> list[dict]:
    parts, marker = [], 0
    while True:
        try:
            r = client().list_parts(Bucket=bucket(), Key=key, UploadId=s3_upload_id, PartNumberMarker=marker)
        except ClientError:
            return parts  # completed or aborted
        parts += [{"part_number": p["PartNumber"], "etag": p["ETag"].strip('"'), "size": p["Size"]}
                  for p in r.get("Parts", [])]
        if not r.get("IsTruncated"):
            return parts
        marker = r["NextPartNumberMarker"]


def complete_multipart(key: str, s3_upload_id: str, parts: list[dict]) -> None:
    client().complete_multipart_upload(
        Bucket=bucket(), Key=key, UploadId=s3_upload_id,
        MultipartUpload={"Parts": [{"PartNumber": p["part_number"], "ETag": p["etag"]} for p in parts]})


def head(key: str) -> dict | None:
    try:
        return client().head_object(Bucket=bucket(), Key=key)
    except ClientError:
        return None


def existing_sizes(prefix: str) -> dict[str, int]:
    out, token = {}, None
    while True:
        kw = {"Bucket": bucket(), "Prefix": prefix, **({"ContinuationToken": token} if token else {})}
        r = client().list_objects_v2(**kw)
        out.update({o["Key"]: o["Size"] for o in r.get("Contents", [])})
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def get_stream(key: str):
    return client().get_object(Bucket=bucket(), Key=key)["Body"]


def get_bytes(key: str, limit: int | None = None) -> bytes:
    kw = {"Range": f"bytes=0-{limit - 1}"} if limit else {}
    return client().get_object(Bucket=bucket(), Key=key, **kw)["Body"].read()


def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    client().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)


def put_file(key: str, path: str, content_type: str = "application/octet-stream") -> None:
    client().upload_file(path, bucket(), key, ExtraArgs={"ContentType": content_type})


def copy(src: str, dst: str) -> None:
    client().copy_object(Bucket=bucket(), Key=dst, CopySource={"Bucket": bucket(), "Key": src})


def delete(key: str) -> None:
    client().delete_object(Bucket=bucket(), Key=key)


# Uppy-style signing (POST /uploads/{id}/sign): one presigned URL per S3 operation.
SIGNABLE = {
    ("PUT", False): "put_object",
    ("POST", False): "create_multipart_upload",
    ("PUT", True): "upload_part",
    ("POST", True): "complete_multipart_upload",
    ("GET", True): "list_parts",
    ("DELETE", True): "abort_multipart_upload",
}


def presign(op: str, key: str, s3_upload_id: str | None = None, part_number: int | None = None) -> str:
    params = {"Bucket": bucket(), "Key": key}
    if s3_upload_id:
        params["UploadId"] = s3_upload_id
    if part_number:
        params["PartNumber"] = part_number
    http = {"create_multipart_upload": "POST", "complete_multipart_upload": "POST"}.get(op)
    return client().generate_presigned_url(op, Params=params, ExpiresIn=get_settings().presign_ttl_seconds,
                                           HttpMethod=http)


def delete_prefix(prefix: str) -> int:
    """Deletes every object under prefix (batches of 1000). Returns how many were deleted."""
    if not prefix or not prefix.startswith("org/") or prefix.count("/") < 3:
        raise ValueError(f"refusing to delete prefix {prefix!r}")
    n = 0
    while True:
        r = client().list_objects_v2(Bucket=bucket(), Prefix=prefix, MaxKeys=1000)
        objs = [{"Key": o["Key"]} for o in r.get("Contents", [])]
        if not objs:
            return n
        client().delete_objects(Bucket=bucket(), Delete={"Objects": objs, "Quiet": True})
        n += len(objs)
