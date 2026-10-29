"""NFR-SEC3: every id inside a request body must be a TenantRef (resolved against the caller's org)."""

from app.main import app

# "Schema.field" allowed without TenantRef, each with its reason:
ALLOWED_PLAIN = {
    "SignIn.upload_id",  # the S3 multipart UploadId string, not a platform row id
    "FixesIn.fix_ids",  # auto-fix codes (enum), not row ids
    "StarterImportIn.starter_id",  # a global catalog slug (starters/catalog.json), not a tenant row
}


def _walk(schema, components, path, out, seen):
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        if name in seen:
            return
        seen.add(name)
        _walk(components[name], components, name, out, seen)
        return
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            _walk(sub, components, path, out, seen)
    if "items" in schema:
        _walk(schema["items"], components, path, out, seen)
    for prop, sub in schema.get("properties", {}).items():
        if prop.endswith(("_id", "_ids")):
            out.append((f"{path}.{prop}", sub))
        _walk(sub, components, path, out, seen)


def test_every_body_id_is_a_tenant_ref():
    spec = app.openapi()
    components = spec["components"]["schemas"]
    fields = []
    for path_item in spec["paths"].values():
        for op in path_item.values():
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            if body:
                _walk(body, components, "body", fields, set())
    assert fields, "expected at least one id field in a request body"
    bad = [name for name, sub in fields if "x-tenant-ref" not in sub and name not in ALLOWED_PLAIN]
    assert bad == [], f"request-body ids without TenantRef: {bad}"
