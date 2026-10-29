"""bundle_files: the per-file list shown before download (tokenizer files grouped, uncompressed sizes)."""
import io
import zipfile

from app.ml.export import bundle_files


def test_bundle_files_groups_tokenizer_and_reports_uncompressed_sizes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.onnx", b"\0" * 5000)
        z.writestr("labels.txt", "cat\ndog\n")
        z.writestr("tokenizer/vocab.txt", "a" * 300)
        z.writestr("tokenizer/config.json", "{}")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
        files = {f["name"]: f for f in bundle_files(z)}
    assert files["model.onnx"]["size_bytes"] == 5000
    assert files["labels.txt"]["size_bytes"] == 8 and files["labels.txt"]["description"] == "class names"
    assert files["tokenizer/"] == {"name": "tokenizer/", "size_bytes": 302, "files": 2, "description": "text tokenizer + config"}
