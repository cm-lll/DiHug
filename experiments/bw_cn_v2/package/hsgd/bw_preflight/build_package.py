"""Explicit local packaging; never installs dependencies or starts a device run."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from .common import PACKAGE, sha


def inspector_diff(original, patched):
    diff = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True),
                                     patched.decode().splitlines(keepends=True),
                                     fromfile="a/hsgd/vendor/transconv_layer.py",
                                     tofile="b/hsgd/vendor/transconv_layer.py"))
    if not diff or "@@" not in diff:
        raise RuntimeError("Inspector compatibility diff is empty; refuse packaging")
    return diff.encode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--previous-manifest", type=Path, required=True)
    p.add_argument("--local-evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "dihug_sparse_bw_preflight_v4.zip"
    if archive.exists():
        raise RuntimeError("archive exists; do not overwrite")
    files = {}
    for base in (PACKAGE / "hsgd", PACKAGE / "tests"):
        for file in sorted(base.rglob("*")):
            if file.is_file() and (file.suffix in (".py", ".json") or file.name == "LICENSE"):
                files[str(file.relative_to(PACKAGE))] = file.read_bytes()
    files["data/IMDB_movie_reference.npz"] = args.reference.read_bytes()
    files["data/IMDB_meta.json"] = args.meta.read_bytes()
    files["manifest_v1_source.json"] = args.previous_manifest.read_bytes()
    files["local_acceptance.json"] = args.local_evidence.read_bytes()
    old = subprocess.check_output(["git", "-C", str(PACKAGE), "show",
          "b3d3b1dc44c49f4fe01d8d8a6faf9d5420e7707f:sparse_restart/hsgd/vendor/transconv_layer.py"])
    original = hashlib.sha256(old).hexdigest()
    assert original == "cffda25c47f70070e4523aa1dd5cad9d37c31f31b90ff54cf7b0269afcb03813"
    files["pyg_inspector_compat.diff"] = inspector_diff(old, (PACKAGE/"hsgd/vendor/transconv_layer.py").read_bytes())
    manifest = dict(version="bw_preflight_v4", source_commit=subprocess.check_output(
        ["git", "-C", str(PACKAGE), "rev-parse", "HEAD"], text=True).strip(),
        modified_tracked_files=subprocess.check_output(["git", "-C", str(PACKAGE), "diff", "--name-only"], text=True).splitlines(),
        original_transconv_sha256=original, patched_transconv_sha256=sha(PACKAGE / "hsgd/vendor/transconv_layer.py"),
        supplied_patch_sha256="de22427d586d9ddbb96a11551a2afdc4ae4f0e4a6e4f3d4a7e0fc036897beca8",
        supplied_patch_file_present=False,
        patch_provenance="minimal locally rebuilt collect_param_data-first/distribute fallback; not claiming byte identity with supplied patch",
        scope="manual A/B/C engineering only; no automatic formal training; no M/K/critic/completer",
        backend_policy="preserve vendor DTK PyTorch on BW and NVIDIA PyTorch locally; no installer in runtime",
        files={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)) for name,data in sorted(files.items())})
    files["manifest_v4.json"] = json.dumps(manifest, indent=2).encode()
    checksums = "".join(hashlib.sha256(data).hexdigest() + "  " + name + "\n" for name,data in sorted(files.items()))
    files["SHA256SUMS"] = checksums.encode()
    folder = "dihug_sparse_bw_preflight_v4/"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info = zipfile.ZipInfo(folder + name, date_time=(2026,10,14,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    (args.output / "manifest_v4.json").write_bytes(files["manifest_v4.json"])
    (args.output / "SHA256SUMS").write_text(checksums + sha(archive) + "  " + archive.name + "\n")
    print(json.dumps(dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
                         content_files=len(files), source_commit=manifest["source_commit"]), indent=2))


if __name__ == "__main__":
    main()
