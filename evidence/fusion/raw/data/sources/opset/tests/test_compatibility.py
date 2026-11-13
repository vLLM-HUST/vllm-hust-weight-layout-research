import json
from importlib.resources import files

from vllm_hust_operator_optimizations._version import __version__
from vllm_hust_operator_optimizations.compatibility import check_revision


def test_manifest_and_verified_line_are_packaged_and_consistent():
    package = files("vllm_hust_operator_optimizations")
    manifest = json.loads(
        package.joinpath("manifests/vllm-hust-extension-v0.3.json").read_text()
    )
    compatibility = json.loads(package.joinpath("compatibility/verified-line.json").read_text())

    assert manifest["extension_version"] == __version__
    assert manifest["schema_version"] == "0.3-experimental"
    assert manifest["resource_claims"]
    assert manifest["host"]["version_range"] == "==0.1.dev20240"
    assert compatibility["packages"]["torch_npu"] == "2.10.0.post4"
    assert len(compatibility["repositories"]) == 3
    assert compatibility["validated"]["optimizations"] == [
        "persistent-matmul-empty",
        "linear-swiglu-graph",
    ]


def test_revision_check_rejects_missing_checkout():
    missing = "definitely-not-a-real-checkout"
    errors = check_revision(missing, "deadbeef", "vllm")

    assert errors == [f"vllm: checkout not found: {missing}"]
