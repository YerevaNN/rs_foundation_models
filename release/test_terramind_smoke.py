import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

PINNED = {
    "torch": "2.7.1+cpu",
    "torchvision": "0.22.1+cpu",
    "terratorch": "1.1.1",
    "timm": "1.0.20",
    "transformers": "4.57.3",
    "mmengine": "0.10.7",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-source", type=Path, required=True,
                        help="rs_finetune directory exported from TerraMind commit af2a54f")
    args = parser.parse_args()
    for package, expected in PINNED.items():
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {actual}")
    sys.path.insert(0, str(args.branch_source))
    import torch
    from change_detection_pytorch.encoders.terramind import TerraMindEncoder
    model = TerraMindEncoder(
        model_name="terramind_v1_base", pretrained=False, modalities=["S2L2A"],
        img_size=224, patch_size=16, for_cls=True,
        bands={"S2L2A": ["B02", "B03", "B04"]},
    ).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    result = {
        "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "bands": model.bands,
    }
    if result["shape"] != [1, 768] or not result["finite"]:
        raise RuntimeError(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
