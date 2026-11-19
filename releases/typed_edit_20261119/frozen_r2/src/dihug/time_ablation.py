#!/usr/bin/env python3
"""Create a paired checkpoint that cannot use the diffusion-time feature."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from dihug.core.mixed_edit_diffusion import (
    mixed_edit_time_feature_index,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu")
    family_count = len(payload["prior"]["family_names"])
    time_index = mixed_edit_time_feature_index(family_count)
    state_dict = payload["state_dict"]
    first_weight = state_dict.get("network.0.weight")
    if first_weight is None:
        raise KeyError("checkpoint does not contain network.0.weight")
    if time_index >= first_weight.shape[1]:
        raise IndexError(
            f"time feature {time_index} outside input width {first_weight.shape[1]}"
        )

    payload = dict(payload)
    payload["state_dict"] = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }
    payload["state_dict"]["network.0.weight"][:, time_index] = 0.0
    contract = dict(payload.get("diffusion_contract", {}))
    contract["time_conditioning"] = False
    contract["time_ablation"] = {
        "method": "zero_time_input_weights",
        "source_checkpoint": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "time_feature_index": int(time_index),
    }
    payload["diffusion_contract"] = contract

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    saved = torch.load(args.output, map_location="cpu")
    weight = saved["state_dict"]["network.0.weight"][:, time_index]
    if torch.count_nonzero(weight).item() != 0:
        raise AssertionError("saved checkpoint retained nonzero time weights")
    print(
        f"[TIME-ABLATION] input={args.input} output={args.output} "
        f"time_index={time_index}"
    )


if __name__ == "__main__":
    main()
