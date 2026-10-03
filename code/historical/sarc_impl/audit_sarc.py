"""Print a deterministic JSON audit summary for SARC staging (CPU only)."""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from sarc import (
    SARCPyramidStaging,
    SARCStagingBlock,
    SingleDW7StagingBlock,
    count_trainable_parameters,
    embed_dilated_3x3_in_7x7,
    sarc_budget,
)


def randomize_bn(bn: nn.BatchNorm2d) -> None:
    with torch.no_grad():
        bn.running_mean.uniform_(-0.7, 0.7)
        bn.running_var.uniform_(0.2, 2.0)
        bn.weight.uniform_(0.4, 1.4)
        bn.bias.uniform_(-0.3, 0.3)


def main() -> None:
    torch.manual_seed(19092026)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    x_embed = torch.randn(2, 4, 17, 19, dtype=torch.float64, device=device)
    kernel = torch.randn(4, 1, 3, 3, dtype=torch.float64, device=device)
    embedding_errors = {}
    for dilation in (1, 2, 3):
        reference = F.conv2d(x_embed, kernel, padding=dilation, dilation=dilation, groups=4)
        fused = F.conv2d(x_embed, embed_dilated_3x3_in_7x7(kernel, dilation), padding=3, groups=4)
        embedding_errors[str(dilation)] = float((reference - fused).abs().max())

    full = SARCStagingBlock(7, gamma_init=0.23).double().eval()
    for bn in [full.main_bn, *full.branch_bns, full.mix_bn]:
        randomize_bn(bn)
    with torch.no_grad():
        full.branch_logits.copy_(torch.tensor([-0.7, 0.4, 1.1], dtype=torch.float64))
    x_full = torch.randn(2, 7, 23, 21, dtype=torch.float64)
    full_error = float((full(x_full) - full.to_deploy()(x_full)).abs().max())

    r1_block = SingleDW7StagingBlock(6, gamma_init=-0.17).double().eval()
    randomize_bn(r1_block.main_bn)
    randomize_bn(r1_block.mix_bn)
    x_r1 = torch.randn(2, 6, 15, 17, dtype=torch.float64)
    r1_error = float((r1_block(x_r1) - r1_block.to_deploy()(x_r1)).abs().max())

    y = torch.linspace(-4, 4, 257, dtype=torch.float64)
    gamma = torch.tensor(0.2, dtype=torch.float64)
    invalid_gamma_fold_gap = float((gamma * F.silu(y) - F.silu(gamma * y)).abs().max())

    budget = sarc_budget()
    r1 = SARCPyramidStaging(mode="R1")
    r2 = SARCPyramidStaging(mode="R2")
    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "dilated_embedding_max_abs_error": embedding_errors,
        "r1_deploy_max_abs_error_float64": r1_error,
        "r2_r3_deploy_max_abs_error_float64": full_error,
        "invalid_gamma_through_silu_gap": invalid_gamma_fold_gap,
        "actual_trainable_parameters": {
            "R1": count_trainable_parameters(r1),
            "R2_R3": count_trainable_parameters(r2),
        },
        "budget": budget.__dict__,
        "relative_to_yolo26n": {
            "deploy_parameter_fraction": budget.deploy_total_parameters / 2_378_736,
            "deploy_flops_fraction": budget.deploy_flops_two_per_mac / 5.3e9,
            "training_branch_conv_mac_ratio_to_deploy": (
                budget.training_conv_macs_r2_r3 / budget.deploy_macs
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

