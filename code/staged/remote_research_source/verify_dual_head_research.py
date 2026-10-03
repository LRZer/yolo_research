"""Focused regression tests for YOLO26 dual-head diagnostics and collaborative loss."""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss


MODEL_YAML = "/home/ubuntu/rz/yolo_study/yolo26_voc_research/configs/models/yolo26n_baseline.yaml"


def clone_grads(parameters):
    """Clone gradients, substituting zeros only for parameters without gradients."""
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]


def assert_equivalent_grads(actual, expected, label):
    """Assert exact CPU equality or tightly bounded CUDA repeat-reduction error."""
    assert len(actual) == len(expected), f"{label}: gradient-list length differs"
    exact = all(torch.equal(left, right) for left, right in zip(actual, expected))
    diff_sq = sum(float((left.float() - right.float()).square().sum()) for left, right in zip(actual, expected))
    ref_sq = sum(float(right.float().square().sum()) for right in expected)
    max_abs_diff = max(float((left.float() - right.float()).abs().max()) for left, right in zip(actual, expected))
    relative_l2 = math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-12)
    if actual and actual[0].is_cuda:
        # Re-evaluating detector reductions on CUDA is not guaranteed bitwise identical even when the algebra is.
        # CPU is checked exactly below; this CUDA bound catches material divergence without treating reduction-order
        # noise as an implementation failure.
        assert relative_l2 < 5e-3 and max_abs_diff < 5e-2, {
            "label": label,
            "relative_l2": relative_l2,
            "max_abs_diff": max_abs_diff,
        }
    else:
        assert exact, {"label": label, "relative_l2": relative_l2, "max_abs_diff": max_abs_diff}
    return {"exact": exact, "relative_l2": relative_l2, "max_abs_diff": max_abs_diff}


def logit(probability):
    """Convert a scalar probability strictly inside (0, 1) to a logit."""
    return math.log(probability / (1.0 - probability))


def synthetic_criterion(
    *,
    temperature=1.0,
    quality_gate=False,
    supervision_gate="none",
    reduction="active",
    state="none",
    gain=1.0,
    epochs=20,
):
    """Construct only the state needed to exercise E2ELoss._collaborative_loss."""
    criterion = E2ELoss.__new__(E2ELoss)
    criterion.kd_gain = float(gain)
    criterion.kd_temp = float(temperature)
    criterion.kd_min_quality = 0.25
    criterion.kd_margin = 0.05
    criterion.kd_warmup = 0.0
    criterion.kd_ramp = 0.0
    criterion.kd_fade = 0.0
    criterion.kd_gate = bool(quality_gate)
    criterion.kd_supervision_gate = supervision_gate
    criterion.kd_reduction = reduction
    criterion.kd_state = state
    criterion.kd_state_scale = 1.0
    criterion.kd_reliability = float("nan")
    criterion.kd_reliability_ref = float("nan")
    criterion._kd_epoch_common = None
    criterion._kd_epoch_teacher_better = None
    criterion.tqgb_cls_m2o_gain = 0.0
    criterion.tqgb_cls_o2m_gain = 0.0
    criterion.tqgb_min_score = 0.25
    criterion.tqgb_cls_margin = 0.05
    criterion.tqgb_direction = "none"
    criterion.tqgb_projection = True
    criterion.updates = 0
    criterion.one2one = SimpleNamespace(hyp=SimpleNamespace(epochs=epochs))
    return criterion


def synthetic_case(
    teacher_bac,
    student_bac,
    labels,
    targets,
    fg_o2m=None,
    fg_o2o=None,
    gt_o2m=None,
    gt_o2o=None,
    labels_o2o=None,
):
    """Build differentiable head tensors and assignment dictionaries for synthetic KD checks."""
    teacher_bac = torch.as_tensor(teacher_bac, dtype=torch.float32)
    student_bac = torch.as_tensor(student_bac, dtype=torch.float32)
    batch_size, anchors, classes = teacher_bac.shape
    assert student_bac.shape == teacher_bac.shape
    teacher_scores = teacher_bac.permute(0, 2, 1).contiguous().requires_grad_()
    student_scores = student_bac.permute(0, 2, 1).contiguous().requires_grad_()
    labels = torch.as_tensor(labels, dtype=torch.long)
    labels_o2o = labels if labels_o2o is None else torch.as_tensor(labels_o2o, dtype=torch.long)
    targets = torch.as_tensor(targets, dtype=torch.float32)
    assert targets.shape == (batch_size, anchors, classes)
    if fg_o2m is None:
        fg_o2m = torch.ones(batch_size, anchors, dtype=torch.bool)
    else:
        fg_o2m = torch.as_tensor(fg_o2m, dtype=torch.bool)
    if fg_o2o is None:
        fg_o2o = torch.ones(batch_size, anchors, dtype=torch.bool)
    else:
        fg_o2o = torch.as_tensor(fg_o2o, dtype=torch.bool)
    if gt_o2m is None:
        gt_o2m = torch.zeros(batch_size, anchors, dtype=torch.long)
    else:
        gt_o2m = torch.as_tensor(gt_o2m, dtype=torch.long)
    if gt_o2o is None:
        gt_o2o = torch.zeros(batch_size, anchors, dtype=torch.long)
    else:
        gt_o2o = torch.as_tensor(gt_o2o, dtype=torch.long)

    target_boxes = torch.tensor([0.0, 0.0, 2.0, 2.0]).expand(batch_size, anchors, 4).clone()
    pred_o2m = target_boxes.clone()
    pred_o2o = target_boxes.clone()
    one2many = {"scores": teacher_scores}
    one2one = {"scores": student_scores}
    shared = {
        "target_bboxes": target_boxes,
        "target_scores": targets,
        "num_gt": torch.tensor(float(max(int(fg_o2o.sum()), 1))),
    }
    assign_o2m = {
        **shared,
        "target_labels": labels,
        "fg_mask": fg_o2m,
        "target_gt_idx": gt_o2m,
        "pred_bboxes": pred_o2m,
    }
    assign_o2o = {
        **shared,
        "target_labels": labels_o2o,
        "fg_mask": fg_o2o,
        "target_gt_idx": gt_o2o,
        "pred_bboxes": pred_o2o,
    }
    return one2many, one2one, assign_o2m, assign_o2o, teacher_scores, student_scores


def run_synthetic_tests():
    """Exercise labels, masks, gates, temperature gradients, aggregation, and schedule boundaries on CPU."""
    # A foreground anchor with all-zero soft scores must still gather its discrete class-2 label, never argmax class 0.
    case = synthetic_case(
        teacher_bac=[[[-3.0, -2.0, 2.0], [-1.0, -1.0, -1.0]]],
        student_bac=[[[-4.0, -3.0, 0.0], [-1.0, -1.0, -1.0]]],
        labels=[[[2], [1]]],
        targets=torch.zeros(1, 2, 3),
        fg_o2m=[[True, False]],
        fg_o2o=[[True, False]],
    )
    criterion = synthetic_criterion()
    kd_loss, stats = criterion._collaborative_loss(*case[:4])
    assert torch.isclose(stats["kd_p_teacher_mean"], torch.tensor(2.0).sigmoid())
    assert torch.isclose(stats["kd_p_student_mean"], torch.tensor(0.0).sigmoid())
    assert stats["kd_y_mean"].item() == 0.0
    kd_loss.backward()
    teacher_grad, student_grad = case[4].grad, case[5].grad
    assert teacher_grad is None or torch.count_nonzero(teacher_grad).item() == 0, "teacher was not stop-gradient"
    assert torch.count_nonzero(student_grad).item() == 1, "KD touched a non-assigned class or anchor"
    assert student_grad[0, 2, 0].item() != 0.0, "KD missed the discrete class-2 target"

    # No common same-GT foreground must give finite zeros for loss, counts, and every diagnostic.
    empty_case = synthetic_case(
        teacher_bac=[[[1.0], [1.0]]],
        student_bac=[[[0.0], [0.0]]],
        labels=[[0, 0]],
        targets=[[[0.8], [0.8]]],
        fg_o2m=[[True, False]],
        fg_o2o=[[False, True]],
    )
    empty_loss, empty_stats = synthetic_criterion(quality_gate=True)._collaborative_loss(*empty_case[:4])
    assert empty_loss.item() == 0.0
    assert empty_stats["kd_candidate_count"].item() == 0.0
    assert empty_stats["kd_effective_count"].item() == 0.0
    assert all(torch.isfinite(value).all() for value in empty_stats.values())

    # Gate semantics: quality-gate candidates are counted before the optional O2O-supervision safety gate.
    gate_case = synthetic_case(
        teacher_bac=[[[logit(0.6)], [logit(0.9)], [logit(0.9)]]],
        student_bac=[[[logit(0.2)], [logit(0.8)], [logit(0.2)]]],
        labels=[[0, 0, 0]],
        targets=[[[0.8], [0.1], [0.5]]],
    )
    expected_effective = {"none": 3.0, "direction": 2.0, "closer": 1.0}
    for mode, expected_count in expected_effective.items():
        _, gate_stats = synthetic_criterion(quality_gate=True, supervision_gate=mode)._collaborative_loss(*gate_case[:4])
        assert gate_stats["kd_candidate_count"].item() == 3.0
        assert gate_stats["kd_effective_count"].item() == expected_count
        assert torch.isclose(gate_stats["kd_candidate_rate"], torch.tensor(1.0))
        assert torch.isclose(gate_stats["kd_effective_rate"], torch.tensor(expected_count / 3.0))
    _, gate_stats = synthetic_criterion(quality_gate=True)._collaborative_loss(*gate_case[:4])
    assert torch.isclose(gate_stats["kd_grad_direction_agree"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(gate_stats["kd_teacher_closer"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(gate_stats["kd_teacher_overshoot"], torch.tensor(2.0 / 3.0))

    # The KL gradient is T * (sigmoid(student/T) - sigmoid(teacher/T)) after the conventional T^2 scaling.
    for temperature in (0.5, 1.0, 2.0):
        temperature_case = synthetic_case(
            teacher_bac=[[[1.2]]],
            student_bac=[[[-0.3]]],
            labels=[[0]],
            targets=[[[0.7]]],
        )
        temperature_loss, _ = synthetic_criterion(temperature=temperature)._collaborative_loss(*temperature_case[:4])
        temperature_loss.backward()
        expected = temperature * (
            torch.sigmoid(torch.tensor(-0.3 / temperature)) - torch.sigmoid(torch.tensor(1.2 / temperature))
        )
        actual = temperature_case[5].grad[0, 0, 0]
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6), (
            f"T={temperature}: gradient {float(actual)} != {float(expected)}"
        )
        assert temperature_case[4].grad is None, f"T={temperature}: teacher received a gradient"

    # Average anchors within each image first, then average images in the local batch.
    aggregate_case = synthetic_case(
        teacher_bac=[[[1.0], [0.0]], [[1.5], [0.5]]],
        student_bac=[[[0.0], [0.0]], [[-0.5], [-0.5]]],
        labels=[[0, 0], [0, 0]],
        targets=[[[0.8], [0.0]], [[0.8], [0.8]]],
        fg_o2m=[[True, False], [True, True]],
        fg_o2o=[[True, False], [True, True]],
    )
    aggregate_loss, _ = synthetic_criterion()._collaborative_loss(*aggregate_case[:4])
    teacher_logits = torch.tensor([[1.0, 0.0], [1.5, 0.5]])
    student_logits = torch.tensor([[0.0, 0.0], [-0.5, -0.5]])
    teacher_prob = teacher_logits.sigmoid()
    point_kl = teacher_prob * (teacher_prob.log() - torch.nn.functional.logsigmoid(student_logits))
    point_kl += (1 - teacher_prob) * (
        (1 - teacher_prob).log() - torch.nn.functional.logsigmoid(-student_logits)
    )
    expected_aggregate = torch.stack((point_kl[0, 0], point_kl[1].mean())).mean()
    assert torch.allclose(aggregate_loss, expected_aggregate, atol=1e-7, rtol=1e-6)

    # Common-positive normalization lets candidate coverage reduce total KD strength instead of renormalizing every
    # selected subset back to full strength. With two selected anchors out of four common anchors the ratio is 1/2.
    reduction_case_active = synthetic_case(
        teacher_bac=[[[logit(0.9)], [logit(0.9)], [logit(0.30)], [logit(0.30)]]],
        student_bac=[[[logit(0.2)], [logit(0.2)], [logit(0.29)], [logit(0.29)]]],
        labels=[[0, 0, 0, 0]],
        targets=[[[0.8], [0.8], [0.8], [0.8]]],
    )
    reduction_case_common = synthetic_case(
        teacher_bac=[[[logit(0.9)], [logit(0.9)], [logit(0.30)], [logit(0.30)]]],
        student_bac=[[[logit(0.2)], [logit(0.2)], [logit(0.29)], [logit(0.29)]]],
        labels=[[0, 0, 0, 0]],
        targets=[[[0.8], [0.8], [0.8], [0.8]]],
    )
    active_loss, _ = synthetic_criterion(quality_gate=True, reduction="active")._collaborative_loss(
        *reduction_case_active[:4]
    )
    common_loss, common_stats = synthetic_criterion(quality_gate=True, reduction="common")._collaborative_loss(
        *reduction_case_common[:4]
    )
    assert common_stats["kd_candidate_count"].item() == 2.0
    assert torch.allclose(common_loss, active_loss * 0.5, atol=1e-7, rtol=1e-6)

    # A train-only reference of rho=0.8 gives scale 1.0; later rho=0.65 and rho=0.5 causally produce 0.5 and 0.
    state = synthetic_criterion(state="teacher_better")
    state.kd_warmup = 5.0
    state.updates = 5
    for better, expected_scale in ((80.0, 1.0), (65.0, 0.5), (50.0, 0.0)):
        state._kd_epoch_common = torch.tensor(100.0)
        state._kd_epoch_teacher_better = torch.tensor(better)
        state._update_kd_state()
        assert math.isclose(state.kd_state_scale, expected_scale, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(state.kd_reliability_ref, 0.8, rel_tol=0.0, abs_tol=1e-7)
    state._kd_epoch_common = torch.tensor(100.0)
    state._kd_epoch_teacher_better = torch.tensor(90.0)
    state._update_kd_state()
    assert state.kd_state_scale == 0.0, "retired teacher was reactivated by a later reliability fluctuation"

    # Epoch schedule boundaries: five warmup epochs, four-epoch ramp, and three-epoch terminal fade.
    schedule = synthetic_criterion(epochs=20)
    schedule.kd_warmup = 5.0
    schedule.kd_ramp = 4.0
    schedule.kd_fade = 3.0
    expected_scales = {0: 0.0, 4: 0.0, 5: 0.25, 8: 1.0, 16: 1.0, 17: 2.0 / 3.0, 18: 1.0 / 3.0, 19: 0.0}
    for updates, expected in expected_scales.items():
        schedule.updates = updates
        assert math.isclose(schedule._kd_scale(), expected, rel_tol=0.0, abs_tol=1e-12), (
            updates,
            schedule._kd_scale(),
            expected,
        )

    # TQGB routes each classification winner only into the losing head. Supervision-segment projection preserves a
    # teacher that moves the receiver toward its own target, clips overshoot to that target, and removes a teacher on
    # the wrong side of the receiver entirely.
    tqgb_case = synthetic_case(
        teacher_bac=[[[logit(0.8)], [logit(0.8)], [logit(0.2)], [logit(0.2)]]],
        student_bac=[[[logit(0.2)], [logit(0.2)], [logit(0.8)], [logit(0.8)]]],
        labels=[[0, 0, 0, 0]],
        targets=[[[0.6], [0.1], [0.7], [0.1]]],
    )
    tqgb = synthetic_criterion(gain=0.0)
    tqgb.tqgb_cls_m2o_gain = 1.0
    tqgb.tqgb_cls_o2m_gain = 1.0
    tqgb.tqgb_min_score = 0.25
    tqgb.tqgb_cls_margin = 0.05
    tqgb.tqgb_direction = "bidirectional"
    tqgb.tqgb_projection = True
    tqgb_loss, tqgb_stats = tqgb._tqgb_classification_loss(*tqgb_case[:4])
    assert tqgb_stats["tqgb_cls_m2o_count"].item() == 2.0
    assert tqgb_stats["tqgb_cls_o2m_count"].item() == 2.0
    assert torch.isclose(tqgb_stats["tqgb_cls_m2o_projection_keep"], torch.tensor(0.5))
    assert torch.isclose(tqgb_stats["tqgb_cls_o2m_projection_keep"], torch.tensor(0.5))
    tqgb_loss.backward()
    o2m_grad = tqgb_case[4].grad.permute(0, 2, 1)
    o2o_grad = tqgb_case[5].grad.permute(0, 2, 1)
    assert o2o_grad[0, 0, 0].item() != 0 and o2m_grad[0, 0, 0].item() == 0
    assert o2o_grad[0, 1, 0].item() == 0 and o2m_grad[0, 1, 0].item() == 0
    assert o2m_grad[0, 2, 0].item() != 0 and o2o_grad[0, 2, 0].item() == 0
    assert o2m_grad[0, 3, 0].item() == 0 and o2o_grad[0, 3, 0].item() == 0

    # A same-GT foreground pair with different assigned classes is not a valid TQGB correspondence.
    different_class_case = synthetic_case(
        teacher_bac=[[[logit(0.8), logit(0.1)]]],
        student_bac=[[[logit(0.2), logit(0.7)]]],
        labels=[[[0]]],
        labels_o2o=[[[1]]],
        targets=[[[0.6, 0.0]]],
        gt_o2m=[[0]],
        gt_o2o=[[0]],
    )
    different_class_loss, different_class_stats = tqgb._tqgb_classification_loss(*different_class_case[:4])
    assert different_class_loss.item() == 0.0
    assert different_class_stats["tqgb_common_count"].item() == 0.0

    # The m2o ablation cannot leak a reverse gradient, and common-anchor normalization retains route sparsity.
    m2o_case = synthetic_case(
        teacher_bac=[[[logit(0.8)], [logit(0.8)], [logit(0.2)], [logit(0.2)]]],
        student_bac=[[[logit(0.2)], [logit(0.2)], [logit(0.8)], [logit(0.8)]]],
        labels=[[0, 0, 0, 0]],
        targets=[[[0.6], [0.6], [0.7], [0.7]]],
    )
    tqgb.tqgb_direction = "m2o"
    m2o_loss, m2o_stats = tqgb._tqgb_classification_loss(*m2o_case[:4])
    assert m2o_stats["tqgb_cls_m2o_count"].item() == 2.0
    assert m2o_stats["tqgb_cls_o2m_count"].item() == 0.0
    m2o_loss.backward()
    assert m2o_case[4].grad is None, "m2o teacher received a gradient"
    assert torch.count_nonzero(m2o_case[5].grad).item() == 2

    # Empty correspondence and a disabled gain remain finite zeros.
    tqgb_empty = synthetic_case(
        teacher_bac=[[[logit(0.8)]]],
        student_bac=[[[logit(0.2)]]],
        labels=[[0]],
        targets=[[[0.6]]],
        gt_o2m=[[0]],
        gt_o2o=[[1]],
    )
    tqgb_empty_loss, tqgb_empty_stats = tqgb._tqgb_classification_loss(*tqgb_empty[:4])
    assert tqgb_empty_loss.item() == 0.0
    assert all(torch.isfinite(value).all() for value in tqgb_empty_stats.values())
    tqgb.tqgb_cls_m2o_gain = 0.0
    tqgb.tqgb_cls_o2m_gain = 0.0
    disabled_loss, _ = tqgb._tqgb_classification_loss(*m2o_case[:4])
    assert disabled_loss.item() == 0.0

    return {
        "target_labels_zero_soft_score": True,
        "mask_edges": True,
        "gate_semantics": True,
        "temperature_gradients": True,
        "per_image_aggregation": True,
        "reduction_modes": True,
        "teacher_state_controller": True,
        "schedule_boundaries": True,
        "tqgb_projection_and_routes": True,
        "tqgb_same_gt_different_class_excluded": True,
        "tqgb_empty_and_disabled": True,
    }


def run_model_tests(model_yaml, device):
    """Check exact official-base equivalence and end-to-end gradient isolation with a real YOLO26 model."""
    torch.manual_seed(123)
    model = YOLO(model_yaml).model.to(device).train()
    model.args = get_cfg()
    model.args.epochs = 10
    model.args.o2o_kd_gain = 0.0
    model.args.o2o_kd_supervision_gate = "none"
    model.args.tqgb_cls_m2o_gain = 0.0
    model.args.tqgb_cls_o2m_gain = 0.0
    model.args.tqgb_direction = "none"
    model.criterion = model.init_criterion()
    criterion = model.criterion

    images = torch.rand(2, 3, 128, 128, device=device)
    batch = {
        "img": images,
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.tensor([[14.0], [6.0]], device=device),
        "bboxes": torch.tensor([[0.50, 0.50, 0.28, 0.34], [0.36, 0.62, 0.22, 0.25]], device=device),
    }

    preds = model.predict(images)
    # Break the initially identical deep-copied heads so the synthetic KD term has a nonzero gradient.
    preds["one2one"]["scores"] = preds["one2one"]["scores"] - 0.75

    criterion.kd_gain = 0.0
    actual, items = criterion(preds, batch)
    _, raw_o2m, _ = criterion.one2many.get_assigned_targets_and_loss(
        preds["one2many"], batch, return_details=True
    )
    _, raw_o2o, _ = criterion.one2one.get_assigned_targets_and_loss(preds["one2one"], batch, return_details=True)
    batch_size = images.shape[0]
    official = raw_o2m * batch_size * criterion.o2m + raw_o2o * batch_size * criterion.o2o
    assert torch.equal(actual[:3], official), "gain=0 changed the official three-component base loss"
    assert actual[3].item() == 0.0, "gain=0 produced a nonzero collaborative loss"
    assert actual[4].item() == 0.0, "gain=0 produced a nonzero TQGB loss"
    assert all(torch.isfinite(value).all() for value in items.values()), "diagnostic contains NaN/Inf"

    all_parameters = list(model.parameters())
    model.zero_grad(set_to_none=True)
    actual[:3].sum().backward(retain_graph=True)
    actual_grads = clone_grads(all_parameters)
    model.zero_grad(set_to_none=True)
    official.sum().backward()
    official_grads = clone_grads(all_parameters)
    gradient_equivalence = assert_equivalent_grads(actual_grads, official_grads, "official base loss")

    empty = {
        "img": images[:1],
        "batch_idx": torch.empty(0, device=device),
        "cls": torch.empty(0, 1, device=device),
        "bboxes": torch.empty(0, 4, device=device),
    }
    empty_preds = model.predict(images[:1])
    empty_loss, empty_items = criterion(empty_preds, empty)
    assert torch.isfinite(empty_loss).all(), "empty-target loss contains NaN/Inf"
    assert all(torch.isfinite(value).all() for value in empty_items.values()), "empty diagnostic contains NaN/Inf"

    preds = model.predict(images)
    preds["one2one"]["scores"] = preds["one2one"]["scores"] - 0.75
    criterion.kd_gain = 1.0
    criterion.kd_gate = False
    criterion.kd_supervision_gate = "none"
    criterion.kd_warmup = 0.0
    criterion.kd_ramp = 0.0
    criterion.kd_fade = 0.0
    losses, active_items = criterion(preds, batch)
    assert losses[3].item() > 0, "synthetic collaborative loss did not activate"
    assert active_items["kd_effective_count"].item() > 0, "no same-GT matched anchors were found"

    head = model.model[-1]
    backbone_params = list(model.model[0].parameters())
    o2m_params = list(head.cv2.parameters()) + list(head.cv3.parameters())
    o2o_box_params = list(head.one2one_cv2.parameters())
    o2o_cls_params = list(head.one2one_cv3.parameters())

    model.zero_grad(set_to_none=True)
    losses[3].backward()
    kd_backbone = clone_grads(backbone_params)
    kd_o2m = clone_grads(o2m_params)
    kd_o2o_box = clone_grads(o2o_box_params)
    kd_o2o_cls = clone_grads(o2o_cls_params)

    backbone_grad = max(tensor.abs().max().item() for tensor in kd_backbone)
    o2m_grad = max(tensor.abs().max().item() for tensor in kd_o2m)
    o2o_box_grad = max(tensor.abs().max().item() for tensor in kd_o2o_box)
    o2o_cls_grad = max(tensor.abs().max().item() for tensor in kd_o2o_cls)
    assert backbone_grad == 0.0, f"KD reached the backbone: {backbone_grad}"
    assert o2m_grad == 0.0, f"KD reached the O2M teacher: {o2m_grad}"
    assert o2o_box_grad == 0.0, f"classification KD reached the O2O box branch: {o2o_box_grad}"
    assert o2o_cls_grad > 0, "KD did not reach the O2O classification branch"

    # TQGB O2M->O2O routes only into the O2O class branch, matching the deployment-head detach topology.
    preds = model.predict(images)
    preds["one2one"]["scores"] = preds["one2one"]["scores"] - 0.75
    criterion.kd_gain = 0.0
    criterion.tqgb_cls_m2o_gain = 1.0
    criterion.tqgb_cls_o2m_gain = 1.0
    criterion.tqgb_min_score = 0.0
    criterion.tqgb_cls_margin = 0.0
    criterion.tqgb_direction = "m2o"
    criterion.tqgb_projection = False
    tqgb_m2o_losses, tqgb_m2o_items = criterion(preds, batch)
    assert tqgb_m2o_losses[4].item() > 0
    assert tqgb_m2o_items["tqgb_cls_m2o_count"].item() > 0
    model.zero_grad(set_to_none=True)
    tqgb_m2o_losses[4].backward()
    tqgb_m2o_backbone = max(tensor.abs().max().item() for tensor in clone_grads(backbone_params))
    tqgb_m2o_o2m = max(tensor.abs().max().item() for tensor in clone_grads(o2m_params))
    tqgb_m2o_o2o_box = max(tensor.abs().max().item() for tensor in clone_grads(o2o_box_params))
    tqgb_m2o_o2o_cls = max(tensor.abs().max().item() for tensor in clone_grads(o2o_cls_params))
    assert tqgb_m2o_backbone == 0.0
    assert tqgb_m2o_o2m == 0.0
    assert tqgb_m2o_o2o_box == 0.0
    assert tqgb_m2o_o2o_cls > 0

    # TQGB O2O->O2M uses the deployment head only as a detached teacher and bridges through O2M into the trunk.
    preds = model.predict(images)
    preds["one2one"]["scores"] = preds["one2one"]["scores"] + 0.75
    criterion.tqgb_direction = "bidirectional"
    tqgb_reverse_losses, tqgb_reverse_items = criterion(preds, batch)
    assert tqgb_reverse_losses[4].item() > 0
    assert tqgb_reverse_items["tqgb_cls_o2m_count"].item() > 0
    model.zero_grad(set_to_none=True)
    tqgb_reverse_losses[4].backward()
    tqgb_reverse_backbone = max(tensor.abs().max().item() for tensor in clone_grads(backbone_params))
    tqgb_reverse_o2m = max(tensor.abs().max().item() for tensor in clone_grads(o2m_params))
    tqgb_reverse_o2o_box = max(tensor.abs().max().item() for tensor in clone_grads(o2o_box_params))
    tqgb_reverse_o2o_cls = max(tensor.abs().max().item() for tensor in clone_grads(o2o_cls_params))
    assert tqgb_reverse_backbone > 0
    assert tqgb_reverse_o2m > 0
    assert tqgb_reverse_o2o_box == 0.0
    assert tqgb_reverse_o2o_cls == 0.0

    return {
        "official_loss_exact": True,
        "official_gradient_exact": gradient_equivalence["exact"],
        "official_gradient_relative_l2": gradient_equivalence["relative_l2"],
        "official_gradient_max_abs_diff": gradient_equivalence["max_abs_diff"],
        "empty_targets_finite": True,
        "kd_loss": float(losses[3]),
        "kd_effective_count": float(active_items["kd_effective_count"]),
        "backbone_kd_grad": backbone_grad,
        "o2m_kd_grad": o2m_grad,
        "o2o_box_kd_grad": o2o_box_grad,
        "o2o_cls_kd_grad": o2o_cls_grad,
        "tqgb_m2o_backbone_grad": tqgb_m2o_backbone,
        "tqgb_m2o_o2m_grad": tqgb_m2o_o2m,
        "tqgb_m2o_o2o_box_grad": tqgb_m2o_o2o_box,
        "tqgb_m2o_o2o_cls_grad": tqgb_m2o_o2o_cls,
        "tqgb_reverse_backbone_grad": tqgb_reverse_backbone,
        "tqgb_reverse_o2m_grad": tqgb_reverse_o2m,
        "tqgb_reverse_o2o_box_grad": tqgb_reverse_o2o_box,
        "tqgb_reverse_o2o_cls_grad": tqgb_reverse_o2o_cls,
    }


def main():
    """Run synthetic CPU-capable checks and, unless skipped, real-model integration checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_YAML)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()

    report = {"status": "PASS", **run_synthetic_tests()}
    if not args.synthetic_only:
        report.update(run_model_tests(args.model, torch.device(args.device)))
    print(report)


if __name__ == "__main__":
    main()
