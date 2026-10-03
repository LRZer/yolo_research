"""CPU-only tests for the local YOLO26 SARC integration candidate."""

from __future__ import annotations

import copy
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "snapshot" / "ultralytics_research"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(ROOT)
sys.path.insert(0, str(REPO))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import SARCBlock, SARCDeployBlock, SARCFixedBlock
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA


MODEL_DIR = REPO / "ultralytics" / "cfg" / "models" / "26"
EXPECTED_BASE_PARAMETERS = 2_572_280


def make_args(gain: float = 0.0, target: str = "native", permutation: str = "none"):
    """Return a complete default training namespace with explicit SARC settings."""
    return get_cfg(
        overrides={
            "epochs": 50,
            "sarc_cal_gain": gain,
            "sarc_cal_target": target,
            "sarc_cal_level_permutation": permutation,
            "o2o_kd_gain": 0.0,
            "tqgb_cls_m2o_gain": 0.0,
            "tqgb_cls_o2m_gain": 0.0,
            "tqgb_direction": "none",
        }
    )


def make_batch() -> dict[str, torch.Tensor]:
    """Build a deterministic two-image CPU detection batch."""
    image = torch.linspace(0, 1, 2 * 3 * 64 * 64, dtype=torch.float32).reshape(2, 3, 64, 64)
    return {
        "img": image,
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.30, 0.35, 0.22, 0.18], [0.68, 0.62, 0.25, 0.20]]),
    }


def randomize_bn(module: nn.Module) -> None:
    """Give every BN nontrivial frozen statistics for fusion checks."""
    for bn in (m for m in module.modules() if isinstance(m, nn.BatchNorm2d)):
        with torch.no_grad():
            bn.running_mean.uniform_(-0.7, 0.7)
            bn.running_var.uniform_(0.2, 2.0)
            bn.weight.uniform_(0.4, 1.4)
            bn.bias.uniform_(-0.3, 0.3)


def ddp_histogram_worker(rank: int, world_size: int, port: int) -> None:
    """Exercise the real Gloo SUM collective followed by identical per-rank EMA updates."""
    os.environ["USE_LIBUV"] = "0"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        block = SARCBlock(4, 4)
        values = torch.tensor([1.8, 2.2, 2.9]) if rank == 0 else torch.tensor([3.2, 3.8, 4.1, 4.4])
        counts, mass = block.assignment_soft_counts(values)
        statistics = torch.cat((counts, mass.reshape(1)))
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        block.update_assignment_counts(statistics[:3], statistics[3])
        state = torch.cat(
            (block.sarc_hist_ema, block.sarc_hist_mass.reshape(1), block.sarc_hist_updates.float().reshape(1))
        )
        gathered = [torch.empty_like(state) for _ in range(world_size)]
        dist.all_gather(gathered, state)
        if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
            raise AssertionError("SARC EMA state diverged across CPU DDP ranks")
    finally:
        dist.destroy_process_group()


class SARCIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(19092026)

    def test_real_yamls_parse_and_wire_p3_p4_p5(self) -> None:
        baseline = DetectionModel(str(MODEL_DIR / "yolo26.yaml"), verbose=False)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), EXPECTED_BASE_PARAMETERS)
        self.assertFalse(any(getattr(m, "sarc_mode", None) for m in baseline.modules()))

        cases = {
            "r1": (SARCFixedBlock, 23_747),
            "r2": (SARCBlock, 38_540),
            "r3": (SARCBlock, 38_540),
        }
        for name, (block_type, parameter_delta) in cases.items():
            with self.subTest(name=name):
                model = DetectionModel(str(MODEL_DIR / f"yolo26n-sarc-{name}.yaml"), verbose=False)
                blocks = [model.model[i] for i in (23, 24, 25)]
                self.assertTrue(all(isinstance(block, block_type) for block in blocks))
                self.assertEqual([block.channels for block in blocks], [64, 128, 256])
                self.assertEqual(model.model[-1].f, [23, 24, 25])
                self.assertEqual(model.stride.tolist(), [8.0, 16.0, 32.0])
                self.assertEqual(
                    sum(p.numel() for p in model.parameters()), EXPECTED_BASE_PARAMETERS + parameter_delta
                )

    def test_true_detection_loss_forward_backward_and_exact_zero_controls(self) -> None:
        for name, gain in (("r1", 0.0), ("r2", 0.0), ("r3", 0.01)):
            with self.subTest(name=name):
                torch.manual_seed(101)
                model = DetectionModel(str(MODEL_DIR / f"yolo26n-sarc-{name}.yaml"), nc=3, verbose=False)
                model.args = make_args(gain=gain)
                model.train()
                loss, items = model(make_batch())
                self.assertTrue(torch.isfinite(loss).all())
                self.assertIn("sarc_cal_loss", items)
                if name in {"r1", "r2"}:
                    self.assertEqual(items["sarc_cal_loss"].item(), 0.0)
                else:
                    self.assertEqual(items["sarc_cal_loss"].item(), 0.0)  # 16-update collection warmup
                    self.assertTrue(all(block.sarc_hist_updates.item() == 1 for block in model.model[23:26]))
                    with torch.no_grad():
                        for block, target in zip(
                            model.model[23:26],
                            (
                                torch.tensor([0.8, 0.1, 0.1]),
                                torch.tensor([0.1, 0.8, 0.1]),
                                torch.tensor([0.1, 0.1, 0.8]),
                            ),
                        ):
                            block.sarc_hist_ema.copy_(target)
                            block.sarc_hist_updates.fill_(16)
                    self.assertGreater(model.criterion._sarc_calibration_loss().item(), 0.0)
                loss.sum().backward()
                sarc_gradients = [
                    parameter.grad
                    for module in model.model[23:26]
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(any(gradient is not None for gradient in sarc_gradients))
                self.assertTrue(all(torch.isfinite(g).all() for g in sarc_gradients if g is not None))

    def test_r0_off_has_no_sarc_loss_or_state(self) -> None:
        torch.manual_seed(202)
        model = DetectionModel(str(MODEL_DIR / "yolo26.yaml"), nc=3, verbose=False)
        model.args = make_args()
        loss, items = model(make_batch())
        self.assertTrue(torch.isfinite(loss).all())
        self.assertNotIn("sarc_cal_loss", items)
        self.assertFalse(any("sarc_" in key for key in model.state_dict()))
        loss.sum().backward()

    def test_standalone_and_real_model_fusion(self) -> None:
        for block in (SARCFixedBlock(6, 6, gamma_init=-0.17), SARCBlock(7, 7, gamma_init=0.23)):
            block = block.double().eval()
            randomize_bn(block)
            x = torch.randn(2, block.channels, 17, 19, dtype=torch.float64)
            with torch.no_grad():
                reference = block(x)
                deploy = block.to_deploy()
                converted = deploy(x)
            self.assertLess((reference - converted).abs().max().item(), 2e-12)
            self.assertIsInstance(deploy, SARCDeployBlock)
            self.assertIn("gamma", dict(deploy.named_buffers()))
            self.assertNotIn("gamma", dict(deploy.named_parameters()))
            block.fuse()
            with torch.no_grad():
                in_place = block(x)
            self.assertLess((reference - in_place).abs().max().item(), 2e-12)
            self.assertIn("gamma", dict(block.named_buffers()))
            self.assertNotIn("gamma", dict(block.named_parameters()))

        torch.manual_seed(303)
        model = DetectionModel(str(MODEL_DIR / "yolo26n-sarc-r2.yaml"), nc=3, verbose=False).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            reference = model(x)[0]
            fused_model = copy.deepcopy(model).fuse(verbose=False)
            converted = fused_model(x)[0]
        self.assertLess((reference - converted).abs().max().item(), 2e-4)
        for block in fused_model.model[23:26]:
            self.assertTrue(hasattr(block, "reparam"))
            self.assertIn("gamma", dict(block.named_buffers()))

        fused_baseline = DetectionModel(str(MODEL_DIR / "yolo26.yaml"), verbose=False).eval().fuse(verbose=False)
        baseline_parameters = sum(parameter.numel() for parameter in fused_baseline.parameters())
        for name in ("r1", "r2", "r3"):
            fused = (
                DetectionModel(str(MODEL_DIR / f"yolo26n-sarc-{name}.yaml"), verbose=False)
                .eval()
                .fuse(verbose=False)
            )
            self.assertEqual(sum(parameter.numel() for parameter in fused.parameters()) - baseline_parameters, 22_400)

    def test_sum_counts_then_ema_matches_concatenated_cpu_reference(self) -> None:
        shard_a = torch.tensor([1.8, 2.2, 2.9])
        shard_b = torch.tensor([3.2, 3.8, 4.1, 4.4])
        reduced = SARCBlock(4, 4)
        replica = SARCBlock(4, 4)
        direct = SARCBlock(4, 4)
        count_a, mass_a = reduced.assignment_soft_counts(shard_a)
        count_b, mass_b = reduced.assignment_soft_counts(shard_b)
        reduced.update_assignment_counts(count_a + count_b, mass_a + mass_b)
        replica.update_assignment_counts(count_a + count_b, mass_a + mass_b)
        counts, mass = direct.assignment_soft_counts(torch.cat((shard_a, shard_b)))
        direct.update_assignment_counts(counts, mass)
        self.assertTrue(torch.equal(reduced.sarc_hist_ema, replica.sarc_hist_ema))
        self.assertTrue(torch.equal(reduced.sarc_hist_mass, replica.sarc_hist_mass))
        self.assertTrue(torch.allclose(reduced.sarc_hist_ema, direct.sarc_hist_ema, atol=1e-7, rtol=0))

    def test_real_two_rank_cpu_ddp_sum_then_ema_is_bitwise_synchronized(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(ddp_histogram_worker, args=(2, port), nprocs=2, join=True)

    def test_target_overrides_are_opt_in_and_fail_fast(self) -> None:
        model = DetectionModel(str(MODEL_DIR / "yolo26n-sarc-r3.yaml"), nc=3, verbose=False)
        model.args = make_args(gain=0.01, target="uniform")
        criterion = model.init_criterion()
        self.assertEqual(criterion.sarc_cal_target, "uniform")

        model.args = make_args(gain=0.01, target="permute", permutation="2,1,0")
        criterion = model.init_criterion()
        self.assertEqual(criterion.sarc_level_permutation, (2, 1, 0))

        model.args = make_args(gain=0.0, target="uniform")
        with self.assertRaisesRegex(ValueError, "positive sarc_cal_gain"):
            model.init_criterion()
        model.args = make_args(gain=0.01, target="permute", permutation="none")
        with self.assertRaisesRegex(ValueError, "permutation"):
            model.init_criterion()

    def test_calibration_buffers_round_trip_in_model_state(self) -> None:
        source = DetectionModel(str(MODEL_DIR / "yolo26n-sarc-r3.yaml"), nc=3, verbose=False)
        with torch.no_grad():
            for index, block in enumerate(source.model[23:26], 1):
                block.sarc_hist_ema.copy_(torch.tensor([0.1 * index, 0.2, 0.8 - 0.1 * index]))
                block.sarc_hist_updates.fill_(10 + index)
                block.sarc_hist_mass.fill_(100 + index)
        state = copy.deepcopy(source.state_dict())
        target = DetectionModel(str(MODEL_DIR / "yolo26n-sarc-r3.yaml"), nc=3, verbose=False)
        target.load_state_dict(state)
        for source_block, target_block in zip(source.model[23:26], target.model[23:26]):
            for name in SARCBlock.sarc_state_names:
                self.assertTrue(torch.equal(getattr(source_block, name), getattr(target_block, name)))

    def test_full_checkpoint_saves_live_calibration_state_not_model_ema_state(self) -> None:
        model = DetectionModel(str(MODEL_DIR / "yolo26n-sarc-r3.yaml"), nc=3, verbose=False)
        model.args = make_args(gain=0.01)
        ema = ModelEMA(model)
        with torch.no_grad():
            for index, block in enumerate(model.model[23:26], 1):
                block.sarc_hist_ema.copy_(torch.tensor([0.5, 0.25, 0.25]))
                block.sarc_hist_updates.fill_(20 + index)
                block.sarc_hist_mass.fill_(128 * index)

        with tempfile.TemporaryDirectory() as directory:
            trainer = object.__new__(BaseTrainer)
            trainer.model = model
            trainer.ema = ema
            trainer.epoch = 0
            trainer.best_fitness = trainer.fitness = 0.0
            trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            trainer.scaler = SimpleNamespace(state_dict=lambda: {})
            trainer.args = model.args
            trainer.metrics = {}
            recorded_results = {"epoch": [1.0], "train/sarc_cal_loss": [0.0125]}
            trainer.read_results_csv = lambda: recorded_results
            trainer.wdir = Path(directory)
            trainer.last = trainer.wdir / "last.pt"
            trainer.best = trainer.wdir / "best.pt"
            trainer.save_period = -1
            trainer.save_model()
            checkpoint = torch.load(trainer.last, map_location="cpu", weights_only=False)
            checkpoint_model = checkpoint["ema"]

        self.assertEqual(checkpoint["train_results"]["train/sarc_cal_loss"], [0.0125])

        for live, saved in zip(model.model[23:26], checkpoint_model.model[23:26]):
            self.assertTrue(torch.equal(live.sarc_hist_ema.half(), saved.sarc_hist_ema))
            self.assertTrue(torch.equal(live.sarc_hist_updates, saved.sarc_hist_updates))
            self.assertTrue(torch.equal(live.sarc_hist_mass.half(), saved.sarc_hist_mass))


if __name__ == "__main__":
    unittest.main(verbosity=2)
