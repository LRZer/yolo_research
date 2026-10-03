"""G6: split global/crop BN statistics, with all trainable parameters shared.

Reuse the frozen G4 data, exact sampler, optimization and checkpoint loop. Only
the BN normalization partitions change; affine tensors remain tied. Historical
modules are imported, never edited. All convolutions still see one 2N batch.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

import train_cub_global_local_ddp as paired

ROOT = paired.ROOT
PAIR_SHA = '55a65aa4717c531c852ad68f8f1523be10c8324aa138874c1211c64fabf6c464'
GROUP = 'G6_splitstats_global_crop'
ARCHITECTURE = 'shared_resnet50_splitstats_syncbn_shared_affine_concat4096'
CONTROL_CONFIG = ROOT / 'configs/cub_resnet50_G4_global_crop_s2005_e125_ddp23.json'
CONTROL_CONFIG_SHA = 'f3581f75fe29751a7a27667c11c08745cd91e7bf0284caa29d126b201a820ba7'
CONTROL_CSV_SHA = '6e1dd318f70c4d6b16e64643481e6188c3e2a891a3f22857f768ac26f0184a62'
CONTROL_CKPT_SHA = {
    'last.pt': 'a77931f95f0ee2d4017fd05077adf9ad38b634ed1837d1752a95bc5c63da25b0',
    'best.pt': '193149be5fa9233eb8cd7e035e708e12b354752bd7366015053938fd47535ae2',
}
ORIGINAL_MODEL = paired.SharedResNetPair
ORIGINAL_CONFIG_LOAD = paired.config_load
ORIGINAL_SOURCE_HASHES = paired.source_hashes
ORIGINAL_TRAIN_EPOCH = paired.train_epoch
SMOKE = False
SMOKE_DIAGNOSTICS = {}


class SplitStatsBatchNorm(torch.nn.Module):
    """Independent view statistics; identical, aliased affine Parameter objects."""
    def __init__(self, source, sync=True):
        super().__init__()
        if not source.affine or not source.track_running_stats:
            raise ValueError('The frozen backbone requires affine and running statistics')
        cls = torch.nn.SyncBatchNorm if sync else torch.nn.BatchNorm2d
        factory = dict(device=source.weight.device, dtype=source.weight.dtype)
        self.global_stats = cls(source.num_features, eps=source.eps,
                                momentum=source.momentum, affine=True,
                                track_running_stats=True, **factory)
        self.local_stats = cls(source.num_features, eps=source.eps,
                               momentum=source.momentum, affine=True,
                               track_running_stats=True, **factory)
        # Native BN kernels apply affine, avoiding an extra half-precision
        # multiply/add path. DDP and SGD see each aliased Parameter only once.
        for branch in (self.global_stats, self.local_stats):
            branch.weight = source.weight
            branch.bias = source.bias
            branch.running_mean.copy_(source.running_mean)
            branch.running_var.copy_(source.running_var)
            branch.num_batches_tracked.copy_(source.num_batches_tracked)
        self.train(source.training)

    def forward(self, x):
        if x.ndim != 4 or x.shape[0] == 0 or x.shape[0] % 2:
            raise ValueError('BN expects [all global views, all local views] with equal counts')
        n = x.shape[0] // 2
        return torch.cat((self.global_stats(x[:n]), self.local_stats(x[n:])), dim=0)


def replace_statistics(module, sync):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, SplitStatsBatchNorm(child, sync=sync))
            count += 1
        else:
            count += replace_statistics(child, sync)
    return count


class SplitStatsResNetPair(ORIGINAL_MODEL):
    def __init__(self, sync=True):
        # Exactly the original random initialization order and parameter objects.
        super().__init__(sync=False)
        self.split_layer_count = replace_statistics(self.encoder, sync)
        if self.split_layer_count != 53:
            raise AssertionError('Unexpected ResNet50 BN layer layout')


def config_load(path):
    raw = Path(path).read_bytes()
    c = json.loads(raw)
    control, control_sha = ORIGINAL_CONFIG_LOAD(CONTROL_CONFIG)
    if control_sha != CONTROL_CONFIG_SHA or paired.engine.sha256(Path(paired.__file__)) != PAIR_SHA:
        raise RuntimeError('Frozen G4 code or configuration changed')
    expected = {**control, 'group': GROUP, 'architecture': ARCHITECTURE,
                'run_dir': str(ROOT / 'runs/resnet50_G6_splitstats_global_crop_native500_s2005_e125_ddp23'),
                'bn_statistics': 'per_view', 'bn_affine': 'shared'}
    if c != expected:
        raise ValueError('G6 must differ from G4 only by split BN statistics and run identity')
    return c, hashlib.sha256(raw).hexdigest()


def source_hashes():
    old = ORIGINAL_SOURCE_HASHES()
    return {**old, 'trainer': paired.engine.sha256(Path(__file__)), 'paired_trainer': old['trainer']}


def assert_aliases(model):
    layers = [m for m in model.modules() if isinstance(m, SplitStatsBatchNorm)]
    assert len(layers) == 53
    for m in layers:
        assert m.global_stats.weight is m.local_stats.weight
        assert m.global_stats.bias is m.local_stats.bias
        assert m.global_stats.running_mean is not m.local_stats.running_mean
        assert m.global_stats.running_var is not m.local_stats.running_var
        assert m.global_stats.num_batches_tracked is not m.local_stats.num_batches_tracked
    return layers


def cpu_bn_check():
    torch.manual_seed(2005)
    control = ORIGINAL_MODEL(sync=False)
    torch.manual_seed(2005)
    candidate = SplitStatsResNetPair(sync=False)
    before, after = list(control.parameters()), list(candidate.parameters())
    assert len(before) == len(after)
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    count = sum(p.numel() for p in after)
    assert count == 24_327_432
    layers = assert_aliases(candidate)
    # Check buffer initialization mapping, and serialization preserves tying.
    original_bn = [m for m in control.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for old, new in zip(original_bn, layers, strict=True):
        for branch in (new.global_stats, new.local_stats):
            assert torch.equal(old.running_mean, branch.running_mean)
            assert torch.equal(old.running_var, branch.running_var)
    stream = io.BytesIO()
    torch.save(candidate.state_dict(), stream)
    stream.seek(0)
    candidate.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert_aliases(candidate)
    del control, candidate, before, after, stream

    source = torch.nn.BatchNorm2d(3, momentum=.1).double()
    with torch.no_grad():
        source.weight.copy_(torch.tensor([.9, 1.2, .7], dtype=torch.float64))
        source.bias.copy_(torch.tensor([.1, -.2, .3], dtype=torch.float64))
    split = SplitStatsBatchNorm(source, sync=False)
    reference_g, reference_l = copy.deepcopy(source), copy.deepcopy(source)
    xg = (torch.randn(4, 3, 5, 4, dtype=torch.float64) + 2).requires_grad_()
    xl = (torch.randn(4, 3, 5, 4, dtype=torch.float64) * 3 + 10).requires_grad_()
    rg, rl = xg.detach().clone().requires_grad_(), xl.detach().clone().requires_grad_()
    output = split(torch.cat((xg, xl)))
    reference = torch.cat((reference_g(rg), reference_l(rl)))
    torch.testing.assert_close(output, reference, rtol=1e-12, atol=1e-12)
    coefficients = torch.randn_like(output)
    (output * coefficients).sum().backward()
    (reference * coefficients).sum().backward()
    torch.testing.assert_close(xg.grad, rg.grad, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(xl.grad, rl.grad, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(split.global_stats.weight.grad,
                               reference_g.weight.grad + reference_l.weight.grad, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(split.global_stats.bias.grad,
                               reference_g.bias.grad + reference_l.bias.grad, rtol=1e-11, atol=1e-11)
    for branch, ref in ((split.global_stats, reference_g), (split.local_stats, reference_l)):
        torch.testing.assert_close(branch.running_mean, ref.running_mean, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(branch.running_var, ref.running_var, rtol=1e-12, atol=1e-12)
    split.eval()
    reference_g.eval()
    reference_l.eval()
    torch.testing.assert_close(split(torch.cat((xg.detach(), xl.detach()))),
                               torch.cat((reference_g(xg.detach()), reference_l(xl.detach()))),
                               rtol=1e-12, atol=1e-12)
    independent_a = SplitStatsBatchNorm(copy.deepcopy(source), sync=False).train()
    independent_b = SplitStatsBatchNorm(copy.deepcopy(source), sync=False).train()
    oa = independent_a(torch.cat((xg.detach(), xl.detach())))[:4]
    ob = independent_b(torch.cat((xg.detach(), xl.detach() * 100)))[:4]
    assert torch.equal(oa, ob)
    assert not torch.equal(split.global_stats.running_mean, split.local_stats.running_mean)
    return dict(parameter_count=count, split_bn_layers=53, all_initial_parameters_identical_to_g4=True,
                affine_objects_shared=True, independent_running_buffers=True,
                native_bn_forward_and_gradients_verified=True,
                training_global_output_independent_of_local_input=True,
                eval_separate_statistics_verified=True, serialization_preserves_parameter_tying=True,
                per_view_bn_samples_per_global_microbatch=16, g4_mixed_bn_samples=32)


def control_check(out):
    c, digest = ORIGINAL_CONFIG_LOAD(CONTROL_CONFIG)
    assert digest == CONTROL_CONFIG_SHA
    run = Path(c['run_dir'])
    audit = json.loads((run / 'audit.json').read_text())
    assert audit['status'] == 'verified_native_e125_ddp_pair'
    assert audit['config_sha256'] == digest and audit['source_sha256'] == ORIGINAL_SOURCE_HASHES()
    assert paired.engine.sha256(run / 'results.csv') == audit['csv_sha256'] == CONTROL_CSV_SHA
    rows = paired.rows_load(run / 'results.csv')
    assert len(rows) == 125 and rows[-1] == audit['final_row']
    for name, digest_ckpt in CONTROL_CKPT_SHA.items():
        assert paired.engine.sha256(run / name) == digest_ckpt == audit['checkpoints'][name]['sha256']
        checkpoint = torch.load(run / name, map_location='cpu', weights_only=False)
        assert checkpoint['config'] == c and checkpoint['config_sha256'] == digest
        assert checkpoint['source_sha256'] == ORIGINAL_SOURCE_HASHES()
        assert all(torch.isfinite(x).all().item() for x in checkpoint['model'].values())
        model = ORIGINAL_MODEL(sync=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        ep = checkpoint['completed_epoch']
        assert ep == audit['checkpoints'][name]['epoch']
        assert checkpoint['last_row'] == {k: float(rows[ep - 1][k]) for k in paired.CSV_FIELDS}
        del checkpoint, model
    paired.inputs(c)
    result = dict(status='verified_readonly_g4_control', config_sha256=digest,
                  csv_sha256=CONTROL_CSV_SHA, checkpoint_sha256=CONTROL_CKPT_SHA,
                  source_sha256=ORIGINAL_SOURCE_HASHES(), historical_audit=audit)
    paired.engine.atomic_json(out, result)
    print(json.dumps(result), flush=True)


def ddp_bn_check(out):
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3' and int(os.environ['WORLD_SIZE']) == 2 and local == rank
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=dt.timedelta(minutes=3), device_id=device)
    torch.manual_seed(2005)
    source = torch.nn.BatchNorm2d(3, momentum=.1).to(device)
    split = SplitStatsBatchNorm(source, sync=True).to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(split, device_ids=[local])
    xg = (torch.arange(24, device=device, dtype=torch.float32).reshape(2, 3, 2, 2) / 13 + rank * 4).requires_grad_()
    xl = (torch.arange(24, device=device, dtype=torch.float32).reshape(2, 3, 2, 2) / 7 + 20 + rank * 2).requires_grad_()
    gathered_g, gathered_l = [torch.empty_like(xg) for _ in range(2)], [torch.empty_like(xl) for _ in range(2)]
    dist.all_gather(gathered_g, xg.detach())
    dist.all_gather(gathered_l, xl.detach())
    full_g, full_l = torch.cat(gathered_g), torch.cat(gathered_l)
    rg, rl = full_g.clone().requires_grad_(), full_l.clone().requires_grad_()
    rw, rb = source.weight.detach().clone().requires_grad_(), source.bias.detach().clone().requires_grad_()
    reference_g = F.batch_norm(rg, None, None, rw, rb, training=True, eps=source.eps)
    reference_l = F.batch_norm(rl, None, None, rw, rb, training=True, eps=source.eps)
    output = ddp(torch.cat((xg, xl)))
    expected = torch.cat((reference_g[rank * 2:(rank + 1) * 2], reference_l[rank * 2:(rank + 1) * 2]))
    torch.testing.assert_close(output, expected, rtol=3e-5, atol=3e-5)
    output.square().mean().backward()
    torch.cat((reference_g, reference_l)).square().mean().backward()
    torch.testing.assert_close(split.global_stats.weight.grad, rw.grad, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(split.global_stats.bias.grad, rb.grad, rtol=3e-5, atol=3e-5)
    # BN input gradients are local-objective sums before DDP parameter averaging.
    torch.testing.assert_close(xg.grad / 2, rg.grad[rank * 2:(rank + 1) * 2], rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(xl.grad / 2, rl.grad[rank * 2:(rank + 1) * 2], rtol=5e-5, atol=5e-6)
    for branch, values in ((split.global_stats, full_g), (split.local_stats, full_l)):
        torch.testing.assert_close(branch.running_mean, values.mean((0, 2, 3)) * .1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(branch.running_var, .9 + values.var((0, 2, 3), unbiased=True) * .1,
                                   rtol=1e-5, atol=1e-5)
    split.eval()
    reference_eval = torch.cat((F.batch_norm(xg.detach(), split.global_stats.running_mean,
                                            split.global_stats.running_var, source.weight, source.bias,
                                            training=False, eps=source.eps),
                               F.batch_norm(xl.detach(), split.local_stats.running_mean,
                                            split.local_stats.running_var, source.weight, source.bias,
                                            training=False, eps=source.eps)))
    torch.testing.assert_close(split(torch.cat((xg.detach(), xl.detach()))), reference_eval, rtol=1e-5, atol=1e-5)
    dist.barrier()
    if rank == 0:
        result = dict(status='verified_splitstats_syncbn_ddp', world_size=2,
                      synchronized_per_view_moments=True, shared_affine_ddp_gradients=True,
                      input_gradients_verified=True, independent_eval_buffers=True,
                      source_sha256=source_hashes())
        paired.engine.atomic_json(out, result)
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_epoch(*args, **kwargs):
    values = ORIGINAL_TRAIN_EPOCH(*args, **kwargs)
    if SMOKE:
        model = args[0].module
        layers = assert_aliases(model)
        assert all(m.global_stats.num_batches_tracked.item() == 11 and
                   m.local_stats.num_batches_tracked.item() == 11 for m in layers)
        digests = [None, None]
        dist.all_gather_object(digests, state_digest(model))
        assert digests[0] == digests[1]
        SMOKE_DIAGNOSTICS.update(all_model_parameters_and_buffers_identical_across_ranks=True,
                                 model_state_sha256=digests[0], bn_microbatches_each_view=11)
    return values


def audit(c, digest):
    result = paired.audit(c, digest)
    for name in ('last.pt', 'best.pt'):
        checkpoint = torch.load(Path(c['run_dir']) / name, map_location='cpu', weights_only=False)
        model = SplitStatsResNetPair(sync=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        layers = assert_aliases(model)
        expected_batches = checkpoint['completed_epoch'] * len(paired.ExactPairSampler(5994, 2005, 1, 0))
        for key, value in checkpoint['model'].items():
            if '.global_stats.' in key and (key.endswith('.weight') or key.endswith('.bias')):
                assert torch.equal(value, checkpoint['model'][key.replace('.global_stats.', '.local_stats.')])
        assert all(m.global_stats.num_batches_tracked.item() == expected_batches and
                   m.local_stats.num_batches_tracked.item() == expected_batches for m in layers)
        del model, checkpoint
    result.update(status='verified_native_e125_splitstats', parameter_count=24_327_432,
                  split_bn_layers=53, bn_affine='shared', bn_statistics='per_view',
                  control_config_sha256=CONTROL_CONFIG_SHA, control_csv_sha256=CONTROL_CSV_SHA)
    paired.engine.atomic_json(Path(c['run_dir']) / 'audit.json', result)
    print(json.dumps(result), flush=True)
    return result


def install_hooks():
    paired.SharedResNetPair = SplitStatsResNetPair
    paired.config_load = config_load
    paired.source_hashes = source_hashes
    paired.train_epoch = train_epoch


def main():
    global SMOKE
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('check', 'control-check', 'norm-smoke', 'smoke', 'train', 'audit'):
        mode.add_argument('--' + flag, action='store_true')
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    c, digest = config_load(args.config)
    if args.control_check:
        if not args.out:
            parser.error('--control-check requires --out')
        control_check(args.out)
        return
    install_hooks()
    if args.check:
        if not args.preview or not args.out:
            parser.error('--check requires --preview and --out')
        diagnostics = cpu_bn_check()
        paired.cpu_check(c, digest, args.preview, args.out)
        report = json.loads(args.out.read_text())
        report.update(status='verified_cpu_splitstats_pipeline', normalization_checks=diagnostics)
        paired.engine.atomic_json(args.out, report)
        print(json.dumps(report), flush=True)
    elif args.norm_smoke:
        if not args.out:
            parser.error('--norm-smoke requires --out')
        ddp_bn_check(args.out)
    elif args.audit:
        audit(c, digest)
    else:
        if args.smoke and not args.out:
            parser.error('--smoke requires --out')
        SMOKE = args.smoke
        paired.distributed_run(c, digest, args.smoke, args.out)
        if args.smoke and int(os.environ['RANK']) == 0:
            report = json.loads(args.out.read_text())
            report.update(status='verified_splitstats_ddp_smoke', normalization_checks=SMOKE_DIAGNOSTICS,
                          bn_affine='shared', bn_statistics='per_view')
            paired.engine.atomic_json(args.out, report)
            print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
