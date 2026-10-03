"""GT global/local Oracle classification: exact two-rank sampling, native E125 stop.

G4 uses a full native image and its GT crop. G5 substitutes a constant normalized
zero global view. One shared scratch ResNet50, SyncBN, concatenated features,
and one fused cross-entropy are used in both groups. Original data are read only.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models import resnet50

import train_cub_resnet50_a as engine

ROOT = Path('/home/ubuntu/rz/foveated_classification_research')
STATS_SHA = 'e1caf32dd617b8d91720d117ff8527c75a5dbc1b82c71fbb1eabaecd88a7b309'
BBOX_SHA = 'c19d9f096c15564793408e5ba054b8d715a6805819d98f037c977f369ef4abd3'
ENGINE_SHA = '6aa91ec82be7cee3dc985e2e5824ed8a5470cd7e7f2dc7cfd5e6a1f702a9f8a2'
GROUPS = {'G4_global_crop': 'full', 'G5_blank_crop': 'blank'}
EXTRA_FIELDS = ('update_groups', 'amp_skipped_steps')
CSV_FIELDS = engine.CSV_FIELDS + EXTRA_FIELDS


def config_load(path):
    raw = Path(path).read_bytes()
    c = json.loads(raw)
    expected = dict(architecture='shared_resnet50_syncbn_concat4096', epochs=200,
                    target_epochs=125, seed=2005, canvas=500, effective_batch=64,
                    microbatch_per_rank=8, accumulation=4, eval_batch_per_rank=16,
                    workers=4, workers_per_rank=2, world_size=2, devices_physical=[2, 3],
                    numa_node=0, crop_margin=0, base_lr=0.025, min_lr=0.00001,
                    warmup_epochs=5, momentum=0.9, weight_decay=0.0005, amp_initial_scale=1024)
    for k, v in expected.items():
        if c.get(k) != v:
            raise ValueError(f'Frozen setting {k}: {c.get(k)!r} != {v!r}')
    if c.get('group') not in GROUPS or c.get('global_view') != GROUPS[c['group']]:
        raise ValueError('Invalid group/global view')
    if c['microbatch_per_rank'] * c['world_size'] * c['accumulation'] != c['effective_batch']:
        raise ValueError('Effective batch is not 64 pairs')
    if c['workers_per_rank'] * c['world_size'] != c['workers']:
        raise ValueError('Workers4 means four workers total')
    expected_run = ROOT / 'runs' / f"resnet50_{c['group']}_native500_s2005_e125_ddp23"
    if Path(c['run_dir']) != expected_run:
        raise ValueError('Unexpected task run directory')
    if engine.sha256(Path(engine.__file__)) != ENGINE_SHA:
        raise ValueError('Historical engine changed')
    return c, hashlib.sha256(raw).hexdigest()


def inputs(c):
    stats_path = Path(c['stats_json'])
    if engine.sha256(stats_path) != STATS_SHA:
        raise ValueError('Frozen official-train statistics changed')
    stats = json.loads(stats_path.read_text())
    root = Path(c['data_root'])
    if engine.sha256(root / 'bounding_boxes.txt') != BBOX_SHA:
        raise ValueError('Approved GT boxes changed')
    train, test = engine.load_metadata(root, stats)
    boxes = {i: tuple(map(float, v.split())) for i, v in engine.parse_table(root / 'bounding_boxes.txt').items()}
    if set(boxes) != {r[0] for r in train + test}:
        raise ValueError('GT IDs differ from dataset IDs')
    # These are the frozen stats schema used by the existing A-F engine.
    mean = [x / 255 for x in stats['mean_rgb_0_255']]
    std = [x / 255 for x in stats['std_rgb_0_255']]
    fill = tuple(round(x) for x in stats['mean_rgb_0_255'])
    return train, test, boxes, mean, std, fill


def crop_bounds(raw, width, height):
    x, y, w, h = raw
    if not all(math.isfinite(v) for v in raw) or w <= 0 or h <= 0:
        raise ValueError(f'Invalid box {raw}')
    bounds = (max(0, min(width, math.floor(x))), max(0, min(height, math.floor(y))),
              max(0, min(width, math.ceil(x + w))), max(0, min(height, math.ceil(y + h))))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError(f'Empty clipped box {raw} in {width}x{height}')
    return bounds


def padded(image, fill):
    w, h = image.size
    if not 0 < w <= 500 or not 0 < h <= 500:
        raise ValueError(f'Invalid native size {image.size}')
    canvas = Image.new('RGB', (500, 500), fill)
    canvas.paste(image, ((500 - w) // 2, (500 - h) // 2))
    return canvas


class PairDataset(Dataset):
    def __init__(self, records, boxes, mean, std, fill, c, train, epoch=0):
        self.records, self.boxes = records, boxes
        self.mean, self.std, self.fill = mean, std, fill
        self.c, self.train, self.epoch = c, train, epoch

    def __len__(self):
        return len(self.records)

    def views(self, index):
        identifier, path, label = self.records[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        w, h = image.size
        # Clip and rasterize BEFORE reflection: flipping this half-open integer
        # crop gives an exact reflected pixel set, without fractional rounding drift.
        l, t, r, b = crop_bounds(self.boxes[identifier], w, h)
        if self.train:
            rng = random.Random(self.c['seed'] * 1_000_003 + self.epoch * 100_003 + identifier)
            if rng.random() < 0.5:
                image = TF.hflip(image)
                l, r = w - r, w - l
            image = TF.adjust_brightness(image, rng.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, rng.uniform(0.9, 1.1))
            image = TF.adjust_saturation(image, rng.uniform(0.9, 1.1))
        crop = image.crop((l, t, r, b))
        scale = 500 / max(crop.size)
        size = tuple(max(1, min(500, round(v * scale))) for v in crop.size)
        local = padded(crop.resize(size, Image.Resampling.BICUBIC), self.fill)
        return padded(image, self.fill), local, label

    def __getitem__(self, index):
        global_image, local_image, label = self.views(index)
        local_tensor = TF.normalize(TF.to_tensor(local_image), self.mean, self.std)
        global_tensor = (torch.zeros_like(local_tensor) if self.c['global_view'] == 'blank' else
                         TF.normalize(TF.to_tensor(global_image), self.mean, self.std))
        return global_tensor, local_tensor, label


class ExactPairSampler(Sampler):
    """Partition each GLOBAL microbatch; no padded/duplicated/dropped images."""
    def __init__(self, n, seed, epoch, rank, world=2, effective=64, micro=8):
        if n % world:
            raise ValueError('This exact paired sampler requires count divisible by ranks')
        gen = torch.Generator().manual_seed(seed + epoch * 100_003)
        permutation = torch.randperm(n, generator=gen).tolist()
        self.batches, self.info = [], []
        for offset in range(0, n, effective):
            group = permutation[offset:offset + effective]
            chunks = [group[k:k + micro * world] for k in range(0, len(group), micro * world)]
            for k, chunk in enumerate(chunks):
                if len(chunk) % world:
                    raise AssertionError('Unequal rank microbatch')
                self.batches.append(chunk[rank::world])
                self.info.append((len(group), k == len(chunks) - 1))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class SharedResNetPair(torch.nn.Module):
    def __init__(self, sync=True):
        super().__init__()
        self.encoder = resnet50(weights=None)
        self.encoder.fc = torch.nn.Identity()
        self.fusion = torch.nn.Linear(4096, 200)
        if sync:
            self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

    def forward(self, global_view, crop_view):
        n = global_view.shape[0]
        encoded = self.encoder(torch.cat((global_view, crop_view), dim=0))
        return self.fusion(torch.cat((encoded[:n], encoded[n:]), dim=1))


def source_hashes():
    return {'trainer': engine.sha256(Path(__file__)), 'historical_engine': ENGINE_SHA,
            'stats': STATS_SHA, 'bbox': BBOX_SHA}


def cpu_check(c, config_sha, preview, out):
    train, test, boxes, mean, std, fill = inputs(c)
    bad = []
    for identifier, path, _ in train + test:
        with Image.open(path) as image:
            crop_bounds(boxes[identifier], *image.size)
            if max(image.size) > 500:
                bad.append(identifier)
    assert not bad
    samplers = [ExactPairSampler(5994, 2005, 1, r) for r in range(2)]
    flat = [i for s in samplers for batch in s for i in batch]
    assert sorted(flat) == list(range(5994)) and len(set(flat)) == 5994
    assert samplers[0].info == samplers[1].info
    assert sum(end for _, end in samplers[0].info) == 94
    assert [len(b) for b in samplers[0].batches[-3:]] == [8, 8, 5]
    assert samplers[0].info[-1] == (42, True)
    # Analytic DDP-average plus accumulation check against full-batch mean.
    errors = []
    torch.manual_seed(2005)
    for count in (64, 42):
        x, y = torch.randn(count, 7, dtype=torch.float64), torch.randn(count, 3, dtype=torch.float64)
        w = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
        ref = torch.autograd.grad(((x @ w - y) ** 2).mean(dim=1).mean(), w)[0]
        parts = []
        for rank in range(2):
            g = torch.zeros_like(w)
            for k in range(0, count, 16):
                ids = torch.arange(k, min(k + 16, count))[rank::2]
                loss = ((x[ids] @ w - y[ids]) ** 2).mean(dim=1).sum() * 2 / count
                g += torch.autograd.grad(loss, w)[0]
            parts.append(g)
        error = float((torch.stack(parts).mean(0) - ref).abs().max())
        assert error < 1e-12
        errors.append(error)
    dataset = PairDataset(train, boxes, mean, std, fill, c, False)
    manifest = json.loads((preview / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['bbox_file_sha256'] == BBOX_SHA
    by_id = {r[0]: i for i, r in enumerate(train)}
    matched = []
    for sample in manifest['samples']:
        identifier = sample['image_id']
        folder = next(preview.glob(f'{identifier:05d}_*'))
        assert engine.sha256(train[by_id[identifier]][1]) == sample['source_sha256']
        global_image, local_image, _ = dataset.views(by_id[identifier])
        for image, name in ((global_image, 'global_input_500.png'), (local_image, 'local_input_500.png')):
            assert engine.sha256(folder / name) == sample['output_sha256'][name]
            with Image.open(folder / name) as expected:
                assert image.size == expected.size and image.tobytes() == expected.convert('RGB').tobytes()
        matched.append(identifier)
    for identifier in matched:
        index = by_id[identifier]
        augmented = PairDataset(train, boxes, mean, std, fill, c, True, 7)
        other = {**c, 'global_view': 'blank' if c['global_view'] == 'full' else 'full'}
        control = PairDataset(train, boxes, mean, std, fill, other, True, 7)
        a, b = augmented[index], augmented[index]
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert torch.equal(a[1], control[index][1])
        blank = a[0] if c['global_view'] == 'blank' else control[index][0]
        assert torch.count_nonzero(blank).item() == 0
        path = train[index][1]
        with Image.open(path) as original:
            original = original.convert('RGB')
        l, t, r, bottom = crop_bounds(boxes[identifier], *original.size)
        reflected = TF.hflip(original).crop((original.width - r, t, original.width - l, bottom))
        assert reflected.tobytes() == TF.hflip(original.crop((l, t, r, bottom))).tobytes()
    report = dict(status='verified_cpu_pair_pipeline', group=c['group'], config_sha256=config_sha,
                  source_sha256=source_hashes(), train_count=5994, test_count=5794,
                  boxes_checked=11788, preview_ids_matched=matched,
                  exact_updates_per_epoch=94, tail_group_pairs=42, tail_per_rank_microbatches=[8, 8, 5],
                  gradient_max_abs_errors=errors, same_crop_across_groups=True, blank_exact_zero=True,
                  torch=torch.__version__, torchvision=torchvision.__version__, pillow=PIL.__version__)
    engine.atomic_json(out, report)
    print(json.dumps(report), flush=True)


def reduced_metrics(totals):
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return engine.aggregate(totals)


def train_epoch(model, dataset, c, epoch, rank, device, optimizer, scaler):
    sampler = ExactPairSampler(len(dataset), c['seed'], epoch, rank)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=c['workers_per_rank'],
                        pin_memory=True, persistent_workers=False)
    model.train()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    updates, skipped, groups = 0, 0, 0
    for (global_view, crop_view, targets), (group_target, end) in zip(loader, sampler.info, strict=True):
        global_view, crop_view = global_view.to(device, non_blocking=True), crop_view.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sync = contextlib.nullcontext() if end else model.no_sync()
        with sync:
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(global_view, crop_view)
                per_pair = F.cross_entropy(logits.float(), targets, reduction='none')
            if not bool(torch.isfinite(per_pair).all()):
                raise FloatingPointError('Nonfinite pair loss')
            # DDP averages rank gradients: multiply local SUM by world size.
            scaler.scale(per_pair.sum() * dist.get_world_size() / group_target).backward()
        correct1, correct5 = engine.topk_counts(logits.detach(), targets)
        totals += torch.stack((per_pair.detach().sum(), correct1, correct5,
                               torch.tensor(targets.numel(), device=device))).to(torch.float64)
        if end:
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            was_skipped = int(scaler.get_scale() < old_scale)
            skipped += was_skipped
            updates += 1 - was_skipped
            groups += 1
            optimizer.zero_grad(set_to_none=True)
    values = reduced_metrics(totals)
    checks = torch.tensor([updates, skipped, groups], device=device, dtype=torch.int64)
    check_min, check_max = checks.clone(), checks.clone()
    dist.all_reduce(check_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(check_max, op=dist.ReduceOp.MAX)
    if not torch.equal(check_min, check_max) or values[3] != len(dataset) or groups != math.ceil(len(dataset) / 64):
        raise AssertionError('Rank update/count mismatch')
    return (*values, updates, skipped, groups)


@torch.no_grad()
def eval_epoch(model, dataset, c, rank, device):
    # Exact non-padding evaluation partition; SyncBN uses running stats in eval.
    subset = torch.utils.data.Subset(dataset, range(rank, len(dataset), 2))
    loader = DataLoader(subset, batch_size=c['eval_batch_per_rank'], num_workers=c['workers_per_rank'],
                        pin_memory=True, persistent_workers=False)
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for global_view, crop_view, targets in loader:
        global_view, crop_view = global_view.to(device, non_blocking=True), crop_view.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model.module(global_view, crop_view)
        losses = F.cross_entropy(logits.float(), targets, reduction='none')
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError('Nonfinite test loss')
        a, b = engine.topk_counts(logits, targets)
        totals += torch.stack((losses.sum(), a, b, torch.tensor(targets.numel(), device=device))).to(torch.float64)
    values = reduced_metrics(totals)
    if values[3] != len(dataset):
        raise AssertionError('Test duplication/drop')
    return values


def rows_load(path):
    import csv
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == CSV_FIELDS
        rows = list(reader)
    assert [int(r['epoch']) for r in rows] == list(range(1, len(rows) + 1))
    assert all(math.isfinite(float(v)) for r in rows for v in r.values())
    return rows


def row_append(path, row):
    import csv
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def audit(c, config_sha):
    run = Path(c['run_dir'])
    rows = rows_load(run / 'results.csv')
    assert len(rows) == c['target_epochs'] == 125
    for row in rows:
        assert int(row['train_images']) == 5994 and int(row['test_images']) == 5794
        assert int(row['update_groups']) == 94
        assert int(row['optimizer_steps']) + int(row['amp_skipped_steps']) == 94
        assert math.isclose(float(row['lr']), engine.learning_rate(int(row['epoch']), c), rel_tol=1e-12)
        for key in ('train_top1', 'train_top5', 'test_top1', 'test_top5'):
            assert 0 <= float(row[key]) <= 100
        assert float(row['train_top1']) <= float(row['train_top5'])
        assert float(row['test_top1']) <= float(row['test_top5'])
    inputs(c)
    models = {}
    for name in ('last.pt', 'best.pt'):
        checkpoint = torch.load(run / name, map_location='cpu', weights_only=False)
        assert checkpoint['config_sha256'] == config_sha and checkpoint['config'] == c
        assert checkpoint['source_sha256'] == source_hashes()
        assert checkpoint['model']['fusion.weight'].shape == (200, 4096)
        assert all(torch.isfinite(x).all().item() for x in checkpoint['model'].values())
        model = SharedResNetPair(sync=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        assert checkpoint['optimizer'] and checkpoint['scaler']
        assert len(checkpoint['rank_rng']) == 2
        ep = checkpoint['completed_epoch']
        assert 1 <= ep <= 125
        assert all(float(checkpoint['last_row'][k]) == float(rows[ep - 1][k]) for k in CSV_FIELDS)
        if name == 'last.pt':
            assert ep == 125
        models[name] = dict(epoch=ep, sha256=engine.sha256(run / name))
        del model, checkpoint
    state = json.loads((run / 'state.json').read_text())
    assert state['status'] == 'completed' and state['completed_epoch'] == 125
    assert state['config_sha256'] == config_sha
    result = dict(status='verified_native_e125_ddp_pair', group=c['group'], config_sha256=config_sha,
                  source_sha256=source_hashes(), csv_sha256=engine.sha256(run / 'results.csv'),
                  checkpoints=models, final_row=rows[-1], actual_epochs=125, lr_horizon=200,
                  amp_skipped_steps_total=sum(int(r['amp_skipped_steps']) for r in rows),
                  mean_epoch_seconds=sum(float(r['epoch_seconds']) for r in rows) / 125,
                  mean_train_seconds=sum(float(r['train_seconds']) for r in rows) / 125,
                  mean_test_seconds=sum(float(r['test_seconds']) for r in rows) / 125)
    engine.atomic_json(run / 'audit.json', result)
    print(json.dumps(result), flush=True)
    return result


def distributed_run(c, config_sha, smoke, out):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
        raise RuntimeError('Only physical GPU2,3 are authorized')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or local_rank != rank:
        raise RuntimeError('Exactly two local GPU ranks required')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=dt.timedelta(minutes=8), device_id=device)
    uuids = ['66f2712d-a00e-2fdb-e6da-be24056fa1d4', 'a47e8163-6e6f-6686-c5ee-c98142f6cb27']
    actual_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if actual_uuid != uuids[rank]:
        raise RuntimeError(f'Rank {rank} did not map to authorized physical GPU: {actual_uuid}')
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, test, boxes, mean, std, fill = inputs(c)
    if smoke:
        generator = torch.Generator().manual_seed(2005)
        train = [train[i] for i in torch.randperm(len(train), generator=generator)[:170].tolist()]
        test = [test[i] for i in torch.randperm(len(test), generator=generator)[:34].tolist()]
    run = Path(c['run_dir'])
    if not smoke and rank == 0:
        if run.exists():
            raise FileExistsError('Fresh launch must not replace an existing run')
        run.mkdir(parents=True)
        engine.atomic_json(run / 'config.json', c)
        engine.atomic_json(run / 'launch_runtime.json', dict(config_sha256=config_sha,
            source_sha256=source_hashes(), rank0_pid=os.getpid(), devices_physical=[2, 3],
            numa_policy=Path('/proc/self/status').read_text().split('Mems_allowed_list:')[-1].splitlines()[0].strip(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            torch=torch.__version__, torchvision=torchvision.__version__, pillow=PIL.__version__,
            shared_gpu=True, oracle_train_and_test=True))
    dist.barrier()
    model = DDP(SharedResNetPair(sync=True).to(device), device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=c['base_lr'], momentum=c['momentum'], weight_decay=c['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=c['amp_initial_scale'])
    best_top1, best_epoch = -1.0, 0
    limit = 1 if smoke else c['target_epochs']
    timings = []
    for epoch in range(1, limit + 1):
        lr = engine.learning_rate(epoch, c)
        for group in optimizer.param_groups:
            group['lr'] = lr
        train_set = PairDataset(train, boxes, mean, std, fill, c, True, epoch)
        test_set = PairDataset(test, boxes, mean, std, fill, c, False)
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        start = time.monotonic()
        tr_loss, tr_1, tr_5, tr_count, steps, skips, groups = train_epoch(model, train_set, c, epoch, rank, device, optimizer, scaler)
        torch.cuda.synchronize(device)
        train_time = time.monotonic() - start
        start_test = time.monotonic()
        te_loss, te_1, te_5, te_count = eval_epoch(model, test_set, c, rank, device)
        torch.cuda.synchronize(device)
        test_time = time.monotonic() - start_test
        measures = torch.tensor([train_time, test_time, torch.cuda.max_memory_allocated(device) / 2**30,
                                 torch.cuda.max_memory_reserved(device) / 2**30], device=device, dtype=torch.float64)
        local_measures = measures.cpu().tolist()
        dist.all_reduce(measures, op=dist.ReduceOp.MAX)
        train_time, test_time, peak, reserved = measures.cpu().tolist()
        row = dict(epoch=epoch, lr=lr, train_loss=tr_loss, train_top1=tr_1, train_top5=tr_5,
                   test_loss=te_loss, test_top1=te_1, test_top5=te_5, train_seconds=train_time,
                   test_seconds=test_time, epoch_seconds=train_time + test_time, gpu_max_memory_gib=peak,
                   train_images=tr_count, test_images=te_count, optimizer_steps=steps,
                   update_groups=groups, amp_skipped_steps=skips)
        if smoke:
            checks = torch.stack([p.detach().double().sum() for p in model.module.fusion.parameters()]).to(device)
            all_checks = [torch.zeros_like(checks) for _ in range(2)]
            dist.all_gather(all_checks, checks)
            assert torch.equal(all_checks[0], all_checks[1])
            rank_reports = [None, None]
            import subprocess
            dist.all_gather_object(rank_reports, dict(rank=rank, pid=os.getpid(), peak_allocated_gib=local_measures[2],
                                                     peak_reserved_gib=local_measures[3], device=str(device),
                                                     physical_gpu=rank + 2, device_uuid=actual_uuid,
                                                     numa_policy=subprocess.check_output(['numactl', '--show'], text=True)))
            if rank == 0:
                result = dict(status='verified_ddp_smoke', group=c['group'], config_sha256=config_sha,
                              source_sha256=source_hashes(), row=row, rank_reports=rank_reports,
                              fusion_identical_across_ranks=True, parameter_count=sum(p.numel() for p in model.parameters()))
                engine.atomic_json(out, result)
                print(json.dumps(result), flush=True)
        else:
            rngs = [None, None]
            dist.all_gather_object(rngs, dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
                                            numpy=np.random.get_state(), python=random.getstate()))
            improved = te_1 > best_top1
            if improved:
                best_top1, best_epoch = te_1, epoch
            if rank == 0:
                checkpoint = dict(completed_epoch=epoch, config=c, config_sha256=config_sha,
                                  source_sha256=source_hashes(), model=model.module.state_dict(),
                                  optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                  rank_rng=rngs, last_row=row, best_top1=best_top1, best_epoch=best_epoch)
                engine.save_checkpoint(run / 'last.pt', checkpoint)
                if improved:
                    engine.save_checkpoint(run / 'best.pt', checkpoint)
                row_append(run / 'results.csv', row)
                timings.append(row['epoch_seconds'])
                state = dict(status='completed' if epoch == limit else 'running', completed_epoch=epoch,
                             target_epochs=125, lr_horizon=200, config_sha256=config_sha,
                             rank0_pid=os.getpid(), row=row, best_top1=best_top1, best_epoch=best_epoch,
                             updated_at_cst=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                             remaining_seconds_estimate=(limit - epoch) * sum(timings[-5:]) / len(timings[-5:]))
                engine.atomic_json(run / 'state.json', state)
                print(json.dumps(state), flush=True)
            dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--audit', action='store_true')
    p.add_argument('--preview', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    c, config_sha = config_load(args.config)
    if args.check:
        if not args.preview or not args.out:
            p.error('--check needs --preview and --out')
        cpu_check(c, config_sha, args.preview, args.out)
    elif args.audit:
        audit(c, config_sha)
    else:
        if args.smoke and not args.out:
            p.error('--smoke needs --out')
        distributed_run(c, config_sha, args.smoke, args.out)


if __name__ == '__main__':
    main()
