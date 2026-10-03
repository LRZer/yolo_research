"""Single-GPU CUB ResNet50 scratch A/original baseline, with resumable epochs.

Official train images train the model. Official test images are evaluated at every
epoch as the user requested; they are therefore a monitored test, not a blind
final test. No bounding box, mask, resize, or pretrained weight is used here.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50


CSV_FIELDS = (
    "epoch", "lr", "train_loss", "train_top1", "train_top5", "test_loss",
    "test_top1", "test_top5", "train_seconds", "test_seconds", "epoch_seconds",
    "gpu_max_memory_gib", "train_images", "test_images", "optimizer_steps",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_table(path: Path) -> dict[int, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        identifier, value = line.split(maxsplit=1)
        key = int(identifier)
        if key in rows:
            raise ValueError(f"Duplicate ID in {path}: {key}")
        rows[key] = value
    return rows


def load_metadata(data_root: Path, stats: dict) -> tuple[list[tuple[int, Path, int]], list[tuple[int, Path, int]]]:
    paths = parse_table(data_root / "images.txt")
    labels = parse_table(data_root / "image_class_labels.txt")
    splits = parse_table(data_root / "train_test_split.txt")
    if len(paths) != 11788 or set(paths) != set(labels) or set(paths) != set(splits):
        raise ValueError("CUB metadata tables do not match the audited 11,788 IDs")
    for name, expected in stats["metadata_sha256"].items():
        actual = sha256(data_root / name)
        if actual != expected:
            raise ValueError(f"Metadata hash mismatch for {name}: {actual} != {expected}")
    train, test = [], []
    for identifier in sorted(paths):
        label = int(labels[identifier]) - 1
        if not 0 <= label < 200:
            raise ValueError(f"Invalid class {label + 1} for image {identifier}")
        path = data_root / "images" / paths[identifier]
        if not path.is_file():
            raise FileNotFoundError(path)
        row = (identifier, path, label)
        if splits[identifier] == "1":
            train.append(row)
        elif splits[identifier] == "0":
            test.append(row)
        else:
            raise ValueError(f"Invalid split for image {identifier}")
    if len(train) != 5994 or len(test) != 5794 or len({r[2] for r in train}) != 200:
        raise ValueError("Unexpected official train/test or class counts")
    return train, test


class CUBOriginal(Dataset):
    def __init__(self, records: list[tuple[int, Path, int]], mean: list[float], std: list[float],
                 fill: tuple[int, int, int], train: bool, seed: int, epoch: int = 0):
        self.records = records
        self.mean = mean
        self.std = std
        self.fill = fill
        self.train = train
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        identifier, path, label = self.records[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        if width > 500 or height > 500:
            raise ValueError(f"Image {identifier} exceeds 500x500")
        if self.train:
            rng = random.Random(self.seed * 1_000_003 + self.epoch * 100_003 + identifier)
            if rng.random() < 0.5:
                image = TF.hflip(image)
            image = TF.adjust_brightness(image, rng.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, rng.uniform(0.9, 1.1))
            image = TF.adjust_saturation(image, rng.uniform(0.9, 1.1))
        canvas = Image.new("RGB", (500, 500), self.fill)
        canvas.paste(image, ((500 - width) // 2, (500 - height) // 2))
        tensor = TF.normalize(TF.to_tensor(canvas), mean=self.mean, std=self.std)
        return tensor, label


def topk_counts(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    top5 = logits.topk(5, dim=1).indices
    matches = top5.eq(target[:, None])
    return matches[:, 0].sum(), matches.any(dim=1).sum()


def learning_rate(epoch_number: int, config: dict) -> float:
    warmup = config["warmup_epochs"]
    total = config["epochs"]
    base = config["base_lr"]
    floor = config["min_lr"]
    if epoch_number <= warmup:
        return base * epoch_number / warmup
    progress = (epoch_number - warmup) / (total - warmup)
    return floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def aggregate(metric: torch.Tensor) -> tuple[float, float, float, int]:
    values = metric.detach().cpu().tolist()
    total = int(values[3])
    return values[0] / total, 100.0 * values[1] / total, 100.0 * values[2] / total, total


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler, device: torch.device,
                    effective_batch: int, microbatch: int, image_count: int) -> tuple[float, float, float, int, int]:
    model.train()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    group_images = 0
    group_target = min(effective_batch, image_count)
    processed = 0
    updates = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        amount = targets.numel()
        if amount > microbatch or group_images + amount > group_target:
            raise AssertionError("Microbatch exceeds the effective-batch boundary")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(images)
            per_image = F.cross_entropy(logits.float(), targets, reduction="none")
            loss = per_image.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(per_image.sum() / group_target).backward()
        correct1, correct5 = topk_counts(logits.detach(), targets)
        totals += torch.stack((per_image.detach().sum(), correct1, correct5,
                               torch.tensor(amount, device=device))).to(torch.float64)
        group_images += amount
        processed += amount
        if group_images == group_target:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            group_images = 0
            group_target = min(effective_batch, image_count - processed)
    if processed != image_count or group_images != 0 or updates != math.ceil(image_count / effective_batch):
        raise AssertionError(f"Training saw {processed} images and made {updates} updates")
    loss, top1, top5, counted = aggregate(totals)
    return loss, top1, top5, counted, updates


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float, int]:
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(images)
        per_image = F.cross_entropy(logits.float(), targets, reduction="none")
        correct1, correct5 = topk_counts(logits, targets)
        totals += torch.stack((per_image.sum(), correct1, correct5,
                               torch.tensor(targets.numel(), device=device))).to(torch.float64)
    return aggregate(totals)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("results.csv header does not match this trainer")
        rows = list(reader)
    if [int(row["epoch"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("results.csv epochs are not continuous from 1")
    return rows


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    expected = {
        "group": "A_original", "architecture": "torchvision_resnet50_scratch",
        "epochs": 200, "seed": 2005, "canvas": 500,
        "effective_batch": 64, "microbatch": 16, "accumulation": 4, "workers": 4,
        "device_physical": 0, "numa_node": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected {key}: {config.get(key)!r}, expected {value!r}")
    if config["microbatch"] * config["accumulation"] != config["effective_batch"]:
        raise ValueError("Invalid batch accumulation configuration")
    return config, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="CPU-only data/config check; no run directory created")
    parser.add_argument("--train", action="store_true", help="Run training on one visible CUDA GPU")
    parser.add_argument("--smoke", action="store_true", help="One small train update on GPU0; no run files")
    args = parser.parse_args()
    if sum((args.check, args.train, args.smoke)) != 1:
        parser.error("Choose exactly one of --check, --smoke, or --train")
    config, config_sha = load_config(args.config)
    data_root = Path(config["data_root"])
    stats_path = Path(config["stats_json"])
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats["official_train_images"] != 5994 or stats["official_test_images"] != 5794:
        raise ValueError("Training statistics do not describe the official split")
    train_rows, test_rows = load_metadata(data_root, stats)
    if args.check:
        print(json.dumps({"status": "checked", "train": len(train_rows), "test": len(test_rows),
                          "classes": 200, "config_sha256": config_sha,
                          "stats_sha256": sha256(stats_path)}, indent=2), flush=True)
        return

    physical_device = str(config["device_physical"])
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Exactly one CUDA-visible GPU is required (physical GPU{physical_device})")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical_device:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {physical_device}")
    device = torch.device("cuda:0")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    mean = stats["mean_rgb_0_1"]
    std = stats["std_rgb_0_1"]
    fill = tuple(int(np.rint(value)) for value in stats["mean_rgb_0_255"])
    test_dataset = CUBOriginal(test_rows, mean, std, fill, False, config["seed"])
    test_loader = DataLoader(test_dataset, batch_size=config["eval_batch"], shuffle=False,
                             num_workers=config["workers"], pin_memory=True, persistent_workers=True)
    model = resnet50(weights=None, num_classes=200).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["base_lr"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    if args.smoke:
        smoke_rows = train_rows[:32]
        smoke_dataset = CUBOriginal(smoke_rows, mean, std, fill, True, config["seed"], 1)
        smoke_loader = DataLoader(smoke_dataset, batch_size=config["microbatch"], shuffle=False,
                                  num_workers=config["workers"], pin_memory=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        loss, top1, top5, images, updates = train_one_epoch(
            model, smoke_loader, optimizer, scaler, device,
            config["effective_batch"], config["microbatch"], len(smoke_rows)
        )
        torch.cuda.synchronize(device)
        print(json.dumps({"status": "smoke_passed", "images": images, "updates": updates,
                          "loss": loss, "top1": top1, "top5": top5,
                          "seconds": time.monotonic() - started,
                          "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3}),
              flush=True)
        return
    run = Path(config["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    last_path = run / "last.pt"
    csv_path = run / "results.csv"
    state_path = run / "state.json"
    stats_sha = sha256(stats_path)
    start_epoch = 1
    prior_rows = read_csv(csv_path)
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["config_sha256"] != config_sha or checkpoint["stats_sha256"] != stats_sha:
            raise ValueError("Checkpoint config/stats hash mismatch; refusing resume")
        completed = int(checkpoint["epoch"])
        if completed < 1 or completed > config["epochs"]:
            raise ValueError("Checkpoint epoch out of bounds")
        if len(prior_rows) == completed - 1:
            append_csv(csv_path, checkpoint["last_row"])
            prior_rows = read_csv(csv_path)
        if len(prior_rows) != completed or int(prior_rows[-1]["epoch"]) != completed:
            raise ValueError("Checkpoint and results.csv epoch mismatch; refusing resume")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = completed + 1
        print(f"RESUME verified from epoch {completed}", flush=True)
    elif prior_rows:
        raise ValueError("results.csv exists without a valid last.pt; refusing restart")

    if start_epoch > config["epochs"]:
        print("Training is already complete; no new optimizer steps", flush=True)
        return
    begin = time.time()
    durations = [float(row["epoch_seconds"]) for row in prior_rows]
    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_clock = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        lr = learning_rate(epoch, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_dataset = CUBOriginal(train_rows, mean, std, fill, True, config["seed"], epoch)
        train_generator = torch.Generator().manual_seed(config["seed"] + epoch * 100_003)
        train_loader = DataLoader(train_dataset, batch_size=config["microbatch"], shuffle=True,
                                  generator=train_generator, num_workers=config["workers"],
                                  pin_memory=True, persistent_workers=False)
        torch.cuda.synchronize(device)
        train_clock = time.monotonic()
        train_loss, train_top1, train_top5, train_images, optimizer_steps = train_one_epoch(
            model, train_loader, optimizer, scaler, device, config["effective_batch"],
            config["microbatch"], len(train_rows)
        )
        del train_loader
        torch.cuda.synchronize(device)
        train_seconds = time.monotonic() - train_clock
        test_clock = time.monotonic()
        test_loss, test_top1, test_top5, test_images = evaluate(model, test_loader, device)
        torch.cuda.synchronize(device)
        test_seconds = time.monotonic() - test_clock
        if train_images != 5994 or test_images != 5794:
            raise AssertionError("Actual train/test sample counts differ from official split")
        epoch_seconds = time.monotonic() - epoch_clock
        row = {
            "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "train_top1": train_top1, "train_top5": train_top5,
            "test_loss": test_loss, "test_top1": test_top1, "test_top5": test_top5,
            "train_seconds": train_seconds, "test_seconds": test_seconds,
            "epoch_seconds": epoch_seconds,
            "gpu_max_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "train_images": train_images, "test_images": test_images,
            "optimizer_steps": optimizer_steps,
        }
        save_checkpoint(last_path, {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config_sha256": config_sha,
            "stats_sha256": stats_sha, "last_row": row,
        })
        append_csv(csv_path, row)
        durations.append(epoch_seconds)
        eta_seconds = statistics.median(durations[-5:]) * (config["epochs"] - epoch)
        finish = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=eta_seconds)
        state = {
            "status": "running" if epoch < config["epochs"] else "training_complete_pending_audit",
            "epoch": epoch, "total_epochs": config["epochs"], "seed": config["seed"],
            "physical_gpu": config["device_physical"], "numa_node": config["numa_node"],
            "effective_batch": 64, "microbatch": 16, "accumulation": 4, "workers": 4,
            "last_epoch_seconds": epoch_seconds,
            "last_train_seconds": train_seconds, "last_test_seconds": test_seconds,
            "median_last_five_epoch_seconds": statistics.median(durations[-5:]),
            "estimated_finish_utc": finish.isoformat(),
            "last_test_top1": test_top1, "last_test_top5": test_top5,
            "config_sha256": config_sha, "stats_sha256": stats_sha,
            "started_process_utc": dt.datetime.fromtimestamp(begin, dt.timezone.utc).isoformat(),
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_json(state_path, state)
        print(f"EPOCH {epoch:03d}/200 train={train_seconds:.1f}s test={test_seconds:.1f}s "
              f"total={epoch_seconds:.1f}s train_top1={train_top1:.3f} "
              f"test_top1={test_top1:.3f} test_top5={test_top5:.3f} "
              f"eta_utc={finish.isoformat()}", flush=True)
    print("TRAINING_200_EPOCHS_FINISHED_PENDING_STRICT_AUDIT", flush=True)


if __name__ == "__main__":
    main()
