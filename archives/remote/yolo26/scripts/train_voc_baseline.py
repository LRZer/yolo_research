"""从随机权重训练原版 YOLO26n + PASCAL VOC 正式基线。

这个文件本身不决定“用什么模型、数据或参数”；这些实验条件全部记录在
``configs/voc_yolo26n_baseline_4x3090_b64_g4567.yaml``。脚本的职责只有两件事：

1. 在训练前检查环境、源码、模型、数据和输出目录，防止错误实验跑十几个小时；
2. 把 YAML 中已经锁定的参数原样交给 Ultralytics 的 ``model.train()``。

特别注意：模型始终由 ``.yaml`` 结构文件创建，且 ``pretrained=False`` 被强制检查，
因此不会把任何 ``.pt`` 预训练权重当作初始模型。

示例：
    # 只预检：检查模型、数据、源码版本、GPU 和输出目录；不会训练。
    /home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python scripts/train_voc_baseline.py --dry-run

    # 预检通过并确认后，才执行这一条来启动 150 epoch 正式训练。
    /home/ubuntu/rz/yolo_study/.conda/yolo26/bin/python scripts/train_voc_baseline.py
"""

# 允许在类型注解中直接使用 Path、dict[str, ...] 等现代 Python 写法。
from __future__ import annotations

# argparse：读取 --dry-run 这类命令行开关；hashlib：计算文件 SHA-256；
# subprocess：只读地查询 Git 提交号和工作区状态。
import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

# torch：检查 CUDA/BF16；ultralytics：检查实际导入的版本和位置；
# yaml：读取实验配置；YOLO：构建模型、调用训练。
import torch
import ultralytics
import yaml
from ultralytics import YOLO


# 当前文件在 ``项目根目录/scripts/`` 中，因此 parents[1] 就是 yolo26_voc_research。
# 这样无论用户从哪个终端目录运行脚本，都不会找错配置文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "voc_yolo26n_baseline_4x3090_b64_g4567.yaml"


def parse_args() -> argparse.Namespace:
    """读取安全的命令行选项；核心实验参数仍必须留在 YAML，避免被临时命令改掉。"""
    parser = argparse.ArgumentParser(description="Run the formal from-scratch YOLO26n/VOC baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="基线实验 YAML 的路径（默认是 configs/voc_yolo26n_baseline.yaml）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="执行全部检查并构建随机模型，但绝不训练，也不创建结果目录。",
    )
    return parser.parse_args()


def load_yaml(path: Path, label: str) -> dict[str, Any]:
    """读取一个 YAML 字典；文件缺失或格式不对时，给出明确错误而不是让后面难以定位。"""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    # safe_load 只读取普通 YAML 数据，不执行 YAML 内可能嵌入的 Python 对象。
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a YAML mapping: {path}")
    return data


def require(mapping: dict[str, Any], key: str, section: str) -> Any:
    """取出必填字段；缺失时指出准确的 ``配置段.字段``，例如 ``training.epochs``。"""
    if key not in mapping:
        raise KeyError(f"Missing required setting '{section}.{key}' in the experiment config.")
    return mapping[key]


def git_output(source_root: Path, *arguments: str) -> str:
    """在锁定的 Ultralytics 源码目录执行只读 Git 命令，例如查询当前提交号。"""
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def validate_positive_int(value: Any, name: str) -> int:
    """确保 epoch、图像尺寸、batch 等设置是大于 0 的整数。"""
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def validate_experiment(config_path: Path) -> dict[str, Any]:
    """检查“这确实是一次可复现、从零训练的公平基线”，并返回已解析的设置。"""
    # 先取出 YAML 的六个顶级部分；后续所有条件都只从它们读取。
    spec = load_yaml(config_path, "Experiment configuration")
    model_spec = require(spec, "model", "root")
    data_spec = require(spec, "dataset", "root")
    train_spec = require(spec, "training", "root")
    output_spec = require(spec, "output", "root")
    provenance = require(spec, "provenance", "root")
    experiment = require(spec, "experiment", "root")
    if not all(isinstance(item, dict) for item in (model_spec, data_spec, train_spec, output_spec, provenance, experiment)):
        raise ValueError("experiment, provenance, model, dataset, training and output must all be YAML sections.")

    # 模型路径相对于研究项目根目录解析，而不是相对于“当前终端所在目录”解析。
    model_path = PROJECT_ROOT / require(model_spec, "yaml", "model")
    model_path = model_path.resolve()
    model_yaml = load_yaml(model_path, "Model YAML")
    model_scale = require(model_spec, "scale", "model")
    # 双重确认是 n：实验记录写一次，模型 YAML 也写一次；两者不同就立即停止。
    if model_yaml.get("scale") != model_scale:
        raise ValueError(
            f"Model scale mismatch: experiment requests {model_scale!r}, model YAML contains {model_yaml.get('scale')!r}."
        )
    # 结构必须来自 YAML；若误填 .pt，就可能违反“从零训练”的实验规则。
    if model_path.suffix.lower() != ".yaml":
        raise ValueError(f"The baseline model must be a .yaml architecture file, not {model_path.name!r}.")
    if model_yaml.get("nc") != require(model_spec, "upstream_default_nc", "model"):
        raise ValueError("The baseline model YAML no longer has the pinned upstream default class count.")
    if require(model_spec, "pretrained", "model") is not False:
        raise ValueError("Baseline safety check failed: model.pretrained must be false for from-scratch training.")
    if require(model_spec, "task", "model") != "detect":
        raise ValueError("This script is for object detection, so model.task must be 'detect'.")

    # data_yaml 说明训练/验证划分与类别名；data_root 是图片和标签实际所在的位置。
    data_yaml = Path(require(data_spec, "data_yaml", "dataset")).resolve()
    data_root = Path(require(data_spec, "root", "dataset")).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"VOC data YAML not found: {data_yaml}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"VOC dataset root not found: {data_root}")
    # 逐项检查五个 VOC 划分的 images 与 labels 文件夹，避免训练到一半才发现数据不完整。
    for relative_dir in (
        "images/train2007",
        "images/val2007",
        "images/train2012",
        "images/val2012",
        "images/test2007",
        "labels/train2007",
        "labels/val2007",
        "labels/train2012",
        "labels/val2012",
        "labels/test2007",
    ):
        if not (data_root / relative_dir).is_dir():
            raise FileNotFoundError(f"Expected VOC folder is missing: {data_root / relative_dir}")
    data_yaml_contents = load_yaml(data_yaml, "VOC data YAML")
    names = require(data_yaml_contents, "names", "VOC data YAML")
    if len(names) != int(require(data_spec, "classes", "dataset")):
        raise ValueError("The class count in the experiment config does not match VOC.yaml.")

    # VOC.yaml 位于源码中的 .../ultralytics/cfg/datasets/VOC.yaml；向上三级就是源码根目录。
    source_root = Path(ultralytics.__file__).resolve().parent.parent
    expected_commit = require(provenance, "ultralytics_git_commit", "provenance")
    actual_commit = git_output(source_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            "Ultralytics source commit does not match the pinned baseline. "
            f"Expected {expected_commit}, found {actual_commit}."
        )
    # 基线要求官方源码没有“已跟踪但未提交”的改动，否则无法证明使用的是原版模型。
    if git_output(source_root, "status", "--porcelain"):
        raise RuntimeError("Ultralytics source has tracked uncommitted changes; restore a clean source before the baseline run.")
    # 再用 SHA-256 核验官方原始模型图纸，防止同一提交下文件被本地意外改写。
    # Git 在 Windows 工作区可能检出 CRLF，而 Linux 通常检出 LF；它们代表同一份 YAML，
    # 因此先统一为 LF 再计算哈希。这样校验的是模型文字内容，而非操作系统换行符。
    upstream_model = source_root / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"
    normalized_model_bytes = upstream_model.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    actual_model_hash = hashlib.sha256(normalized_model_bytes).hexdigest().upper()
    if actual_model_hash != require(provenance, "upstream_model_sha256", "provenance"):
        raise RuntimeError("The official yolo26.yaml source file does not match the pinned baseline checksum.")
    if ultralytics.__version__ != require(provenance, "ultralytics_version", "provenance"):
        raise RuntimeError(
            "Imported Ultralytics version does not match the pinned baseline: "
            f"expected {provenance['ultralytics_version']}, found {ultralytics.__version__}."
        )
    if torch.__version__ != require(provenance, "torch_version", "provenance"):
        raise RuntimeError(
            "PyTorch version does not match the pinned baseline: "
            f"expected {provenance['torch_version']}, found {torch.__version__}."
        )
    # 防止环境优先导入了别处 pip 安装的 ultralytics，而不是我们锁定的源码目录。
    installed_package = Path(ultralytics.__file__).resolve()
    if source_root not in installed_package.parents:
        raise RuntimeError(f"Ultralytics is not imported from the pinned source checkout: {installed_package}")

    # 正式实验只允许使用配置规定的 GPU；CUDA、显卡编号和 BF16 能力都在这里检查。
    device = str(require(train_spec, "device", "training"))
    if device == "cpu":
        raise RuntimeError("The formal baseline is configured for CUDA GPUs 4,5,6,7, not CPU training.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable, so the configured GPU baseline cannot start.")
    device_tokens = [token.strip() for token in device.split(",")]
    if not device_tokens or any(not token.isdecimal() for token in device_tokens):
        raise ValueError("training.device must be one GPU index such as '0', or comma-separated indices such as '0,1,2,3'.")
    device_indices = tuple(int(token) for token in device_tokens)
    if len(set(device_indices)) != len(device_indices):
        raise ValueError(f"training.device must not repeat a GPU index, got {device!r}.")
    if any(index >= torch.cuda.device_count() for index in device_indices):
        raise RuntimeError(f"Configured GPU(s) {device!r} exceed the {torch.cuda.device_count()} visible GPU(s).")
    amp = require(train_spec, "amp", "training")
    if amp == "bf16":
        original_device_index = torch.cuda.current_device()
        try:
            for device_index in device_indices:
                torch.cuda.set_device(device_index)
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise RuntimeError(
                        f"CUDA:{device_index} does not provide native BF16 AMP required by the baseline config."
                    )
        finally:
            torch.cuda.set_device(original_device_index)

    # 检查数值参数，并强制 fraction=1.0，杜绝把小规模 smoke test 当成正式结果。
    epochs = validate_positive_int(require(train_spec, "epochs", "training"), "training.epochs")
    imgsz = validate_positive_int(require(train_spec, "imgsz", "training"), "training.imgsz")
    batch = validate_positive_int(require(train_spec, "batch", "training"), "training.batch")
    if batch % len(device_indices) != 0:
        raise ValueError(
            f"training.batch={batch} must be divisible by the {len(device_indices)} configured GPU(s) ({device})."
        )
    # workers=0 表示主进程自己加载数据，在 Windows 基线中是合法值；只有负数才非法。
    workers = int(require(train_spec, "workers", "training"))
    if workers < 0:
        raise ValueError(f"training.workers must be zero or a positive integer, got {workers}.")
    fraction = float(require(train_spec, "fraction", "training"))
    if not 0 < fraction <= 1:
        raise ValueError(f"training.fraction must be in (0, 1], got {fraction}.")
    if fraction != 1.0:
        raise ValueError("A formal baseline must use the entire training set, so training.fraction must be 1.0.")

    # exist_ok=false 时，正式同名目录一旦存在就拒绝运行，保护旧的实验结果。
    project_dir = Path(require(output_spec, "project", "output")).resolve()
    run_name = str(require(output_spec, "name", "output"))
    run_dir = project_dir / run_name
    if run_dir.exists() and not bool(require(output_spec, "exist_ok", "output")):
        raise FileExistsError(f"Refusing to overwrite existing experiment results: {run_dir}")

    # 返回已经验证且路径已解析的内容，后面的 main() 不再重新猜测路径或参数。
    return {
        "spec": spec,
        "model_path": model_path,
        "data_yaml": data_yaml,
        "data_root": data_root,
        "source_root": source_root,
        "device": device,
        "device_indices": device_indices,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "fraction": fraction,
        "project_dir": project_dir,
        "run_name": run_name,
        "run_dir": run_dir,
    }


def print_preflight(settings: dict[str, Any], parameter_count: int) -> None:
    """在耗时训练前，把本次实验的关键条件完整打印到终端，供人工最后确认。"""
    spec = settings["spec"]
    print("\n=== YOLO26n / VOC formal baseline preflight ===")
    print(f"Experiment: {spec['experiment']['id']}")
    print(f"Source:     {settings['source_root']} @ {spec['provenance']['ultralytics_git_commit']}")
    print(f"Model:      {settings['model_path']}")
    print(f"Parameters: {parameter_count:,} before VOC changes the Detect head from 80 to 20 classes")
    print(f"Dataset:    {settings['data_root']} ({spec['dataset']['train_images']} train / {spec['dataset']['val_images']} val)")
    gpu_summary = ", ".join(
        f"CUDA:{index} {torch.cuda.get_device_name(index)}" for index in settings["device_indices"]
    )
    print(f"GPU:        {gpu_summary}")
    print(
        f"Training:   {settings['epochs']} epochs, imgsz={settings['imgsz']}, batch={settings['batch']}, "
        f"workers={settings['workers']}, AMP={spec['training']['amp']}, seed={spec['training']['seed']}"
    )
    print(f"Output:     {settings['run_dir']}")
    print("Initialization: random YAML model; pretrained=False; no .pt input weight")


def main() -> None:
    """主流程：预检 → 从 YAML 构建随机模型 → dry-run 停止或正式启动训练。"""
    args = parse_args()
    settings = validate_experiment(args.config)
    spec = settings["spec"]

    # 只从 .yaml 构建网络：这一步没有加载 .pt，也没有训练；它用于验证网络图纸能正确实例化。
    model = YOLO(str(settings["model_path"]), task=spec["model"]["task"])
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    print_preflight(settings, parameter_count)

    # --dry-run 是安全出口：到这里说明所有检查均通过，但不会调用 model.train()。
    if args.dry_run:
        print("\nDry run passed: no training was started and no run directory was created.")
        return

    print("\nStarting formal baseline training...")
    # 只有没有 --dry-run 时才会进入这里；下面每个参数都直接对应 baseline YAML 的记录。
    model.train(
        data=str(settings["data_yaml"]),  # VOC 的 train/val 划分和 20 个类别名称。
        epochs=settings["epochs"],  # 完整遍历训练集的次数，本实验为 150。
        imgsz=settings["imgsz"],  # 网络实际接收的方形图像尺寸，本实验为 640。
        batch=settings["batch"],  # 全局 batch=64；4 卡 DDP 下每卡 16 张。
        device=settings["device"],  # 仅使用 CUDA:4,5,6,7，绝不调用前四张卡。
        workers=settings["workers"],  # 每个 DDP 进程使用 4 个数据加载子进程。
        amp=spec["training"]["amp"],  # 使用已验证的 BF16 自动混合精度。
        optimizer=spec["training"]["optimizer"],  # auto：由当前锁定版本的 Ultralytics 选择优化器。
        fraction=settings["fraction"],  # 1.0：必须使用全部 16,551 张训练图。
        seed=int(spec["training"]["seed"]),  # 固定随机初始化、数据打乱和增强的随机性。
        deterministic=bool(spec["training"]["deterministic"]),  # 尽量提高可复现性。
        val=bool(spec["training"]["val"]),  # 每个 epoch 都在 4,952 张验证图上计算 mAP。
        patience=int(spec["training"]["patience"]),  # 与总轮数同为 150，避免正式实验被提前截断。
        pretrained=False,  # 再次硬编码为 False，作为“从零训练”的最后一道保护。
        project=str(settings["project_dir"]),  # runs 的父目录。
        name=settings["run_name"],  # 当前实验的子目录名。
        save=bool(spec["output"]["save"]),  # 保存 best.pt、last.pt 和周期检查点。
        save_period=int(spec["output"]["save_period"]),  # 每 10 epoch 额外保存一次。
        plots=bool(spec["output"]["plots"]),  # 输出曲线、混淆矩阵、训练和预测示例图。
        exist_ok=bool(spec["output"]["exist_ok"]),  # false：禁止覆盖已有同名实验。
    )


if __name__ == "__main__":
    main()
