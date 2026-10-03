import torch

root = "/home/ubuntu/rz/yolo_study/yolo26_voc_research/runs/voc_yolo26n_baseline_4x3090_b64_e150_g4567_s42/weights"
checkpoints = {
    name: torch.load(f"{root}/{name}.pt", map_location="cpu", weights_only=False)
    for name in ("best", "last")
}
models = {name: checkpoint.get("ema") or checkpoint.get("model") for name, checkpoint in checkpoints.items()}

for name, checkpoint in checkpoints.items():
    model = models[name]
    results = checkpoint.get("train_results", {})
    print(
        name,
        "epoch_meta", checkpoint.get("epoch"),
        "date", checkpoint.get("date"),
        "rows", len(results.get("epoch", [])),
        "last_epoch", results.get("epoch", [None])[-1],
        "params", sum(parameter.numel() for parameter in model.parameters()),
        "dtype", next(model.parameters()).dtype,
    )

best_state = models["best"].state_dict()
last_state = models["last"].state_dict()
print("same_keys", best_state.keys() == last_state.keys())
different = []
max_diff = 0.0
for key in best_state:
    left, right = best_state[key], last_state[key]
    if left.shape != right.shape or not torch.equal(left, right):
        different.append(key)
        if left.shape == right.shape and left.is_floating_point():
            max_diff = max(max_diff, (left.float() - right.float()).abs().max().item())
print("different_tensor_count", len(different))
print("max_abs_diff", max_diff)
