import argparse
from pathlib import Path

import torch

output_dir = Path(__file__).parent / "output"

TOP_K = 10
BAR_WIDTH = 40

parser = argparse.ArgumentParser(description="感情ベクトルの各層で値が大きい次元を表示する")
parser.add_argument("emotion", help="感情名 (例: Joy, neutral)")
args = parser.parse_args()

path = output_dir / f"emotion_vector_{args.emotion}.pt"
if not path.exists():
    available = sorted(p.stem.removeprefix("emotion_vector_") for p in output_dir.glob("emotion_vector_*.pt"))
    parser.error(f"{path} が見つかりません。指定可能な感情: {', '.join(available)}")

emotion_vector: list[torch.Tensor] = torch.load(path, weights_only=True)
print(f"感情: {args.emotion}, 層数: {len(emotion_vector)}, 次元数: {emotion_vector[0].shape[-1]}")

for layer_index, vector in enumerate(emotion_vector):
    values, indices = torch.topk(vector.float(), TOP_K)
    max_value = values[0].item()

    print(f"\n[layer {layer_index}] mean={vector.mean().item():.4f} std={vector.std().item():.4f}")
    for value, index in zip(values.tolist(), indices.tolist()):
        bar_length = round(value / max_value * BAR_WIDTH) if max_value > 0 else 0
        print(f"  dim {index:5d}: {value:10.4f} | {'#' * max(bar_length, 0)}")
