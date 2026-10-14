from pathlib import Path

import torch

step_dir = Path(__file__).parent
input_dir = step_dir.parent / "step2-getHiddenState" / "output"
output_dir = step_dir / "output"
output_dir.mkdir(exist_ok=True)

files = sorted(input_dir.glob("ave_vector_*.pt"))

# 各感情の平均ベクトル（層ごとのリスト）を読み込む
ave_vectors = [torch.load(f, weights_only=True) for f in files]
layer_length = len(ave_vectors[0])

# ニュートラルベクトル = 全感情の平均ベクトルを層ごとに平均したもの
neutral_vector = [
    torch.stack([v[layer] for v in ave_vectors]).mean(dim=0)
    for layer in range(layer_length)
]
torch.save(neutral_vector, output_dir / "emotion_vector_neutral.pt")
print("ニュートラルベクトル先頭5要素 (最終層):", neutral_vector[-1][:5])

# 感情ベクトル = 各感情の平均ベクトル - ニュートラルベクトル
for file_path, ave_vector in zip(files, ave_vectors):
    emotion = file_path.stem.removeprefix("ave_vector_")
    emotion_vector = [ave_vector[layer] - neutral_vector[layer] for layer in range(layer_length)]
    save_path = output_dir / f"emotion_vector_{emotion}.pt"
    torch.save(emotion_vector, save_path)
    print(f"感情ベクトル ({emotion}) 先頭5要素 (最終層):", emotion_vector[-1][:5])
