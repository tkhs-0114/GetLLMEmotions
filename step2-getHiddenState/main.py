import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

step_dir = Path(__file__).parent
project_root = step_dir.parent
input_dir = project_root / "step1-getStory" / "output"
output_dir = step_dir / "output"
output_dir.mkdir(exist_ok=True)

# 保存したディレクトリを指定
save_directory = project_root / "my_local_llama3"

# モデルとトークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained(save_directory)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

emotions = ["Joy", "Anticipation", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger"]

def get_prompts(i=1, emotion="Joy"):
    path = input_dir / f"output-{i}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry["story"] for entry in data if entry["emotion"] == emotion]

def prompt_to_vector(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # output_hidden_states=True を指定することで、全レイヤーの出力を取得できる
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states

    return hidden_states

# hidden_states の長さから層数(Embedding含む)を取得する
initial_layer_vectors = prompt_to_vector(get_prompts()[0])
layer_length = len(initial_layer_vectors)
vector_length = initial_layer_vectors[0].shape[-1]
print(f"モデルの層数 (Embedding含む): {layer_length}")
print(f"ベクトルの次元数: {vector_length}")



tokens_to_skip = 50
skip_count = 0

print(f"start at: {datetime.datetime.now()}")

for emotion in emotions:
    ave_vector: list[torch.Tensor] = [
        torch.zeros(vector_length, device=device, dtype=torch.float32)
        for _ in range(layer_length)
    ]
    count = 0
    for i in range(1, 101):
        print(f"Processing emotion: {emotion}, file index: {i}")
        prompts = get_prompts(i, emotion)
        for prompt in prompts:

            token_length = len(tokenizer(prompt)["input_ids"])
            if token_length <= tokens_to_skip:
                skip_count += 1
                print(f"Prompt '{prompt}' has only {token_length} tokens, skipping...")
                continue

            count += 1

            layer_vectors = prompt_to_vector(prompt)
            
            for layer_index, vector in enumerate(layer_vectors):
                token_vectors = vector.squeeze(0)  # [seq_len, hidden_size]

                current_vector = token_vectors[tokens_to_skip:].mean(dim=0).to(torch.float32)
                
                # オンライン平均: mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
                ave_vector[layer_index] += (current_vector - ave_vector[layer_index]) / count

    save_path = output_dir / f"ave_vector_{emotion}.pt"
    torch.save([v.cpu() for v in ave_vector], save_path)
    print(f"Saved ave_vector for {emotion} to {save_path}")

print(f"Skipped {skip_count} prompts due to insufficient tokens.")
print(f"end at: {datetime.datetime.now()}")
