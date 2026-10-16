import argparse
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_LAYER = 21
MAX_NEW_TOKENS = 200

emotions = ["Joy", "Anticipation", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger"]

step_dir = Path(__file__).parent
project_root = step_dir.parent
model_dir = project_root / "my_local_llama3"
vector_dir = project_root / "step3-getEmovitonVec" / "output"
output_dir = step_dir / "output"

parser = argparse.ArgumentParser(description="プロンプトと生成結果のトークンごとに感情ベクトルとのコサイン類似度を計算する")
parser.add_argument("prompt", help="LLMに入力するプロンプト（最後のHumanの発話）")
parser.add_argument(
    "--history",
    type=Path,
    help='過去の会話のJSONファイル。[{"human": "...", "assistant": "..."}, ...] の形式',
)
args = parser.parse_args()

history = []
if args.history:
    with args.history.open(encoding="utf-8") as f:
        history = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

emotion_matrix = torch.stack([
    torch.load(vector_dir / f"emotion_vector_{emotion}.pt", weights_only=True)[TARGET_LAYER]
    for emotion in emotions
]).to(device=device, dtype=torch.float32)  # [感情数, hidden_size]

formatted_prompt = "".join(
    f"Human: {turn['human']}\n\nAssistant: {turn['assistant']}\n\n" for turn in history
)
formatted_prompt += f"Human: {args.prompt}\n\nAssistant:"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
input_length = inputs["input_ids"].shape[1]

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        # ベースモデルは次の "Human:" ターンまで書き続けるため、そこで止める
        stop_strings=["\nHuman:"],
        tokenizer=tokenizer,
    )
    full_ids = generated[0]

    outputs = model(full_ids.unsqueeze(0), output_hidden_states=True)

hidden = outputs.hidden_states[TARGET_LAYER][0].to(torch.float32)  # [seq_len, hidden_size]
similarity = F.normalize(hidden, dim=-1) @ F.normalize(emotion_matrix, dim=-1).T  # [seq_len, 感情数]

tokens = []
for index, (token_id, scores) in enumerate(zip(full_ids.tolist(), similarity.tolist())):
    tokens.append({
        "index": index,
        "token": tokenizer.decode([token_id]),
        "role": "input" if index < input_length else "output",
        "similarity": dict(zip(emotions, scores)),
    })

result = {
    "history": history,
    "prompt": args.prompt,
    "response": tokenizer.decode(full_ids[input_length:], skip_special_tokens=True),
    "layer": TARGET_LAYER,
    "emotions": emotions,
    "tokens": tokens,
}

output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Response: {result['response']}")
print(f"Saved to {output_path}")
