from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 保存したディレクトリを指定
save_directory = Path(__file__).resolve().parent.parent / "my_local_llama3"

# モデルとトークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained(save_directory)
model.eval()

prompt = "Hello, how are you?"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
	output_ids = model.generate(
		**inputs,
		max_new_tokens=80,
		do_sample=True,
		temperature=0.7,
		top_p=0.9,
		eos_token_id=tokenizer.eos_token_id,
		pad_token_id=tokenizer.eos_token_id,
	)

result = tokenizer.decode(output_ids[0], skip_special_tokens=True)



print(f"Prompt: '{prompt}'")
print(f"result: '{result}'")