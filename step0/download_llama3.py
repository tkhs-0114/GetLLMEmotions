import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_and_save_model(
	model_name: str = "meta-llama/Llama-3.1-8B",
	save_directory: str | Path = PROJECT_ROOT / "my_local_llama3",
) -> None:
	hf_token = os.environ.get("TOKEN")
	if not hf_token:
		raise RuntimeError(
			"環境変数 TOKEN が未設定です。Hugging Face のアクセストークンを TOKEN に設定してください。"
		)

	# セッションにログインして、gated model の取得時に認証が通るようにする
	login(token=hf_token, add_to_git_credential=False)

	save_path = Path(save_directory)
	save_path.mkdir(parents=True, exist_ok=True)

	print(f"Downloading tokenizer: {model_name}")
	tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

	print(f"Downloading model: {model_name}")
	model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

	print(f"Saving to local directory: {save_path.resolve()}")
	tokenizer.save_pretrained(save_path)
	model.save_pretrained(save_path)

	print("Done. Loading back from local directory to verify...")
	AutoTokenizer.from_pretrained(save_path)
	AutoModelForCausalLM.from_pretrained(save_path)
	print("Local load check passed.")


if __name__ == "__main__":
	try:
		download_and_save_model()
	except Exception as exc:
		print(f"Error: {exc}")
		print(
			"補足: モデルが gated の場合、Hugging Face 上で対象モデルのライセンス同意が必要です。"
		)
		raise