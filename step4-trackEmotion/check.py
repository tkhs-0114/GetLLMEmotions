import argparse
import json
import time
import unicodedata
from pathlib import Path

output_dir = Path(__file__).parent / "output"

BAR_WIDTH = 40
INTERVAL_SEC = 1.0

EMOTION_JA = {
    "Joy": "喜び",
    "Anticipation": "期待",
    "Trust": "信頼",
    "Fear": "恐れ",
    "Surprise": "驚き",
    "Sadness": "悲しみ",
    "Disgust": "嫌悪",
    "Anger": "怒り",
}


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in text)


def pad_right(text: str, width: int) -> str:
    return text + " " * max(width - display_width(text), 0)


parser = argparse.ArgumentParser(description="step4のJSONファイルを読み込み、トークンごとの感情類似度を視覚的に表示する")
parser.add_argument("json_file", help="読み込むJSONファイルのパス (またはoutput内のファイル名)")
args = parser.parse_args()

path = Path(args.json_file)
if not path.exists():
    candidate = output_dir / path.name
    if candidate.exists():
        path = candidate
    elif (output_dir / f"{path.name}.json").exists():
        path = output_dir / f"{path.name}.json"
    else:
        available = sorted(p.name for p in output_dir.glob("*.json"))
        parser.error(f"{args.json_file} が見つかりません。指定可能なファイル: {', '.join(available)}")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

emotions: list[str] = data["emotions"]
tokens: list[dict] = data["tokens"]

labels = {
    emotion: f"{emotion}({EMOTION_JA[emotion]})" if emotion in EMOTION_JA else emotion
    for emotion in emotions
}
label_width = max(display_width(label) for label in labels.values())

max_similarity = max(
    (abs(score) for token_info in tokens for score in token_info["similarity"].values()),
    default=1.0,
)

accumulated_text = ""

try:
    for i, token_info in enumerate(tokens):
        accumulated_text += token_info["token"]
        similarity: dict[str, float] = token_info["similarity"]

        lines = [
            "\033[2J\033[H" + accumulated_text,
            "",
            "-" * 60,
            f"Token [{token_info['index'] + 1}/{len(tokens)}] ({token_info['role']}): {repr(token_info['token'])}",
            "-" * 60,
        ]

        for emotion in emotions:
            score = similarity[emotion]
            label = pad_right(labels[emotion], label_width)
            bar_length = round(score / max_similarity * BAR_WIDTH) if max_similarity > 0 else 0
            bar = "#" * max(bar_length, 0)
            lines.append(f"  {label} : {score:8.4f} | {bar}")

        print("\n".join(lines), flush=True)

        if i < len(tokens) - 1:
            time.sleep(INTERVAL_SEC)
except KeyboardInterrupt:
    print("\n中断しました。")
