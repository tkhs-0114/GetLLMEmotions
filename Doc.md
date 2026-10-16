# このプロジェクトの最終目標

このプロジェクトはLLMの感情についての(Anthropicの論文)[https://www.anthropic.com/research/emotion-concepts-function]をローカル環境で再現し、ユーザーの入力と生成時にLLMの感情変化をリアルタイムでCUI上に表示することを目的としています。

## プロジェクトの詳細

対象のローカルLLMはLlama-3.1-8B (`meta-llama/Llama-3.1-8B`) です。
実行環境にはNvidiaのGPUが用意されています。この開発用のローカル端末上では、LLMは実行しません。AIエージェントがプログラムを実行する必要がある場合はユーザーに依頼してください。

対象とする感情は以下の8種類です。

`Joy`, `Anticipation`, `Trust`, `Fear`, `Surprise`, `Sadness`, `Disgust`, `Anger`

各スクリプトのパスはスクリプト自身の位置を基準に解決されるため、どのディレクトリから実行してもよい。

## ディレクトリ構成

```
my_local_llama3/        step0でダウンロードしたモデル
step0/                  モデルの準備
step1-getStory/output/  step1の出力 (ストーリー)
step2-getHiddenState/output/  step2の出力 (感情ごとの平均hidden state)
step3-getEmovitonVec/output/  step3の出力 (感情ベクトル)
step4-trackEmotion/output/    step4の出力 (トークンごとの感情の類似度)
old-project/            移植元の旧プロジェクト
```

## プロジェクトのステップ

### step0

Hugging FaceからLlama-3.1-8Bをダウンロードし、`my_local_llama3/` に保存する。

- `step0/download_llama3.py`: モデルとトークナイザーをダウンロードして保存する。環境変数 `TOKEN` にHugging Faceのアクセストークンを設定して実行する。
- `step0/llm-test.py`: 保存したモデルで簡単なテキスト生成を行い、動作を確認する。

出力: `my_local_llama3/`

### step1

Ollama上のLLM (`llama3.1:8b`、OpenAI互換API `/v1/chat/completions`) を使い、各感情に則った短いストーリーを生成させる。
ストーリーを生成するための話題(トピック)を100個用意する。
各感情・各トピックについて、ストーリーを5個ずつ生成させる (8感情 × 100トピック × 5 = 4000ストーリー)。
プロンプトでは感情名やその同義語を使わず、行動・身体感覚・台詞などで感情を表現するよう指示する。

- スクリプト: `step1-getStory/main.py`
- 入力: 感情の一覧, ストーリーのトピック (スクリプト内に定義)
- 出力: `step1-getStory/output/output-{トピック番号}.json` (トピック番号は1〜100)

各ファイルはJSON配列で、1ファイルあたり 8感情 × 5ストーリー = 40件の要素を持つ。

```json
[
  {
    "emotion": "Joy",
    "topic": "An artist discovers someone has tattooed their work",
    "story_num": 1,
    "story": "..."
  }
]
```

既存ファイルには追記されるため、再実行する場合は先に出力を削除すること。

### step2

感情が反映されたストーリーをローカルLLM (`my_local_llama3/`) に入力し、各層のhidden stateを取得する。
1ストーリーごとに、先頭50トークンを除いたトークンのhidden stateを平均し、それを感情ごとに全ストーリーで平均する。
トークン数が50以下のストーリーはスキップする。

- スクリプト: `step2-getHiddenState/main.py`
- 入力: `step1-getStory/output/output-{1..100}.json`
- 出力: `step2-getHiddenState/output/ave_vector_{感情}.pt`

各ファイルは層ごとの平均ベクトル (`torch.Tensor`, float32, 形状 `[hidden_size]`) のリストで、長さは層数 + 1 (Embedding層を含む)。

### step3

各感情の平均hidden stateから感情ベクトルを求める。

1. 全感情の平均ベクトルを層ごとに平均し、ニュートラルベクトルとする。
2. 各感情の平均ベクトルからニュートラルベクトルを引いたものを感情ベクトルとする。

- スクリプト: `step3-getEmovitonVec/main.py`
- 入力: `step2-getHiddenState/output/ave_vector_{感情}.pt`
- 出力:
  - `step3-getEmovitonVec/output/emotion_vector_neutral.pt` (ニュートラルベクトル)
  - `step3-getEmovitonVec/output/emotion_vector_{感情}.pt` (感情ベクトル)

いずれも層ごとのベクトルのリスト (step2の出力と同じ形式)。

### step4

ローカルLLM (`my_local_llama3/`) にプロンプトを入力して応答を生成させ、入力と出力の各トークンについて、指定した層のhidden stateと各感情ベクトルとのコサイン類似度を計算する。

論文では、モデル全体の約2/3の深さの層 (Llama-3.1-8Bでは `hidden_states[21]` 付近) が出力しようとしている感情をよく表すとされている。そのため使用する層はスクリプト内の定数で切り替えられるようにする。
論文で行われている「中立的な対話の主成分の除去」によるノイズ除去は今回は実施しない。

- スクリプト: `step4-trackEmotion/main.py`
- 入力:
  - プロンプト (コマンドライン引数で指定)
  - `step3-getEmovitonVec/output/emotion_vector_{感情}.pt` (8感情分。`neutral` は使用しない)
- 出力: `step4-trackEmotion/output/{実行日時}.json`

#### 定数

| 定数             | 内容                                                     | 初期値 |
| ---------------- | -------------------------------------------------------- | ------ |
| `TARGET_LAYER`   | 使用する `hidden_states` のインデックス (0はEmbedding層) | `21`   |
| `MAX_NEW_TOKENS` | 生成する最大トークン数                                   | `200`  |

#### 処理の流れ

1. プロンプトを `Human: {プロンプト}\n\nAssistant:` の形式に整形する (対象モデルはベースモデルのため、論文と同じ形式を使う)。
2. モデルに応答を生成させる。再現性のためgreedy decodingとし、ベースモデルが次の `Human:` ターンを書き始めた時点 (`\nHuman:`) で生成を止める。
3. 整形済みプロンプトと生成結果を連結したトークン列を、改めてモデルに1回入力し、`hidden_states[TARGET_LAYER]` から全トークンのhidden stateを取得する (Causalなモデルのため、生成時に逐次取得した値と同じになる)。
4. 各トークンのhidden stateと各感情ベクトル (`TARGET_LAYER` の要素) とのコサイン類似度を計算する。
5. 結果をJSONで保存する。

#### 出力形式

```json
{
  "prompt": "ユーザーが入力したプロンプト",
  "response": "生成された応答",
  "layer": 21,
  "emotions": [
    "Joy",
    "Anticipation",
    "Trust",
    "Fear",
    "Surprise",
    "Sadness",
    "Disgust",
    "Anger"
  ],
  "tokens": [
    {
      "index": 0,
      "token": "Human",
      "role": "input",
      "similarity": {
        "Joy": 0.012,
        "Anticipation": -0.004,
        "Trust": 0.008,
        "Fear": -0.021,
        "Surprise": 0.003,
        "Sadness": -0.01,
        "Disgust": -0.015,
        "Anger": -0.018
      }
    }
  ]
}
```

- `token`: トークンをデコードした文字列
- `role`: 整形済みプロンプトのトークンは `input`、生成されたトークンは `output`
- 先頭の `<|begin_of_text|>` や、生成停止の原因になった `\nHuman:` のトークンも `tokens` に含まれる
- `similarity`: 各感情ベクトルとのコサイン類似度

#### 解釈上の注意

- 論文によると、感情ベクトルはその時点で効いている感情を表し、会話全体を通した持続的な感情状態は表さない。トークンごとの値は変動が大きいため、近くのトークンで平均すると解釈しやすい。
- `Assistant:` のコロンのトークン (応答直前) の値は、これから生成される応答の感情をよく予測するとされている。
