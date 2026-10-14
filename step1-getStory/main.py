import datetime
from pathlib import Path
import random
from time import sleep
import requests
import json
import re

emotions = ["Joy", "Anticipation", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger"]
topics = [
"An artist discovers someone has tattooed their work",
"A family member announces they're converting to a different religion",
"Someone's childhood imaginary friend appears in their niece's drawings",
"A person finds out their biography was written without their knowledge",
"A neighbor starts a renovation project",
"Someone finds their grandmother's engagement ring in a pawn shop",
"A student learns their scholarship application was denied",
"A person's online friend turns out to live in the same city",
"A neighbor wants to install a fence",
"An adult child moves back in with their parents",
"An employee is asked to train their replacement",
"An athlete is asked to switch positions",
"A traveler's flight is delayed, causing them to miss an important event",
"A student is accused of plagiarism",
"A person discovers their mentor has retired without saying goodbye",
"Two friends both apply for the same job",
"A person runs into their ex at a mutual friend's wedding",
"Someone discovers their friend has been lying about their job",
"A person discovers their partner has been taking secret phone calls",
"A person discovers their child has the same teacher they had",
"A person's car is towed from their own driveway",
"Two friends realize they remember a shared event completely differently",
"Someone discovers their mother kept every school assignment",
"A person discovers their teenage diary has been published online",
"Someone finds out their medical records were mixed up with another patient's",
"A person finds out their article was published under someone else's name",
"An athlete doesn't make the team they expected to join",
"An employee is transferred to a different department",
"Someone receives a friend request from a childhood bully",
"A person finds out their surprise party has been cancelled",
"An employee finds out a junior colleague makes more money",
"A person finds out their partner has been learning their native language",
"A chef receives a harsh review from a food critic",
"A person learns their favorite restaurant is closing",
"Someone finds their childhood teddy bear at a yard sale",
"A homeowner discovers previous residents left items in the attic",
"Someone finds an unsigned birthday card in their mailbox",
"Someone discovers a hidden room in their new house",
"Two strangers realize they've been dating the same person",
"A person finds a hidden letter in a used book",
"Two siblings inherit their grandmother's house",
"Someone finds a wallet containing a large sum of cash",
"Someone receives an invitation to their high school reunion",
"Someone discovers their recipe has become famous under another name",
"A college student discovers their roommate has been reading their journal",
"A person finds out they were adopted through a DNA test",
"A family member wants to sell a cherished heirloom",
"Someone receives a package intended for the previous tenant",
"Someone's childhood home is about to be demolished",
"A person's invention is already patented by someone else",
"A neighbor's dog keeps escaping into their yard",
"A coach has to cut a player from the team",
"Someone learns their favorite author plagiarized their stories",
"A student finds out their scholarship was meant for someone else",
"Someone discovers their teenager has a secret social media account",
"Two roommates disagree about getting a pet",
"Two friends plan separate birthday parties on the same day",
"A person learns their childhood best friend doesn't remember them",
"A musician hears their song being performed by someone else",
"A person's manuscript is rejected by their dream publisher",
"A person finds old photos that contradict family stories",
"A person is asked to give a speech at their parent's retirement party",
"A student discovers their teacher follows them on social media",
"A parent finds an old letter they wrote but never sent",
"An employee discovers the company is being sold",
"A person accidentally sends a text to the wrong recipient",
"Two coworkers are stuck in an elevator for three hours",
"A student learns their thesis advisor is leaving the university",
"A person's longtime hobby becomes their child's obsession",
"Two colleagues are both considered for the same promotion",
"Two coworkers discover they went to the same summer camp",
"A tenant receives an eviction notice",
"Someone finds their parent's draft letter of resignation from decades ago",
"Someone finds out their best friend is moving across the country",
"A neighbor's tree falls on their property",
"Someone receives an apology letter years after the incident",
"A person discovers the tree they planted as a child has been cut down",
"Two siblings discover different versions of their inheritance",
"A person finds their childhood home listed for sale online",
"A homeowner learns their house was a former crime scene",
"Someone finds out they have a half-sibling they never knew about",
"A person learns their childhood bully became a therapist",
"Two people discover they've been working on identical projects",
"A person finds their spouse's secret savings account",
"A neighbor complains about noise levels",
"Someone finds their deceased parent's bucket list",
"A teacher receives an unexpected gift from a former student",
"An artist's work is displayed without their permission",
"Someone discovers their neighbor is secretly wealthy",
"A student receives a much lower grade than expected",
"A person learns their college is closing down",
"A neighbor asks to cut down a tree on the property line",
"Two strangers discover they share the same rare medical condition",
"Someone receives flowers with no card attached",
"Someone discovers their partner has been writing a novel about them",
"Someone finds a time capsule they don't remember burying",
"Someone finds their partner's bucket list",
"A neighbor asks to use part of the yard for a garden",
"A person learns their apartment building is going condo",
"Someone finds their college application essay published as an example",
]
n_stories = 5

URL = "http://150.89.236.80:8080/v1/chat/completions"
MODEL = "llama3.1:8b"

headers = {
    'Content-Type': 'application/json'
}

max_retries = 5
errors = []


def load_output(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    raise ValueError("output.json は JSON 配列である必要があります。")


def save_output(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_stories(text: str, total: int):
    # [story n] と ** story n** の両形式に対応してブロックを抽出する
    header_pattern = r"\[\s*story\s+(\d+)\s*\]|\*\*\s*story\s+(\d+)\s*\*\*"
    matches = list(re.finditer(header_pattern, text, flags=re.IGNORECASE))

    story_map = {}
    for i, match in enumerate(matches):
        num_str = match.group(1) or match.group(2)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        story_map[int(num_str)] = body

    missing = [i for i in range(1, total + 1) if i not in story_map or not story_map[i]]
    if missing:
        return [], missing

    return [story_map[i] for i in range(1, total + 1)], []

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

for emotion in emotions:
    for topic_index, topic in enumerate(topics, start=1):
        output_path = output_dir / f"output-{topic_index}.json"
        data = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": 
                 f"""
Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]
[story 2]
[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named.
                 """}
            ],
            "stream": False
        }

        retrycount = 0

        print(f"emotion={emotion}, topic={topic}")
        while retrycount < max_retries:
            try:
                response = requests.post(URL, headers=headers, data=json.dumps(data), timeout=10)

                # レスポンスの確認
                if response.status_code != 200:
                    raise Exception(f"APIエラー: {response.status_code} - {response.text}")
                
                # 返されたJSONからテキスト部分を抽出
                result = response.json()['choices'][0]['message']['content']

                stories, missing = extract_stories(result, n_stories)
                if missing:
                    raise Exception(f"出力形式エラー: [story n] が不足しています missing={missing}")

                output_items = load_output(output_path)
                for idx, story in enumerate(stories, start=1):
                    output_items.append({
                        "emotion": emotion,
                        "topic": topic,
                        "story_num": idx,
                        "story": story
                    })
                save_output(output_path, output_items)
                break

            except Exception as e:
                retrycount += 1
                print(f"通信エラー: {e} (retrycount={retrycount}/{max_retries})")

            if retrycount < max_retries:
                sleep(1)
            else:
                print("最大リトライ回数に達しました。")
                errors.append({
                    "emotion": emotion,
                    "topic": topic,
                })
        sleep(5)
print(f"全ての処理が完了しました。 NOW: {datetime.datetime.now()}")
print(f"エラーが発生したケース: {len(errors)} 件")
for error in errors:
    print(f" - Emotion: {error['emotion']}, Topic: {error['topic']}")