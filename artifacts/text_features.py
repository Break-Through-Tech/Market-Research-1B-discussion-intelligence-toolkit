from standardize_convokit import iter_standardized
import pandas as pd

path = "data/local/standardized-v1/reddit-coarse-discourse-corpus.jsonl"

def word_count(text):
    return len(text.split())

def has_question(text):
    return '?' in text

def exclaim_count(text):
    return text.count("!")

def caps_ratio(text):
    count = 0
    caps_count = 0
    for char in text:
        if char.isalpha():
            count += 1
            if char.isupper():
                caps_count += 1
    if count == 0:
        return 0
    return caps_count / count

def has_quote(text):
    splits = text.split("\n")
    for split in splits:
        if split.startswith(">"):
            return True
    return False

def has_url(text):
    return 'http' in text


def extract_features(utterances):
    rows = []
    for u in utterances:
        if u.metadata.get("is_synthetic"):
            continue
        text = u.text_clean or ""

        dct = {
            'utterance_id': u.id,
            'conversation_id': u.conversation_id,
            'word_count': word_count(text),
            'char_count': len(text),
            'has_question': has_question(text),
            'exclaim_count': exclaim_count(text),
            'caps_ratio': caps_ratio(text),
            'has_quote': has_quote(text),
            'has_url': has_url(text),
            'is_deleted': u.is_deleted
        }
        rows.append(dct)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_utterances = []
    for conversation in iter_standardized(path):
        all_utterances.extend(conversation.utterances)

    df = extract_features(all_utterances)
    print(df.shape)
    print(df.head())
    df.to_csv("data/local/text_features.csv", index=False)
