# -*- coding: utf-8 -*-
import json
import random 

def load_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  

def tag_one_sample(sample):
    text = sample.get("text")
    entities = sample.get("entities")

    # tag sample
    tags = ["O" for _ in text]
    for e_t in entities:
        e, t = e_t.split("-")
        idx = text.find(e)
        tags[idx] = f"B-{t}"
        for i in range(idx+1, idx+len(e)):
            tags[i] = f"I-{t}"
    return zip(text, tags)

def tag_many_samples_gen(samples):
    for sample in samples:
        text_and_tags = tag_one_sample(sample)
        for char, tag in text_and_tags:
            yield f"{char}\t{tag}\n"
        yield "\n"

def train_dev_split(all_samples, rate=0.8):
    random.shuffle(all_samples)
    anchor = int(len(all_samples)*0.8)
    print(f"train: {anchor}")
    print(f"dev: {len(all_samples)-anchor}")
    return all_samples[:anchor], all_samples[anchor:]


if __name__ == "__main__":
    path = r"智源-水利知识图谱构建挑战赛\data\bmes_train\bmes_train.json"
    fancynlp_train_path = r"智源-水利知识图谱构建挑战赛\data2\bmes_train_fancynlp.txt"
    fancynlp_dev_path = r"智源-水利知识图谱构建挑战赛\data2\bmes_dev_fancynlp.txt"
    data = load_file(path)
    trains, devs = train_dev_split(data)


    with open(fancynlp_train_path, "w", encoding="utf-8", newline="") as f:
        for line in tag_many_samples_gen(trains):
            f.write(line)
    
    with open(fancynlp_dev_path, "w", encoding="utf-8", newline="") as f:
        for line in tag_many_samples_gen(devs):
            f.write(line)



