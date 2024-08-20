import random
import json


"""
raw_data = [line.strip() for line in open("prompts.jsonl")]

selected = random.sample(raw_data,10000)

with open("realtoxicity.txt",'w') as writer:
    writer.write("\n".join(selected))
    """


data_list = []

for line in open("realtoxicity.txt"):
    prompt = json.loads(line.strip())["prompt"]["text"]
    data_list.append(prompt)

with open("realtoxicity.prompt.txt",'w') as writer:
    writer.write("\n".join(data_list))
