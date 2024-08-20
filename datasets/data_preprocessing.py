import copy
import glob
import json
import random


def extract_bbq(source_file, target_file):
    with open(target_file, 'w') as writer:
        lines = []

        for line in open(source_file):
            data = json.loads(line)
            if data['context_condition'] != 'ambig': continue
            context = data['context']
            question = data['question']
            ans0, ans1, ans2, label = data['ans0'], data['ans1'], data['ans2'], str(data['label'])
            
            stereotyped_group = " ".join(data["additional_metadata"]['stereotyped_groups'])
            """
            if "Gender" in source_file:
                new_group = []
                for i in stereotyped_group:
                    if i == "M":
                        new_group.append("Male")
                    elif i == "F":
                        new_group.append("Female")

                stereotyped_group.clear()
                stereotyped_group = copy.deepcopy(new_group)
            """
            if label == '0':
                label = "(a)"
            elif label == '1':
                label = "(b)"
            elif label == '2':
                label = "(c)"
            context = context + "\t" + question + "\t"+"(a) " + ans0 + " (b) " + ans1 + " (c) " + ans2 + "\t" + label  + "\t" + stereotyped_group
            lines.append(context)
        writer.write("\n".join(lines))


#extract_bbq("/Users/rayliu/Desktop/project/BBQ/data/Age.jsonl", 'data/bbq.age.txt')
for file in glob.glob("/Users/rayliu/Desktop/project/BBQ/data/*.jsonl"):
    stereotypical_group = file.split("/")[-1].replace(".jsonl","").lower()
    extract_bbq(file,"BBQ/"+stereotypical_group+".txt")


def extract_wikibio(file):
    reader = open(file,'r')
    wikibios = json.load(reader)
    #print(wikibios[0].keys())
    data_list = []

    for bio in wikibios[:200]:
            samples = []
            wiki_text = bio["wiki_bio_text"].strip()

            for gpt_bio, label in zip(bio["gpt3_sentences"], bio["annotation"]):
                samples.append([gpt_bio.strip(),label.strip()])
            sample = random.sample(samples,1)[0]
            gt_label = sample[1]
            data_dict = {"Context":wiki_text,"Sentence":sample[0], "Label":sample[1].replace("_"," ")}



            data_list.append(data_dict)
    #a = json.dumps(data_list)
    with open("wikibio/wikibio_debug.json",'w') as target_file:
        json.dump(data_list,target_file)



#extract_wikibio("wikibio/dataset_v3.json")
"""
with open("BBQ/bbq.debug.txt",'w') as writer:
    writer.write("\n".join(random.sample([i.strip() for i in open("BBQ/sexual_orientation.txt")],200)))"""



