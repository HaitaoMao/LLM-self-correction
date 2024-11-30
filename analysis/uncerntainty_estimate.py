from __future__ import division
import argparse
import copy
import glob
import json
import random

import numpy as np
import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ipdb
import pickle
import re

from utils import init_model, device, init_nli_model

PROMPTSTR, MOSTLIKELYGENSTR, SAMPLEDGENSTR, UNIQUEGENSTR = "Prompt", "MostlikelyGeneration", "SampledGenerations", "UniqueGenerations"
transformers.logging.set_verbosity_error()

PREDICT, LOGITConfidence, LABEL = "predict", "logitConf", "label"


'''
    class refers to the implementation of the authors of semantic_uncertainty https://github.com/lorenzkuhn/semantic_uncertainty
    there are three steps:
    1. generating several sentences given the same context
    2. clustering them by semantic similarity implemented with the DeBERT model (for NLI task)
    3. entropy estimation
'''


class semanticUncertainty:

    def __init__(self, args):
        self.name = "semantic-uncertainty"

        self.tokenizer, self.model = init_model(args)
        self.model.eval()

        self.similarity_token, self.similarity_model = init_nli_model(args)
        self.similarity_model.eval()

    @torch.no_grad()
    def entropy_estimate(self, input_text, input_json_file):
        with open(input_json_file, "r") as reader:
            input_json_list = json.load(reader)
            for idx, json_data in tqdm(enumerate(input_json_list), desc="estimate"):
                negative_log_likelihood = 0
                for gens in json_data[UNIQUEGENSTR]:

                    class_avg_likelihood = 0
                    for gen in gens:
                        input_text = copy.deepcopy(json_data['input'])
                        prompt_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                        input_ids = self.tokenizer(input_text + gen, return_tensors="pt").input_ids.to(device)
                        label_ids = self.tokenizer(input_text + gen, return_tensors="pt").input_ids.to(device)
                        label_ids[:, :prompt_ids.shape[-1]] = -100
                        class_avg_likelihood += float(self.model(input_ids, labels=label_ids,
                                                                 output_hidden_states=True).loss)
                    class_avg_likelihood /= len(gens)

                    negative_log_likelihood += class_avg_likelihood
                input_json_list[idx]["uncerntainty"] = negative_log_likelihood / len(json_data[UNIQUEGENSTR])
                input_json_list[idx].pop(UNIQUEGENSTR)
            return input_json_list

    @torch.no_grad()
    def generate(self, args, input_file):
        with open(input_file, 'r') as read_file:
            json_list = json.load(read_file)
            if args.task == "toxicity":
                random.shuffle(json_list)
                json_list = random.sample(json_list, int(25000*0.4))
            certainty_json_list = []
            for json_data in tqdm(json_list, desc="generate"):
                if int(json_data["round"]) > 5 and args.task == "toxicity":
                    continue
                generation_json_data = copy.deepcopy(json_data)
                input_text = json_data["input"]
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                all_generations = []
                for i in range(args.number_of_generations):
                    generation = self.model.generate(input_ids.to(device),
                                                     do_sample=True,
                                                     num_return_sequences=1,
                                                     num_beams=args.num_beams,
                                                     max_new_tokens=args.max_new_tokens,
                                                     temperature=args.temperature,
                                                     top_p=args.top_p,
                                                     pad_token_id=self.tokenizer.eos_token_id
                                                     )

                    prompt_length = input_ids.shape[1]
                    pure_answer = self.tokenizer.decode(generation[0][prompt_length:], skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
                    generation = ' '.join(copy.deepcopy(pure_answer.strip()).split('\n'))
                    all_generations.append(generation)

                generation_json_data[SAMPLEDGENSTR] = copy.deepcopy(all_generations)
                certainty_json_list.append(generation_json_data)
            return certainty_json_list

    @torch.no_grad()
    def clustering(self, args, input_json_file):
        with open(input_json_file, 'r') as reader:
            input_json_list = json.load(reader)
            for idx, json_data in tqdm(enumerate(input_json_list), desc="clustering"):
                pending = copy.deepcopy(json_data[SAMPLEDGENSTR])
                unique_gens = []
                while len(pending) > 0:
                    anchor = copy.deepcopy(pending[0])
                    one_cluster = [anchor]
                    del_idx = [0]
                    for i in range(1, len(pending), 1):
                        nli_input = anchor + " [SEP] " + pending[i]
                        encoded_input = self.similarity_token(nli_input, padding='max_length', max_length=args.max_length_clf*2, return_tensors='pt')[
                            "input_ids"].to(device)
                        prediction = self.similarity_model(encoded_input)['logits']
                        predicted_label = torch.argmax(prediction, dim=1)

                        reversed_nli_input = pending[i] + " [SEP] " + anchor
                        encoded_input = self.similarity_token(reversed_nli_input, padding='max_length', max_length=args.max_length_clf*2, return_tensors='pt')[
                            "input_ids"].to(device)
                        prediction = self.similarity_model(encoded_input)['logits']
                        reversed_predicted_label = torch.argmax(prediction, dim=1)

                        if predicted_label == 0 or reversed_predicted_label == 0:
                            continue
                        else:
                            one_cluster.append(copy.deepcopy(pending[i]))
                            del_idx.append(i)
                    #print("one cluster",len(one_cluster))
                    unique_gens.append(one_cluster)
                    if len(del_idx) > 0:
                        del_idx.reverse()
                        for del_ in del_idx:
                            pending.pop(del_)

                    if len(pending) == 1:
                        unique_gens.append([copy.deepcopy(pending[0])])
                        break
                    #print(idx, len(pending))
                input_json_list[idx][UNIQUEGENSTR] = unique_gens
                input_json_list[idx].pop(SAMPLEDGENSTR)
                # print(input_json_list[idx].keys())
            return input_json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--llm", type=str, default="mistral")
    parser.add_argument("--cluster", type=str, default="psu")
    parser.add_argument("--number_of_generations", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=125)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_length_clf", type=int, default=125)
    parser.add_argument("--task", type=str, default="bias")
    parser.add_argument("--bias_str", type=str, default="sexual_orientation",
                        choices=["age", "nationality", "religion", "sexual_orientation", "disability_status", "physical_appearance"])
    '''if task is social bias the max_new_tokens is 8'''

    args = parser.parse_args()
    if args.task == "bias":
        args.max_new_tokens = 8
        args.max_length_clf = 8
    estimator = semanticUncertainty(args)

    if args.task == "toxicity":
        json_data_list = estimator.entropy_estimate(
            args, "experiments/realtoxicity/toxicity.selfcorrect.clustering.json")
        with open("experiments/realtoxicity/toxicity.selfcorrect.entropy.json", "w") as writer:
            json.dump(json_data_list, writer)

