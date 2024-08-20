from __future__ import division
import argparse
import copy
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

        if args.step == 2:
            self.similarity_token, self.similarity_model = init_nli_model(args)
            self.similarity_model.eval()
        else:
            self.tokenizer, self.model = init_model(args)
            self.model.eval()


    @torch.no_grad()
    def entropy_estimate(self, input_text, json_list, output_file):
        try:
            with open(output_file, "r") as f:
                estimate_json_list = json.load(f)
                exist_idx = len(estimate_json_list)
        except:
            estimate_json_list = []
            with open(output_file, "w") as f:
                json.dump(estimate_json_list, f)
            exist_idx = 0
        
        json_list = json_list[exist_idx:]
        
        for idx, json_data in tqdm(enumerate(json_list), desc="estimate"):
            estimate_json_data = copy.deepcopy(json_data)
            
            negative_log_likelihood = 0
            for gens in json_data[UNIQUEGENSTR]:
                class_avg_likelihood = 0
                for gen in gens:
                    prompt_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                    input_ids = self.tokenizer(input_text + gen, return_tensors="pt").input_ids.to(device)
                    label_ids = self.tokenizer(input_text + gen, return_tensors="pt").input_ids.to(device)
                    label_ids[:, :prompt_ids.shape[-1]] = -100
                    class_avg_likelihood += float(self.model(input_ids, labels=label_ids,
                                                  output_hidden_states=True).loss)
                class_avg_likelihood /= len(gens)
                negative_log_likelihood += class_avg_likelihood
            estimate_json_data["uncerntainty"] = negative_log_likelihood / len(json_data[UNIQUEGENSTR])
            # estimate_json_list[idx]["uncerntainty"] = negative_log_likelihood / len(json_data[UNIQUEGENSTR])
            # estimate_json_list[idx].drop(UNIQUEGENSTR)
            estimate_json_list.append(estimate_json_data)
            
            with open(output_file, "w") as f:
                json.dump(estimate_json_list, f)
        # return certainty_json_l

    @torch.no_grad()
    def generate(self, args, json_list, output_file):
        try:
            with open(output_file, "r") as f:
                certainty_json_list = json.load(f)
                exist_idx = len(certainty_json_list)
        except:
            certainty_json_list = []
            with open(output_file, "w") as f:
                json.dump(certainty_json_list, f)
            exist_idx = 0
            
        json_list = json_list[exist_idx:]
        
        # certainty_json_list = []
        for json_data in tqdm(json_list, desc="generate"):
            # if int(json_data["round"]) > 5:
            #     continue
            if args.round != 0:
                if int(json_data["round"]) > args.round:
                    continue
            generation_json_data = copy.deepcopy(json_data)
            input_text = json_data["input"]
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            all_generations = []
            for i in range(args.number_of_generations):
                torch.cuda.empty_cache()
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

            #all_generations = map(generate_map_func, [i for i in range(args.number_of_generations)])
            generation_json_data[SAMPLEDGENSTR] = copy.deepcopy(all_generations)
            certainty_json_list.append(generation_json_data)
            
            with open(output_file, "w") as f:
                json.dump(certainty_json_list, f)
        # return certainty_json_list

    @torch.no_grad()    
    def clustering_old(self, args, json_list, output_file):
        try:
            with open(output_file, "r") as f:
                cluster_json_list = json.load(f)
                exist_idx = len(cluster_json_list)
        except:
            cluster_json_list = []
            with open(output_file, "w") as f:
                json.dump(cluster_json_list, f)
            exist_idx = 0
        
        json_list = json_list[exist_idx:]
        
        for idx, json_data in tqdm(enumerate(json_list), desc="clustering"):
            cluster_json_data = copy.deepcopy(json_data)
            pending = copy.deepcopy(json_data[SAMPLEDGENSTR])
            # TODO: check here
            unique_gens = []
            while len(pending) > 0:
                anchor = pending[0]
                one_cluster = [anchor]
                # TODO: check result
                del_idx = [anchor]
                for i in range(1, len(pending), 1):
                    nli_input = anchor + " [SEP] " + pending[i]
                    encoded_input = self.similarity_token(nli_input, padding='max_length', max_length=args.max_length_clf, return_tensors='pt')[
                        "input_ids"].to(device)
                    prediction = self.similarity_model(encoded_input)['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reversed_nli_input = pending[i] + " [SEP] " + anchor
                    encoded_input = self.similarity_token(reversed_nli_input, padding='max_length', max_length=args.max_length_clf, return_tensors='pt')[
                        "input_ids"].to(device)
                    prediction = self.similarity_model(encoded_input)['logits']
                    reversed_predicted_label = torch.argmax(prediction, dim=1)

                    if predicted_label == 0 or reversed_predicted_label == 0:
                        continue
                    else:
                        one_cluster.append(pending[i])
                        del_idx.append(pending[i])
                unique_gens.append(one_cluster)
                if len(del_idx) > 0:
                    for del_ in del_idx:
                        pending.remove(del_)

                if len(pending) == 1:
                    unique_gens.append([pending[0]])
                    break
            cluster_json_data[UNIQUEGENSTR] = unique_gens
            cluster_json_list.append(cluster_json_data)
            
            with open(output_file, "w") as f:
                json.dump(cluster_json_list, f)


    @torch.no_grad()    
    def clustering(self, args, json_list, output_file):
        try:
            with open(output_file, "r") as f:
                cluster_json_list = json.load(f)
                exist_idx = len(cluster_json_list)
        except:
            cluster_json_list = []
            with open(output_file, "w") as f:
                json.dump(cluster_json_list, f)
            exist_idx = 0
        
        json_list = json_list[exist_idx:]
        
        for idx, json_data in tqdm(enumerate(json_list), desc="clustering"):
            cluster_json_data = copy.deepcopy(json_data)
            pending = copy.deepcopy(json_data[SAMPLEDGENSTR])
            # TODO: check here
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
                    
            cluster_json_data[UNIQUEGENSTR] = unique_gens
            cluster_json_list.append(cluster_json_data)
            
            with open(output_file, "w") as f:
                json.dump(cluster_json_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--llm", type=str, default="mistral")
    parser.add_argument("--cluster", type=str, default="psu")
    parser.add_argument("--name", type=str, default="harmful")
    parser.add_argument("--number_of_generations", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=125)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--max_length_clf", type=int, default=125)
    '''if task is social bias the max_new_tokens is 8'''

    args = parser.parse_args()

    estimator = semanticUncertainty(args)
    """
    with open("experiments/realtoxicity/toxicity.baseline.sampling.json", 'w') as writer:
        json_list = estimator.generate(args, "experiments/realtoxicity/babies/toxicity.baseline.archive.json")
        json.dump(json_list, writer)

    with open("experiments/realtoxicity/toxicity.negation.sampling.json", 'w') as writer:
        json_list = estimator.generate(args, "experiments/realtoxicity/babies/toxicity.negation.archive.json")
        json.dump(json_list, writer)
    """
    if args.name == "harmful":
        path_tempelate = "/data/bochuan/selfcorrection/nips24selfcorrection/experiments/harmful/harmful.selfcorrect.{name}.json"
    if args.step == 1:     
        input_path = path_tempelate.format(name="archive")
        output_path = path_tempelate.format(name="sampling")
        
        with open(input_path, 'r') as reader:
            json_list = json.load(reader)
        # ipdb.set_trace()
        estimator.generate(args, json_list, output_path)
    
    elif args.step == 2: 
        input_path = path_tempelate.format(name="sampling")
        output_path = path_tempelate.format(name="cluster")
        
        with open(input_path, 'r') as reader:
            json_list = json.load(reader)
            
        estimator.clustering(args, json_list, output_path)
    
    elif args.step == 3:
        input_path = path_tempelate.format(name="cluster")
        output_path = path_tempelate.format(name="uncertainty")
        
        with open(input_path, 'r') as reader:
            json_list = json.load(reader)
        
        # ipdb.set_trace()
        estimator.entropy_estimate(args, json_list, output_path)