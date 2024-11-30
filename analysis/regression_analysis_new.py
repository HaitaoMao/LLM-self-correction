from __future__ import division
import glob
from tqdm import tqdm
import numpy as np
from numpy import mean
from utils import *
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import ipdb 
import copy
import pickle
from CCS import CCS, MLPProbe, infer

def format_harm(text, is_harm):
    """
    Given an imdb example ("text") and corresponding label (0 for negative, or 1 for positive), 
    returns a zero-shot prompt for that example (which includes that label as the answer).
    
    (This is just one example of a simple, manually created prompt.)
    """
    if is_harm:
        answer = "(a) yes, this is harmful."
    else:
        answer = "(b) no, this is not harmful."
    
    return text + answer


@torch.no_grad()
def get_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given a decoder model and some text, gets the hidden states (in a given layer, by default the last) on that input text

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize (adding the EOS token this time)
    input_ids = tokenizer(input_text + tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)

    # get the last layer, last token hidden states
    hs_tuple = output["hidden_states"]
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

    return hs


@torch.no_grad()
def get_all_hidden_states(model, tokenizer, datas, is_load=True):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    # setup
    model.eval()
    
    data_path = "experiments/harmful/harmful.probe_intermediate.txt"
    
    try:    
        with open(data_path, "rb") as f:
            probe_intermediate_dict = pickle.load(f)
            exist_idx = len(probe_intermediate_dict["pos"])
        max_round = 0
        for data in datas:
            max_round = max(max_round, data['round'])
        exist_idx = int(exist_idx * (max_round / (max_round - 1)))
        datas = datas[exist_idx:]            
    except:
        probe_intermediate_dict = {"pos": [], "neg": [], "labels": []}
        with open(data_path, "wb") as f:
            pickle.dump(probe_intermediate_dict, f)
        
    
    for idx, data in enumerate(datas):
        if data['round'] == 1:
            continue

        pos_inputs = format_harm(data['input'], True)
        neg_inputs = format_harm(data['input'], False)
        
        pos_hs = get_hidden_states(model, tokenizer, pos_inputs)
        neg_hs = get_hidden_states(model, tokenizer, neg_inputs)
        
        probe_intermediate_dict["pos"].append(pos_hs)
        probe_intermediate_dict["neg"].append(neg_hs)
        probe_intermediate_dict["labels"].append(data['label'])
        
        with open(data_path, "wb") as f:
            pickle.dump(probe_intermediate_dict, f)
    
       
def train_probe():
    intermediate_path = "experiments/harmful/harmful.probe_intermediate.txt"
    probe_path = "experiments/harmful/harmful.probe.pt"
    
    with open(intermediate_path, "rb") as f:
        hidden_dict = pickle.load(f)
    
    pos_hs = np.array(hidden_dict["pos"])
    neg_hs = np.array(hidden_dict["neg"])
    ys = np.array(hidden_dict["labels"])
    
    ccs = CCS(neg_hs, pos_hs)
    ccs.repeated_train()

    torch.save(ccs.probe, probe_path)


@torch.no_grad()
def probe_similarity(json_datas):
    probe_path = "experiments/harmful/harmful.probe.pt"
    probe_network = torch.load(probe_path)
    
    intermediate_path = "experiments/harmful/harmful.probe_intermediate.txt"
    with open(intermediate_path, "rb") as f:
        hidden_dict = pickle.load(f)
    
    pos_hidden_states, neg_hidden_states = hidden_dict["pos"], hidden_dict["neg"]
    
    round2sim = {}
    
    max_round = 0
    for data in json_datas:
        max_round = max(max_round, data['round'])
    
    
    for idx, (pos_hidden, neg_hidden) in enumerate(zip(pos_hidden_states, neg_hidden_states)):
        round = idx % (max_round - 1) + 2
        pos_prob = infer(neg_hidden, pos_hidden, probe_network)
        neg_prob = 1 - pos_prob
        if round not in round2sim.keys():
            round2sim[round] = {}
            round2sim[round]['pos'], round2sim[round]['neg'] = [], []
        
        round2sim[round]['pos'].append(float(pos_prob.item()))
        round2sim[round]['neg'].append(float(neg_prob.item()))

    for round in round2sim.keys():
        print(round, "\tpositive\t", np.mean(round2sim[round]['pos']),
              "\tnegative:\t", np.mean(round2sim[round]['neg']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default='advbench', choices=['wmt', 'bbq', 'realtoxicity'])
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--choices", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--questions", type=str, default="llm-self-defense/all_repsonses_llama.csv") # gpt llama
    parser.add_argument("--llm", type=str, default="mistral")  # llama3-70B
    parser.add_argument("--cluster", type=str, default="psu")  # llama3-70B
    parser.add_argument("--step", type=int, default=1)  # llama3-70B

    args = parser.parse_args()
    
    if args.step == 1:
        tokenizer, model = init_model(args)
        model.eval()

        with open("/data/bochuan/selfcorrection/nips24selfcorrection/experiments/harmful/harmful.selfcorrect.archive.json", "r") as f:
            datas = json.load(f)
        
        get_all_hidden_states(model, tokenizer, datas)    
    
    elif args.step == 2:
        train_probe()

    elif args.step == 3:
        with open("/data/bochuan/selfcorrection/nips24selfcorrection/experiments/harmful/harmful.selfcorrect.archive.json", "r") as f:
            datas = json.load(f)

        probe_similarity(datas)