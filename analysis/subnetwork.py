from __future__ import division
import glob
import math
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from utils import *
from scipy.stats import zscore
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import json


"""
    this class implements our analysis of the subnetwork activated by question + instructions, through approximating
    subnetworks with activations of neurons

"""

HOOK_MODULE_STR = "mlp.act_fn"  # "post_attention_layernorm"
BASELINE_SUBNETWORK_STR, SELF_CORRECT_SUBNETWORK_STR, NEGATION_SUBNETWORK_STR = "baseline_subnetwork", "selfcorrect_subnetwork", "negation_subnetwork"


def get_subnetwork_distribution(self, input_texts):
    features = None

    for input_text in input_texts:
        activation = self.get_activation(input_text)
        if features is None:
            features = activation
        else:
            features = torch.cat([features, activation], dim=0)

    return torch.mean(features), torch.cov(features)


@torch.no_grad()
def get_subnetwork(args, tokenizer, llm, input_file, output_dir, round=1):
    activations = {}

    def get_activation(name):
        def hook_func(module, input, output):
            activations[name] = output.cpu().detach().numpy()

        return hook_func

    hooks = []
    for name, module in llm.named_modules():
        if HOOK_MODULE_STR in name:
            hook = module.register_forward_hook(hook=get_activation(name))
            hooks.append(hook)

    if input_file.endswith('.json'):
        with open(input_file, 'r') as file:
            input_texts = [i["input"] for i in json.load(file) if int(i['round']) == round]
    else:
        input_texts = [i.strip() for i in open(args.input_file)]
    full_activation_matrix = None
    for idx, input_text in tqdm(enumerate(input_texts)):
        input_ids = tokenizer(input_text, return_tensors="pt")
        output = llm(input_ids.input_ids.to(device))

        activation_vector = None
        for key in activations.keys():
            activated_state = activations[key][:, -1, :]
            #print(activated_state.shape," shape of activation ")
            if activation_vector is None:
                activation_vector = copy.deepcopy(activated_state)
            else:
                activation_vector = np.concatenate([activation_vector, activated_state], axis=1, dtype=np.float16)

        if full_activation_matrix is None:
            full_activation_matrix = copy.deepcopy(activation_vector)
        else:
            full_activation_matrix = np.concatenate(
                [full_activation_matrix, copy.deepcopy(activation_vector)], axis=0, dtype=np.float16)
    np.save(output_dir, full_activation_matrix)
    for hook in hooks:
        hook.remove()


def get_activation_distri(samples_file, num_pca_components):
    sample_matrix = np.load(samples_file)

    scaler = StandardScaler()
    scaler.fit(sample_matrix)
    standard_matrix = scaler.transform(sample_matrix)

    embedding = Isomap(n_components=num_pca_components)
    dr_matrix = embedding.fit_transform(standard_matrix)
    mean_matrix, cov_matrix = np.mean(dr_matrix, axis=0), np.cov(dr_matrix, rowvar=False)

    return mean_matrix, cov_matrix, embedding, scaler


@torch.no_grad()
def get_in_distri_prob(args, tokenizer, llm, input_file, round, num_pca_components):
    baseline_mean, baseline_cov, baseline_pca, baseline_scaling = get_activation_distri(
        "experiments/realtoxicity/toxicity.baseline.npy", num_pca_components)
    selfcorrect_mean, selfcorrect_cov, selfcorrect_pca, selfcorrect_scaling = get_activation_distri(
        "experiments/realtoxicity/toxicity.selfcorrect.npy", num_pca_components)
    negation_mean, negation_cov, negation_pca, negation_scaling = get_activation_distri(
        "experiments/realtoxicity/toxicity.negation.npy", num_pca_components)


    baseline_inv, baseline_det = np.linalg.inv(baseline_cov), np.linalg.det(baseline_cov)
    selfcorrect_inv, selfcorrect_det = np.linalg.inv(selfcorrect_cov), np.linalg.det(selfcorrect_cov)
    negation_inv, negation_det = np.linalg.inv(negation_cov), np.linalg.det(negation_cov)

    def get_subnetwork_prob(x_activation_vector, target_subnetwork):
        mean, inverse_cov, det_cov, pca, scaling = baseline_mean, baseline_inv, baseline_det, baseline_pca, baseline_scaling

        if target_subnetwork == SELF_CORRECT_SUBNETWORK_STR:
            mean, inverse_cov, det_cov, pca, scaling = selfcorrect_mean, selfcorrect_inv, selfcorrect_det, selfcorrect_pca, selfcorrect_scaling
        elif target_subnetwork == NEGATION_SUBNETWORK_STR:
            mean, inverse_cov, det_cov, pca, scaling = negation_mean, negation_inv, negation_det, negation_pca, negation_scaling
        raw_x = copy.deepcopy(x_activation_vector)
        _raw_x = np.expand_dims(raw_x, axis=0)
        dr_x_ = scaling.transform(_raw_x)
        dr_x = pca.transform(dr_x_)

        x_mean = dr_x - mean

        exp_term = np.exp(0.5 * np.matmul(np.matmul(x_mean, inverse_cov), np.transpose(x_mean)))
        determ_term = np.sqrt(np.power(2.0 * math.pi, num_pca_components) * det_cov)
        return exp_term / determ_term
    
    
    smaple_matrix = np.load("experiments/realtoxicity/toxicity.selfcorrect.round"+str(round)+".npy")
    for idx in range(smaple_matrix.shape[0]):
        activation_vector = smaple_matrix[idx, :]
        in_baseline_prob, in_selfcorrect_prob, in_negation_prob =\
            get_subnetwork_prob(activation_vector, BASELINE_SUBNETWORK_STR),\
            get_subnetwork_prob(activation_vector, SELF_CORRECT_SUBNETWORK_STR),\
            get_subnetwork_prob(activation_vector, NEGATION_SUBNETWORK_STR)

        if idx % 1 == 0:
            print("round:{}\tbaseline_prob:{}\tselfcorrect_prob:{}\tnegation_prob:{}".format(
                round, in_baseline_prob[0][0], in_selfcorrect_prob[0][0], in_negation_prob[0][0]))


def get_dist():
    baseline_distr = np.load("tmp.baseline.distr.npy")
    selfcorrect_round1_distr = np.load("tmp.selfcorrect.distr.npy")
    activation_vector = np.load("tmp.activation.vector.npy")
    mmd_dist = MMD_loss()
    print("dist2selfcorr:{}\tdist2baseline:{}".format(mmd_dist(activation_vector,
          selfcorrect_round1_distr), mmd_dist(activation_vector, baseline_distr)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="mistral")
    parser.add_argument("--cluster", type=str, default="psu")

    args = parser.parse_args()

    get_in_distri_prob(args, tokenizer=None, llm=None,
                       input_file="experiments/realtoxicity/toxicity.selfcorrect.archive.json", round=2, num_pca_components=5)
