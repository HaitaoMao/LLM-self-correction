from __future__ import division
import argparse
import copy
import json

import numpy as np
import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ipdb
import pickle
import re

from utils import init_model, device, init_nli_model

PROMPTSTR, MOSTLIKELYGENSTR, SAMPLEDGENSTR = "Prompt", "MostlikelyGeneration", "SampledGenerations"
transformers.logging.set_verbosity_error()

PREDICT, LOGITConfidence, LABEL = "predict", "logitConf", "label"


class ECEError:
    def __init__(self, args, tokenizer, llm):
        self.name = "ece calibration error"
        self.tokenizer = tokenizer
        self.model = llm
        self.args = args

    def get_logit(self, file_path, prompts):
        sample2predict = {}

        for sample_idx, line in tqdm(enumerate(open(file_path).readlines())):
            history = ""
            context, question, choices, label = line.strip().split("\t")[0], line.strip().split("\t")[1], \
                line.strip().split("\t")[2], line.strip().split("\t")[-1]

            for round, prompt in enumerate(prompts):

                if round not in sample2predict.keys():
                    sample2predict[round] = {}

                instruct = copy.deepcopy(prompt)

                input = instruct.replace("#CONTEXT", context) \
                    .replace("#QUESTION", question) \
                    .replace("#CHOICES", choices)

                if round > 0:
                    input = history + "\n" + input

                inputs = copy.deepcopy(input)
                inputs = self.tokenizer(inputs, return_tensors="pt")
                max_tokens = 10
                # if round == 1: max_tokens = 125
                inputs_ids = self.tokenizer(copy.deepcopy(input),
                                            return_tensors="pt").input_ids.to(device)
                model_outputs = self.model.generate(inputs_ids, max_new_tokens=max_tokens,
                                                    pad_token_id=self.tokenizer.eos_token_id)

                prompt_length = inputs_ids.shape[1]
                pure_answer = self.tokenizer.decode(model_outputs[0][prompt_length:], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
                single_answer = copy.deepcopy(pure_answer.strip())  # .split('\n')[0]
                history = history + "\n\n" + input.strip() + " " + single_answer + "</s>"

                answer_choices = ["(a)", "(b)", "(c)"]
                predict2logit = {}
                determine_sum = 0
                for choice in answer_choices:
                    inputs_ids = self.tokenizer(copy.deepcopy(input) + " " + copy.deepcopy(choice),
                                                return_tensors="pt").input_ids.to(device)
                    label_ids = self.tokenizer(copy.deepcopy(input) + " " + copy.deepcopy(choice),
                                               return_tensors="pt").input_ids
                    prompt_ids = self.tokenizer(copy.deepcopy(input), return_tensors="pt").input_ids.to(device)
                    label_ids[:, :prompt_ids.shape[-1]] = -100
                    label_ids = label_ids.to(device)
                    negative_log_likelihood = self.model(inputs_ids, labels=label_ids).loss
                    logit = torch.exp(-1 * negative_log_likelihood)
                    predict2logit[choice] = float(logit)
                    determine_sum += float(logit)
                    torch.cuda.empty_cache()

                (predict_label, logit) = sorted(predict2logit.items(), key=lambda x: x[1], reverse=True)[0]
                logit /= determine_sum

                sample2predict[round][sample_idx] = {
                    PREDICT: predict_label.strip(), LOGITConfidence: logit, LABEL: label.strip()}
                torch.cuda.empty_cache()

        return sample2predict

    def get_mul_round_logit(self, file_path, prompts):
        sample2predict = self.get_logit(file_path, prompts)
        for round in range(len(sample2predict.keys())):
            ece_error = self.ece_score(sample2predict[round], self.args.num_bins)
            print("Self-correct\tround:{}\tece_error:{}".format(round, ece_error))

    def get_multiple_logit(self, file_path, prompts, is_load=True):
        category = re.search(r'/([^/]+)\.txt$', file_path).group(1)
        # ipdb.set_trace()
        sample2predict = {}
        idx = 0
        lines = open(file_path).readlines()
        if is_load:
            try:
                with open(f"/data/bochuan/selfcorrection/nips24selfcorrection/results/bias_logits/{category}_{len(prompts)}.txt", "rb") as f:
                    sample2predict = pickle.load(f)
                    existing_idx = max(list(sample2predict.keys()))
                    if len(lines) - 1 == existing_idx:
                        return sample2predict
                    lines = lines[existing_idx:]
            except:
                with open(f"/data/bochuan/selfcorrection/nips24selfcorrection/results/bias_logits/{category}_{len(prompts)}.txt", "wb") as f:
                    pickle.dump(sample2predict, f)

        for line in tqdm(lines):
            context, question, str_choices, label = line.strip().split("\t")[0], line.strip().split("\t")[1], \
                line.strip().split("\t")[2], line.strip().split("\t")[-1]
            history, round = "", 1

            records = []
            for prompt in prompts:
                instruct = copy.deepcopy(prompt)

                input = instruct.replace("#CONTEXT", context) \
                    .replace("#QUESTION", question) \
                    .replace("#CHOICES", str_choices)

                if round > 1:
                    input = history + "\n" + input

                choices = ["(a)", "(b)", "(c)"]
                predict2logit = {}
                determine_sum = 0
                for choice in choices:
                    inputs_ids = self.tokenizer(copy.deepcopy(input) + " " + copy.deepcopy(choice),
                                                return_tensors="pt").input_ids.to(device)
                    label_ids = self.tokenizer(copy.deepcopy(input) + " " + copy.deepcopy(choice),
                                               return_tensors="pt").input_ids
                    # print(inputs_ids.shape,label_ids.shape)
                    prompt_ids = self.tokenizer(copy.deepcopy(input), return_tensors="pt").input_ids.to(device)
                    label_ids[:, :prompt_ids.shape[-1]] = -100
                    label_ids = label_ids.to(device)
                    negative_log_likelihood = self.model(inputs_ids, labels=label_ids).loss
                    logit = torch.exp(-1 * negative_log_likelihood)
                    predict2logit[choice] = float(logit)
                    determine_sum += float(logit)
                    torch.cuda.empty_cache()

                (predict_label, logit) = sorted(predict2logit.items(), key=lambda x: x[1], reverse=True)[0]
                logit /= determine_sum

                torch.cuda.empty_cache()

                inputs = copy.deepcopy(input)
                inputs = self.tokenizer(inputs, return_tensors="pt")
                max_tokens = 10
                # if round == 1: max_tokens = 125

                model_outputs = self.model.generate(inputs.input_ids.to('cuda'), max_new_tokens=max_tokens,
                                                    pad_token_id=self.tokenizer.eos_token_id)

                prompt_length = inputs["input_ids"].shape[1]
                pure_answer = self.tokenizer.decode(model_outputs[0][prompt_length:], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
                single_answer = copy.deepcopy(pure_answer.strip())  # .split('\n')[0]
                history = history + "\n\n" + input.strip() + " " + single_answer + "</s>"
                round += 1

                # print(logit)
                records.append({PREDICT: predict_label.strip(), LOGITConfidence: logit, LABEL: label.strip()})
            sample2predict[idx] = records
            with open(f"/data/bochuan/selfcorrection/nips24selfcorrection/results/bias_logits/{category}_{len(prompts)}.txt", "wb") as f:
                pickle.dump(sample2predict, f)
            idx += 1

        return sample2predict

    def ece_score(self, sample2predict, num_bins=2):
        # get logit confidence interval
        confidence_list = []
        for key in sample2predict.keys():
            confidence_list.append(sample2predict[key][LOGITConfidence])

        # get bins
        min_conf, max_conf = min(confidence_list), max(confidence_list)
        interval = (max_conf - min_conf) / num_bins
        # print(interval)
        bins_list = [[] for i in range(num_bins)]

        for i in range(num_bins):
            interval_start, interval_end = min_conf + interval * i, min_conf + interval * (i + 1)
            for key in sample2predict.keys():
                logit_confidence = sample2predict[key][LOGITConfidence]
                if logit_confidence >= interval_start and logit_confidence < interval_end:
                    bins_list[i].append(sample2predict[key])

        ece_error = 0
        # get ece error
        for bins in bins_list:
            acc, conf = 0, 0
            for sample in bins:
                if sample[LABEL] == sample[PREDICT]:
                    acc += 1
                conf += sample[LOGITConfidence]
            acc /= len(bins)
            conf /= len(bins)

            ece_error += abs(acc - conf) * (len(bins) / len(sample2predict.keys()))

        return ece_error

    @torch.no_grad()
    def get_logit_conf(self, file_path, prompts):
        sample2predict = {}

        for sample_idx, line in tqdm(enumerate(open(file_path).readlines())):
            sample2predict[sample_idx] = []
            history = ""
            context, question, choices, label = line.strip().split("\t")[0], line.strip().split("\t")[1], \
                line.strip().split("\t")[2], line.strip().split("\t")[-1]
            first_choice = None
            for round, prompt in enumerate(prompts):

                instruct = copy.deepcopy(prompt)

                input = instruct.replace("#CONTEXT", context) \
                    .replace("#QUESTION", question) \
                    .replace("#CHOICES", choices)

                if round > 0:
                    input = history + "\n" + input

                inputs = copy.deepcopy(input)
                inputs = self.tokenizer(inputs, return_tensors="pt")
                max_tokens = 10
                inputs_ids = self.tokenizer(copy.deepcopy(input),
                                            return_tensors="pt").input_ids.to(device)
                model_outputs = self.model.generate(inputs_ids, max_new_tokens=max_tokens,
                                                    pad_token_id=self.tokenizer.eos_token_id)

                prompt_length = inputs_ids.shape[1]
                pure_answer = self.tokenizer.decode(model_outputs[0][prompt_length:], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
                single_answer = copy.deepcopy(pure_answer.strip())  # .split('\n')[0]
                history = history + "\n\n" + input.strip() + " " + single_answer + "</s>"

                answer_choices = ["(a)", "(b)", "(c)"]
                predict2logit = {}
                determine_sum = 0
                for choice in answer_choices:
                    inputs_ids = self.tokenizer(copy.deepcopy(input) + " " + copy.deepcopy(choice),
                                                return_tensors="pt").input_ids.to(device)
                    label_ids = self.tokenizer(copy.deepcopy(input) + " " + copy.deepcopy(choice),
                                               return_tensors="pt").input_ids
                    prompt_ids = self.tokenizer(copy.deepcopy(input), return_tensors="pt").input_ids.to(device)
                    label_ids[:, :prompt_ids.shape[-1]] = -100
                    label_ids = label_ids.to(device)
                    negative_log_likelihood = self.model(inputs_ids, labels=label_ids).loss
                    logit = torch.exp(-1 * negative_log_likelihood)
                    predict2logit[choice] = float(logit)
                    determine_sum += float(logit)
                    torch.cuda.empty_cache()

                for key in predict2logit.keys():
                    predict2logit[key] /= determine_sum
                (predict_label, logit) = sorted(predict2logit.items(), key=lambda x: x[1], reverse=True)[0]

                torch.cuda.empty_cache()

                if round == 0:
                    first_choice = predict_label
                    sample2predict[sample_idx].append(logit)
                else:
                    sample2predict[sample_idx].append(predict2logit[first_choice])

        avg_logit = np.zeros((1, len(prompts)))
        for key in sample2predict.keys():
            avg_logit += np.array(sample2predict[key])

        print(avg_logit / len(open(file_path).readlines()))


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
        if args.step != 2:
            self.tokenizer, self.model = init_model(args)
            self.model.eval()

        self.similarity_token, self.similarity_model = init_nli_model(args)
        self.similarity_model.eval()

    @torch.no_grad()
    def generate(self, args, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # return the most likely output with greedy decoding
        most_likely_generation = self.model.generate(inputs.input_ids.to(device),
                                                     num_beams=1,
                                                     do_sample=False,
                                                     max_new_tokens=args.max_new_tokens,
                                                     pad_token_id=self.tokenizer.eos_token_id)
        prompt_length = inputs["input_ids"].shape[1]
        pure_answer = self.tokenizer.decode(most_likely_generation[0][prompt_length:], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        most_likely_generation = copy.deepcopy(' '.join(copy.deepcopy(pure_answer.strip()).split('\n')))

        all_generations = []
        # get all potential outputs
        for i in range(args.number_of_generations):
            generation = self.model.generate(inputs.input_ids.to(device),
                                             do_sample=True,
                                             num_return_sequences=1,
                                             num_beams=args.num_beams,
                                             max_new_tokens=args.max_new_tokens,
                                             temperature=args.temperature,
                                             top_p=args.top_p,
                                             pad_token_id=self.tokenizer.eos_token_id
                                             )

            prompt_length = inputs["input_ids"].shape[1]
            pure_answer = self.tokenizer.decode(generation[0][prompt_length:], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
            generation = ' '.join(copy.deepcopy(pure_answer.strip()).split('\n'))
            all_generations.append(generation)

        return {PROMPTSTR: input_text, MOSTLIKELYGENSTR: most_likely_generation, SAMPLEDGENSTR: all_generations}

    @torch.no_grad()
    def entropy_estimate(self, input_text, unique_gens):
        # print(unique_gens)
        negative_log_likelihood = 0
        for gens in unique_gens:

            class_avg_likelihood = 0
            for gen in gens:
                prompt_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                input_ids = self.tokenizer(input_text + gen, return_tensors="pt").input_ids.to(device)
                label_ids = self.tokenizer(input_text + gen, return_tensors="pt").input_ids.to(device)
                label_ids[:, :prompt_ids.shape[-1]] = -100
                class_avg_likelihood += float(self.model(input_ids, labels=label_ids, output_hidden_states=True).loss)
            class_avg_likelihood /= len(gens)

            negative_log_likelihood += class_avg_likelihood

        return negative_log_likelihood / len(unique_gens)

    @torch.no_grad()
    def clustering_batch(self, args, prompt_result):
        sampled_gens = list(set(prompt_result[SAMPLEDGENSTR]))
        # print(sampled_gens)
        if len(sampled_gens) < 1:
            return False
        if len(sampled_gens) == 1:
            return [sampled_gens]
        unique_gens = []
        similarity_dict = {}

        pending = copy.deepcopy(sampled_gens)

        while len(pending) > 0:
            anchor = pending[0]
            one_cluster = [anchor]
            del_idx = []
            for i in range(1, len(pending), 1):
                nli_input = anchor + " [SEP] " + pending[i]
                encoded_input = self.similarity_token(nli_input, padding='max_length', max_length=1000, return_tensors='pt')[
                    "input_ids"].to(device)
                prediction = self.similarity_model(encoded_input)['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reversed_nli_input = pending[i] + " [SEP] " + anchor
                encoded_input = self.similarity_token(reversed_nli_input, padding='max_length', max_length=1000, return_tensors='pt')[
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

        return unique_gens  

    @torch.no_grad()
    def entropy_estimate_batch(self, args, input_text, unique_gens):
        negative_log_likelihood = 0
        for gens in unique_gens:

            class_avg_likelihood = 0
            for gen in gens:
                prompt_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                input_ids = self.tokenizer(input_text + gen, return_tensors="pt").input_ids.to(device)
                label_ids = self.tokenizer(input_text + gen, return_tensors="pt").input_ids.to(device)
                label_ids[:, :prompt_ids.shape[-1]] = -100
                class_avg_likelihood += float(self.model(input_ids, labels=label_ids, output_hidden_states=True).loss)
            class_avg_likelihood /= len(gens)

            negative_log_likelihood += class_avg_likelihood

        return negative_log_likelihood / len(unique_gens)

    @torch.no_grad()
    def get_generations(self, args, data_path, prompting_list, exp_type, is_load=True):
        prompt_round = len(prompting_list)

        uncertainty_results = {"prompts": prompting_list, "round": prompt_round,
                               "results": [], "questions": [], "inputs_texts": []}

        with open(data_path, "r") as file_p:
            datas = file_p.readlines()
        questions = [json.loads(x.strip())["prompt"]['text'] for x in datas]
        name = f"{exp_type}_result"
        save_path = f"/data/bochuan/selfcorrection/nips24selfcorrection/results/toxic_inter_uncertain/{name}.txt"
        if not is_load:
            with open(save_path, "wb") as f:
                pickle.dump(uncertainty_results, f)
        else:
            try:
                with open(save_path, "rb") as f:
                    uncertainty_results = pickle.load(f)
                    existing_idx = len(uncertainty_results['results'])
                    questions = questions[existing_idx]
            except:
                with open(save_path, "wb") as f:
                    pickle.dump(uncertainty_results, f)

        for question in tqdm(questions):
            uncertainty_results["questions"].append(question)
            history = ""
            round = 1
            prompting_results_list, input_texts_list = [], []
            for prompt in prompting_list:
                input_text = copy.deepcopy(prompt.replace("#QUESTION", question))

                input_text = history + input_text

                input_texts_list.append(input_text)

                inputs = self.tokenizer(input_text, return_tensors="pt")

                model_outputs = self.model.generate(inputs.input_ids.to(device), max_new_tokens=args.max_new_tokens,
                                                    pad_token_id=self.tokenizer.eos_token_id)

                prompt_length = inputs["input_ids"].shape[1]
                pure_answer = self.tokenizer.decode(model_outputs[0][prompt_length:], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)

                answer = ' '.join(copy.deepcopy(pure_answer.strip()).split('\n'))
                end_id = "</s>"
                if "llama3" in args.llm:
                    end_id = "<|eot_id|>"
                history = history + input_text.strip() + " " + answer + end_id

                prompting_results = self.generate(args, input_text)

                prompting_results_list.append(prompting_results)
                round += 1

            uncertainty_results["results"].append(prompting_results_list)
            uncertainty_results["inputs_texts"].append(input_texts_list)
            with open(save_path, "wb") as f:
                pickle.dump(uncertainty_results, f)


    @torch.no_grad()
    def get_entropy(self, args, data_path, prompting_list, exp_type, is_load=True):
        name = f"{exp_type}_result"
        load_path = f"/data/bochuan/selfcorrection/nips24selfcorrection/results/toxic_inter_uncertain/{name}.txt"
        load_path2 = f"/data/bochuan/selfcorrection/nips24selfcorrection/results/toxic_unique_gen_uncertain/{name}.txt"
        save_path = f"/data/bochuan/selfcorrection/nips24selfcorrection/results/toxic_result_uncertain/{name}.txt"

        with open(load_path, "rb") as f:
            uncertainty_results = pickle.load(f)

        with open(load_path2, "rb") as f:
            unique_gens_results = pickle.load(f)

        input_texts = uncertainty_results["inputs_texts"]
        # num_samples, num_round
        uncertainty_results = uncertainty_results["results"]
        # num_samples, num_round, [PROMPTSTR, MOSTLIKELYGENSTR, SAMPLEDGENSTR]

        entropy_results = []
        if is_load:
            try:
                with open(save_path, "rb") as f:
                    entropy_results = pickle.load(f)
                    start_idx = len(entropy_results)
                    input_texts = input_texts[start_idx:]
                    uncertainty_results = uncertainty_results[start_idx:]
                    unique_gens_results = unique_gens_results[start_idx:]
            except:
                with open(save_path, "wb") as f:
                    pickle.dump(entropy_results, f)
        else:
            with open(save_path, "wb") as f:
                pickle.dump(entropy_results, f)

        for uncertainty_result, input_text, unique_gens_result in tqdm(zip(uncertainty_results, input_texts, unique_gens_results)):
            single_entropy_results = []
            for round_idx, (prompting_results, prompt, unique_gen) in enumerate(zip(uncertainty_result, input_text, unique_gens_result)):
                entropy = self.entropy_estimate(prompt, unique_gen)
                # prompt
                print_result = "EXPTYPE:{}\tROUND:{}\tENTROPY:{}".format(exp_type, round_idx, entropy)
                single_entropy_results.append(entropy)
                print(entropy)
            entropy_results.append(single_entropy_results)
            with open(save_path, "wb") as f:
                pickle.dump(entropy_results, f)

        
    @torch.no_grad()
    def get_cluster(self, args, data_path, prompting_list, exp_type, is_load=True):
        name = f"{exp_type}_result"
        load_path = f"/data/bochuan/selfcorrection/nips24selfcorrection/results/toxic_inter_uncertain/{name}.txt"
        save_path = f"/data/bochuan/selfcorrection/nips24selfcorrection/results/toxic_unique_gen_uncertain/{name}.txt"

        with open(load_path, "rb") as f:
            uncertainty_results = pickle.load(f)
        input_texts = uncertainty_results["inputs_texts"]
        # num_samples, num_round
        uncertainty_results = uncertainty_results["results"]
        # num_samples, num_round, [PROMPTSTR, MOSTLIKELYGENSTR, SAMPLEDGENSTR]

        unique_gens_results = []
        if is_load:
            try:
                with open(save_path, "rb") as f:
                    unique_gens_results = pickle.load(f)
                    start_idx = len(unique_gens_results)
                    input_texts = input_texts[start_idx:]
                    uncertainty_results = uncertainty_results[start_idx:]
            except:
                with open(save_path, "wb") as f:
                    pickle.dump(unique_gens_results, f)
        else:
            with open(save_path, "wb") as f:
                pickle.dump(unique_gens_results, f)

        for uncertainty_result, input_text in tqdm(zip(uncertainty_results, input_texts)):
            single_unique_gens_results = []
            for round_idx, (prompting_results, prompt) in enumerate(zip(uncertainty_result, input_text)):
                unique_gens = self.clustering(prompting_results)
                single_unique_gens_results.append(unique_gens)
            unique_gens_results.append(single_unique_gens_results)
            with open(save_path, "wb") as f:
                pickle.dump(unique_gens_results, f)

    @torch.no_grad()
    def clustering(self, prompt_result):
        sampled_gens = list(set(prompt_result[SAMPLEDGENSTR]))
        # print(sampled_gens)
        if len(sampled_gens) < 1:
            return False
        if len(sampled_gens) == 1:
            return [sampled_gens]
        unique_gens = []
        similarity_dict = {}

        pending = copy.deepcopy(sampled_gens)

        while len(pending) > 0:
            anchor = pending[0]
            one_cluster = [anchor]
            del_idx = []
            for i in range(1, len(pending), 1):
                nli_input = anchor + " [SEP] " + pending[i]
                encoded_input = self.similarity_token(nli_input, padding='max_length', max_length=1000, return_tensors='pt')[
                    "input_ids"].to(device)
                prediction = self.similarity_model(encoded_input)['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reversed_nli_input = pending[i] + " [SEP] " + anchor
                encoded_input = self.similarity_token(reversed_nli_input, padding='max_length', max_length=1000, return_tensors='pt')[
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

        return unique_gens  # [key for key in similarity_dict.keys() if similarity_dict[key] is True]

    @torch.no_grad()
    def get_uncerntainty_old(self, args, data_path, prompting_list, exp_type):

        with open(data_path, "r") as file_p:
            datas = file_p.readlines()[:2500]
        questions = [json.loads(x.strip())["prompt"]['text'] for x in datas]

        for question in tqdm(questions):
            history = ""
            round = 1
            for prompt in prompting_list:
                input_text = copy.deepcopy(prompt.replace("#QUESTION", question))

                if round > 1:
                    input_text = history + input_text

                inputs = self.tokenizer(input_text, return_tensors="pt")

                model_outputs = self.model.generate(inputs.input_ids.to(device), max_new_tokens=args.max_new_tokens,
                                                    pad_token_id=self.tokenizer.eos_token_id)
                # 12

                prompt_length = inputs["input_ids"].shape[1]
                pure_answer = self.tokenizer.decode(model_outputs[0][prompt_length:], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)

                answer = ' '.join(copy.deepcopy(pure_answer.strip()).split('\n'))
                end_id = "</s>"
                if "llama3" in args.llm:
                    end_id = "<|eot_id|>"
                history = history + input_text.strip() + " " + answer + end_id

                prompting_results = self.generate(args, input_text)
                unique_gens = self.clustering(args, prompting_results)
                entropy = self.entropy_estimate(args, input_text, unique_gens)

                print("EXPTYPE:{}\tROUND:{}\tENTROPY:{}".format(exp_type, round, entropy))

                round += 1
