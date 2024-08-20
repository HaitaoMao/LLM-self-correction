from __future__ import division
import glob
import random
import pandas as pd
import torch
import numpy as np
from utils import *
from tqdm import tqdm
import sklearn


def data_redistri(file):
    data = pd.read_csv(file)
    texts, labels = list(data["comment_text"]), list(data["toxic"])

    idx = 0
    toxic_list, nontoxic_list = [], []
    for text, label in tqdm(zip(texts, labels)):
        if int(label) == 1:
            toxic_list.append(text)
        else:
            nontoxic_list.append(text)
    num_toxic_samples = len(toxic_list)
    nontoxic_list = random.sample(nontoxic_list, num_toxic_samples)

    random.shuffle(toxic_list)
    random.shuffle(nontoxic_list)

    train_toxic_list, valid_toxic_list = toxic_list[:int(
        0.9*num_toxic_samples)], toxic_list[int(0.9*num_toxic_samples):]
    train_nontoxic_list, valid_nontoxic_list = nontoxic_list[:int(
        0.9*num_toxic_samples)], nontoxic_list[int(0.9*num_toxic_samples):]
    # train_toxic_list.extend(train_nontoxic_list)
    # test_toxic_list.extend(test_nontoxic_list)

    train_data = {"texts": train_toxic_list + train_nontoxic_list,
                  "label": [1 for i in range(len(train_toxic_list))] + [0 for i in range(len(train_nontoxic_list))]}
    valid_data = {"texts": valid_toxic_list + valid_nontoxic_list,
                  "label": [1 for i in range(len(valid_toxic_list))] + [0 for i in range(len(valid_nontoxic_list))]}

    train_ = pd.DataFrame(data=train_data)
    valid_ = pd.DataFrame(data=valid_data)

    train_ = sklearn.utils.shuffle(train_)
    valid_ = sklearn.utils.shuffle(valid_)

    train_.to_csv("experiments/realtoxicity/jigsaw/train.csv", index=False)
    valid_.to_csv("experiments/realtoxicity/jigsaw/valid.csv", index=False)

    print("Done")


@torch.no_grad()
def fetch_represenation(texts, tokenizer, llm):
    with torch.no_grad():
        input_ids = tokenizer(texts, return_tensors="pt").input_ids.to(device)

        outputs = llm(input_ids, output_hidden_states=True).hidden_states[-1].to(device)

        hidden_states = outputs[:, -1, :]
        return hidden_states


@torch.no_grad()
def get_acc(args, tokenizer, llm, clf):
    file_list = [file for file in glob.glob(args.valid_folder+"*.npy")]
    Preds, Labels = [], []
    for file in tqdm(file_list, desc="test"):
        data = np.load(file)
        text, label = torch.from_numpy(data[:, :4096]).to(
            device), torch.from_numpy(np.squeeze(data[:, -1]).astype(int)).to(device)
        preds, loss = clf(text, label)
        Labels.extend(copy.deepcopy(label.cpu().detach().numpy()))
        Preds.extend(copy.deepcopy(preds))
        torch.cuda.empty_cache()
    acc = 0
    for pre, label in zip(Preds, Labels):
        if int(pre) == int(label):
            acc += 1
    return acc/len(Preds)


def load_map(file):
    return np.load(file)


def train_clf(args, tokenizer, llm, clf):
    #data = pd.read_csv(args.train_data)

    opt = torch.optim.SGD(clf.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(args.epochs):
        file_list = [file for file in glob.glob(args.train_folder+"*.npy")]
        random.shuffle(file_list)
        train_loss = []
        data = None
        idx = 0
        data_list = []
        for file in tqdm(file_list, desc="epoch {} training".format(epoch)):
            idx += 1

            data_list.append(file)
            # for (text, label) in tqdm(zip(Texts, Targets), desc="{} epoch training".format(epoch))

            """
            input = np.load(file)

            if data is None:
                data = input
            else:
                data = np.concatenate((data, input), axis=0)
            """
            if idx % args.batch_size == 0 and idx > 0:
                batch_data = list(map(load_map, data_list))
                data = np.concatenate(batch_data, axis=0)
                text, label = torch.from_numpy(data[:, :4096]).to(
                    device), torch.from_numpy(np.squeeze(data[:, -1:]).astype(int)).to(device)

                preds, loss = clf(text, label)
                # print(float(loss))
                train_loss.append(float(loss.item()))
                loss.backward()
                opt.step()
                opt.zero_grad()
                torch.cuda.empty_cache()
                data_list = list()

            """
            labels.append(label)
            feature = fetch_represenation(text, tokenizer, llm)
            if feature_embedding is None:
                feature_embedding = feature
            else:
                feature_embedding = torch.cat((feature_embedding, feature), 0)
            if idx % args.batch_size == 0 and idx > 0:
                targets = torch.from_numpy(np.array(copy.deepcopy(labels))).to(device)
                preds, loss = clf(feature_embedding, targets)
                train_loss.append(loss.item())

                loss.backward()
                opt.step()
                opt.zero_grad()

                labels = list()
                del feature_embedding
                feature_embedding = None
                torch.cuda.empty_cache()
            """

        acc = get_acc(args, tokenizer, llm, clf)
        print("epoch:{}\tacc:{}\ttrain_loss:{}".format(epoch, acc, np.mean(train_loss)))
        torch.save(clf.state_dict(), "experiments/realtoxicity/jigsaw/toxicityProbing.{}.pt".format(epoch))


class toxicityClf(nn.Module):
    def __init__(self, feature_dim):
        super(toxicityClf, self).__init__()
        self.linearProber = nn.Linear(feature_dim, 2).to(device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, features, labels):
        logits = self.linearProber(features).to(device)
        pred_softmax = torch.nn.functional.softmax(logits, dim=1)
        preds = np.argmax(pred_softmax.detach().cpu().numpy(), axis=1)
        loss = self.cross_entropy(logits, labels)

        return preds, loss.to(device)


@torch.no_grad()
def collect_data(file, target_folder):
    data = pd.read_csv(file)
    Texts, Targets = list(data["texts"]), list(data["label"])

    labels = list()
    train_loss = []
    idx = 0
    feature_embedding = None

    for (text, label) in tqdm(zip(Texts, Targets)):
        idx += 1

        labels.append(label)
        feature = fetch_represenation(text, tokenizer, llm)
        if feature_embedding is None:
            feature_embedding = feature
        else:
            feature_embedding = torch.cat((feature_embedding, feature), 0)
        if idx % args.batch_size == 0 and idx >= 0:
            targets = torch.from_numpy(np.expand_dims(np.array(copy.deepcopy(labels)), 1)).to(device)

            # print(feature_embedding.shape,targets.shape)
            data = torch.cat((feature_embedding, targets), axis=1)
            # print(data.shape)
            np.save(target_folder+str(int(idx/args.batch_size)), data.cpu().detach().numpy())
            labels = list()
            del feature_embedding
            feature_embedding = None
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="mistral")
    parser.add_argument("--cluster", type=str, default="psu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_folder", type=str, default="experiments/realtoxicity/jigsaw/train/")
    parser.add_argument("--valid_folder", type=str, default="experiments/realtoxicity/jigsaw/validation/")

    args = parser.parse_args()

    tokenizer, llm = init_model(args)
    llm.eval()

    probing_clf = toxicityClf(4096)
    train_clf(args, tokenizer, llm, probing_clf.to(device))
    #collect_data("experiments/realtoxicity/jigsaw/train.csv", "experiments/realtoxicity/jigsaw/train/")
    #collect_data("experiments/realtoxicity/jigsaw/valid.csv", "experiments/realtoxicity/jigsaw/validation/")
