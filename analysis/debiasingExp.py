import os

import pytorch_lightning as pl
import transformers

from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    DataCollatorForLanguageModeling,
    AutoConfig
)
import torch
from torch.utils.data import DataLoader, Dataset

transformers.logging.set_verbosity_error()


class myMaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer):
        self.tokenizer = tokenizer
        self.lines = self.load_lines(file)
        self.ids = self.encode_lines(self.lines)

    def load_lines(self, file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines

    def encode_lines(self, lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, padding='max_length'
        )
        return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)


class myCausal(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.load(args)

    def load(self, args):
        cache_dir = ""
        cache_dir_slim = "/home/zhangxit/files/llms"
        cache_dir_psu = "/data/bochuan/DPO/cache"
        cache_dir_voodoo = "/scratch0/liuguan5/pretrained_models"
        if args.cluster == "psu":
            cache_dir = cache_dir_psu
        elif args.cluster == "slim":
            cache_dir = cache_dir_slim
        elif args.cluster == "voodoo":
            cache_dir = cache_dir_voodoo
            self.config = AutoConfig.from_pretrained(
                'openai-community/gpt2-xl', output_hidden_states=True, output_attentions=True)
            self.causal = AutoModelForCausalLM.from_pretrained(
                'openai-community/gpt2-xl', config=self.config, cache_dir=cache_dir)

    def forward(self, input_ids):
        return self.causal(input_ids=input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        outputs = self.forward(input_ids=input_ids)
        loss = outputs[0]
        return {"loss": loss.sum()}

    def valid_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        outputs = self.forward(input_ids=input_ids, )
        loss = outputs[0]
        self.log('val_loss', loss.sum())

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=args.lr)


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='gpt2',
                        help="Full name or path or URL to trained NLI model")  # bert-base-uncased

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--mlm_prob", type=float, default=0.15, help="Probability to mask random tokens in the input")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--checkpoint_callback", action="store_true", help="Checkpoint callback?")
    parser.add_argument("--logger", action="store_true", help="Do log?")
    parser.add_argument("--pruning", type=bool, default=False)
    parser.add_argument("--cluster", type=str, default="psu")
    parser.add_argument("--llm", type=str, default="gpt2")
    parser.add_argument("--save_path", type=str, default="experiments/debiasing/")
    parser.add_argument("--debiasing_source", type=str, default="lessuncertainty.debiasing.txt")
    args = parser.parse_args()

    if args.cluster == "voodoo":
        os.system("export TRANSFORMERS_CACHE=/scratch0/liuguan5/pretrained_models/")
    
    # print('into debiasing')
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-xl')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = myMaskedLMDataset(args.save_path+args.debiasing_source, tokenizer)
    # valid_dataset = myMaskedLMDataset(args.valid_data, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=8
    )

    model = myCausal(args)

    for i in range(args.epochs):

        if i > 0:
            model.load_state_dict(torch.load(
                f"{args.save_path}{args.debiasing_source}.{i-1}.ckpt", map_location='cuda'), strict=False)

        log_dir = args.save_path + str(i) + ".log"

        checkpoint_callback = ModelCheckpoint(
            dirpath=log_dir,
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            mode="min")

        trainer = pl.Trainer(max_epochs=args.epochs, devices=[0, 1, 2, 3, 4, 5, 6, 7], accelerator="gpu", strategy="ddp",
                             callbacks=[checkpoint_callback],
                             default_root_dir=args.save_path)
        trainer.fit(model, train_loader)
        torch.save(model.causal.state_dict(),
                   f"{args.save_path}{args.debiasing_source}.{i}.ckpt")
