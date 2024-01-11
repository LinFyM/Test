import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from collections import namedtuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
      
class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.PositionalEncoding = PositionalEncoding(config.d_model, config.dropout, config.max_len)
        encoder_layers = TransformerEncoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout, batch_first=True)
        self.model = TransformerEncoder(encoder_layers, config.num_decoder_layers)
        self.fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tgt, tgt_mask=None, causal_mask=None, tgt_is_causal=True):
        # 通过嵌入层
        tgt = self.Embedding(tgt)
        # 通过位置编码层
        tgt = self.PositionalEncoding(tgt)
        # 通过Transformer编码器
        out = self.model(tgt, mask=causal_mask, src_key_padding_mask=tgt_mask, is_causal=tgt_is_causal)
        # 通过全连接层
        out = self.fc(out)
        # # 通过softmax层
        # out = F.softmax(out, dim=-1)
        return out

class Tokenizer:
    def __init__(self, vocab, max_len=512):
        self.vocab = json.load(open(vocab, "r", encoding="utf-8"))
        self.max_len = max_len
        # eos 0, bos 1, unk 2, pad 3

    def encode(self, text):
        chrs = text.strip().split(' ')
        ids = [self.vocab.get(chr, 2) for chr in chrs]
        ids = ids[:self.max_len - 2]
        if ids[0] != 1:
            ids = [1] + ids
        if ids[-1] != 0:
            ids = ids + [0]
        return ids

class TranslateDataset(Dataset):
    def __init__(self, tgt_file, tgt_tokenizer):
        self.tgt_file = tgt_file
        self.tgt_tokenizer = Tokenizer(tgt_tokenizer)
        self.tgt_texts = []
        self._load_data()
     
    def _load_data(self):
        with open(self.tgt_file, "r", encoding="utf-8") as f:
            self.tgt_texts = f.readlines()

    def __len__(self):
        return len(self.tgt_texts)
    
    def __getitem__(self, idx):
        tgt_text = self.tgt_texts[idx]
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)
        return tgt_ids

def make_causal_mask(attention_mask):
    """"
    0 for unmasked, -inf for masked
    """
    bsz, seq_len = attention_mask.shape
    mask = torch.ones((seq_len, seq_len)).tril()
    return mask

def collate_fn(data):
    tgt_input = data
    batch_size = len(tgt_input)
    tgt_seq_len = max([len(item) for item in tgt_input])
    tgt_mask = torch.ones((batch_size, tgt_seq_len)).long()
    tgt = torch.ones((batch_size, tgt_seq_len)).long().fill_(3)
    for i in range(batch_size):
        tgt[i, :len(tgt_input[i])] = torch.tensor(tgt_input[i])
    tgt_mask = (tgt == 3)
    causal_mask = make_causal_mask(tgt_mask)
    return tgt, tgt_mask, causal_mask

def train(model, dataloader, args, eval_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_eval_loss = float('inf')

    # model.eval()
    model.train()

    eval_loss = eval(model, eval_dataloader, args)
    print(f'Eval Loss: {eval_loss}')

    model.train()

    for epoch in range(args.epochs):

        for step, batch in enumerate(dataloader):
            tgt, tgt_mask, causal_mask = [item.to(args.device) for item in batch]

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :-1]
            causal_mask = causal_mask[:-1, :-1]

            # 前向传播
            output = model(tgt_input, tgt_mask=tgt_mask, causal_mask=causal_mask)
            # output = torch.clamp(output, min=1e-4)

            # print(output.contiguous().view(-1, output.size(-1)).shape)
            # print(tgt_output.contiguous().view(-1).shape)
            # print(output.contiguous().view(-1, output.size(-1)).max(), output.contiguous().view(-1, output.size(-1)).min())
            # print(tgt_output.contiguous().view(-1).max(), tgt_output.contiguous().view(-1).min())

            # input("Press any key to continue...")

            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.check_interval == 0:
                eval_loss = eval(model, eval_dataloader, args)

                model.train()

                print(f'Epoch {epoch+1}, Step {step}, Loss: {loss}, Eval Loss: {eval_loss}')

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    torch.save(model.state_dict(), 'best_model_weights.pth')

    return

def eval(model, dataloader, args):

    # model.eval()
    # model.train()

    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            tgt, tgt_mask, causal_mask = [item.to(args.device) for item in batch]

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :-1]
            causal_mask = causal_mask[:-1, :-1]

            output = model(tgt_input, tgt_mask=tgt_mask, causal_mask=causal_mask)
            # output = torch.clamp(output, min=1e-2)

            # print(output.contiguous().view(-1, output.size(-1)).shape)
            # print(tgt_output.contiguous().view(-1).shape)
            # print(output.contiguous().view(-1, output.size(-1)).max(), output.contiguous().view(-1, output.size(-1)).min())
            # print(tgt_output.contiguous().view(-1).max(), tgt_output.contiguous().view(-1).min())

            # input("Press any key to continue...")

            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()

        #     if step == 50:
        #         model.eval()

        # print(f"Step: {step}")

    return total_loss / len(dataloader)

def greedy_generate(model, tgt, args, eos_id):
    model.eval()
    generated = []

    with torch.no_grad():
        for i in range(args.max_len):
            output = model(tgt)
            next_word = output.argmax(dim=-1)[:, -1]
            generated.append(next_word.item())

            if next_word.item() == eos_id:
                break

            tgt = torch.cat([tgt, next_word.unsqueeze(0)], dim=-1)

    model.train()
    return generated

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_file", type=str, default="./cleaned_news/train.en")
    parser.add_argument("--tgt_eval_file", type=str, default="./cleaned_news/val.en")
    parser.add_argument("--tgt_vocab_file", type=str, default="./cleaned_news/train.en.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--check_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_len", type=int, default=5000)
    parser.add_argument('--d_model', type=int, default=8)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    config = {
        "vocab_size": len(json.load(open(args.tgt_vocab_file, "r", encoding="utf-8"))),
        "max_len": args.max_len,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    config = namedtuple("Config", config.keys())(*config.values())
    model = TransformerModel(config)
    model.to(args.device)
    train_dataset = TranslateDataset(args.tgt_file, args.tgt_vocab_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataset = TranslateDataset(args.tgt_eval_file, args.tgt_vocab_file)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    train(model, train_dataloader, args, eval_dataloader)