import argparse
import csv
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
REPORT_DIR = ROOT / "report"
for directory in (RESULTS_DIR, PLOTS_DIR, REPORT_DIR):
    directory.mkdir(exist_ok=True)

SEED = 4106
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHAR_TEXT = """Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."""

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def causal_mask(length, device):
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


class CharWindowDataset(Dataset):
    def __init__(self, encoded, block_size):
        self.encoded = torch.tensor(encoded, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.encoded) - self.block_size)

    def __getitem__(self, idx):
        x = self.encoded[idx : idx + self.block_size]
        y = self.encoded[idx + 1 : idx + self.block_size + 1]
        return x, y


class CharTransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=64, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        bsz, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        h = self.token_embedding(x) + self.position_embedding(positions)
        h = self.blocks(h, mask=causal_mask(seq_len, x.device))
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


def char_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def evaluate_char_model(model, loader):
    model.eval()
    total_loss = total_correct = total_top3 = total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            preds = logits.argmax(-1)
            top3 = logits.topk(min(3, logits.size(-1)), dim=-1).indices
            total_loss += loss.item() * y.numel()
            total_correct += preds.eq(y).sum().item()
            total_top3 += top3.eq(y.unsqueeze(-1)).any(-1).sum().item()
            total_tokens += y.numel()
    loss = total_loss / max(1, total_tokens)
    return {
        "valid_loss": loss,
        "valid_accuracy": total_correct / max(1, total_tokens),
        "valid_top3_accuracy": total_top3 / max(1, total_tokens),
        "valid_perplexity": math.exp(min(20, loss)),
    }


def train_char_run(text, sequence_length, experiment, num_layers=2, nhead=2, d_model=64, epochs=8, max_chars=None):
    set_seed(SEED + sequence_length + num_layers * 11 + nhead * 17)
    if max_chars:
        text = text[:max_chars]
    stoi, itos = char_vocab(text)
    encoded = [stoi[ch] for ch in text]
    cutoff = int(0.8 * len(encoded))
    train_data = CharWindowDataset(encoded[:cutoff], sequence_length)
    valid_data = CharWindowDataset(encoded[max(0, cutoff - sequence_length) :], sequence_length)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)
    model = CharTransformerLM(len(stoi), sequence_length, d_model=d_model, nhead=nhead, num_layers=num_layers).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_tokens = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        metrics = evaluate_char_model(model, valid_loader)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_tokens), **metrics}
        history.append(row)
    train_seconds = time.perf_counter() - start
    infer_x = next(iter(valid_loader))[0][:1].to(DEVICE)
    infer_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model(infer_x)
    infer_seconds = (time.perf_counter() - infer_start) / 20
    seed = torch.tensor([[stoi.get("N", 0)]], dtype=torch.long, device=DEVICE)
    generated = "".join(itos[int(i)] for i in model.generate(seed, 180)[0].cpu())
    final = history[-1].copy()
    final.update({
        "experiment": experiment,
        "sequence_length": sequence_length,
        "num_layers": num_layers,
        "num_heads": nhead,
        "d_model": d_model,
        "parameter_count": count_parameters(model),
        "approx_attention_ops": num_layers * (sequence_length * sequence_length * d_model + sequence_length * d_model * d_model * 12),
        "train_seconds": train_seconds,
        "inference_seconds_per_sample": infer_seconds,
        "vocab_size": len(stoi),
        "generated_output": generated,
    })
    hist_path = RESULTS_DIR / f"{experiment}_history_seq{sequence_length}_L{num_layers}_H{nhead}.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    return final


def normalize_text(text):
    text = unicodedata.normalize("NFD", text.lower().strip())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"([.!?;,])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?;,']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_pairs(path):
    pairs = []
    with Path(path).open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            left, right = raw.rstrip("\n").split("\t")[:2]
            pairs.append((normalize_text(left), normalize_text(right)))
    return pairs


def split_pairs_hw3(pairs):
    split_path = RESULTS_DIR / "hw3_split_indices.json"
    if split_path.exists():
        data = json.loads(split_path.read_text())
        train_idx = data.get("train_idx") or data.get("train_indices")
        val_idx = data.get("val_idx") or data.get("validation_idx") or data.get("validation_indices") or data.get("valid_idx")
        if train_idx is not None and val_idx is not None:
            return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]
    rng = random.Random(SEED)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]


class Vocab:
    def __init__(self, texts, min_freq=1, max_size=None):
        counts = Counter()
        for text in texts:
            counts.update(text.split())
        words = [w for w, c in counts.items() if c >= min_freq]
        words.sort(key=lambda w: (-counts[w], w))
        if max_size:
            words = words[: max(0, max_size - len(SPECIALS))]
        self.itos = SPECIALS + words
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        self.pad_idx = self.stoi[PAD]
        self.sos_idx = self.stoi[SOS]
        self.eos_idx = self.stoi[EOS]
        self.unk_idx = self.stoi[UNK]

    def encode(self, text):
        return [self.sos_idx] + [self.stoi.get(tok, self.unk_idx) for tok in text.split()] + [self.eos_idx]

    def decode(self, ids):
        words = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.sos_idx):
                continue
            words.append(self.itos[idx] if idx < len(self.itos) else UNK)
        return " ".join(words)

    def __len__(self):
        return len(self.itos)


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.rows = []
        for src, tgt in pairs:
            self.rows.append((torch.tensor(src_vocab.encode(src)), torch.tensor(tgt_vocab.encode(tgt)), src, tgt))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_translation(rows):
    src, tgt, src_text, tgt_text = zip(*rows)
    src_pad = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src_pad.long(), tgt_pad.long(), list(src_text), list(tgt_text)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, nhead=2, num_layers=2, dropout=0.1, max_len=80):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.src_pos = nn.Embedding(max_len, d_model)
        self.tgt_pos = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def add_pos(self, emb, table):
        positions = torch.arange(emb.size(1), device=emb.device).unsqueeze(0).expand(emb.size(0), emb.size(1))
        return emb * math.sqrt(self.d_model) + table(positions.clamp_max(table.num_embeddings - 1))

    def forward(self, src, tgt_in):
        src_key_padding_mask = src.eq(0)
        tgt_key_padding_mask = tgt_in.eq(0)
        src_emb = self.add_pos(self.src_embedding(src), self.src_pos)
        tgt_emb = self.add_pos(self.tgt_embedding(tgt_in), self.tgt_pos)
        tgt_mask = causal_mask(tgt_in.size(1), tgt_in.device)
        h = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(h)

    @torch.no_grad()
    def generate(self, src, sos_idx, eos_idx, max_len=30):
        self.eval()
        ys = torch.full((src.size(0), 1), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            logits = self(src, ys)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if bool(next_token.eq(eos_idx).all()):
                break
        return ys[:, 1:]


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference, hypothesis, max_n=4):
    ref = reference.split()
    hyp = hypothesis.split()
    if not hyp:
        return 0.0
    values = []
    for n in range(1, max_n + 1):
        hyp_counts = ngram_counts(hyp, n)
        ref_counts = ngram_counts(ref, n)
        total = max(1, sum(hyp_counts.values()))
        overlap = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        values.append((overlap + 1.0) / (total + 1.0))
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(hyp)))
    return bp * math.exp(sum(math.log(v) for v in values) / max_n)


def corpus_bleu(references, hypotheses, max_n=4):
    if not hypotheses:
        return 0.0
    return sum(sentence_bleu(r, h, max_n=max_n) for r, h in zip(references, hypotheses)) / len(hypotheses)


def sequence_accuracy(references, hypotheses):
    if not references:
        return 0.0
    return sum(r.strip() == h.strip() for r, h in zip(references, hypotheses)) / len(references)


def evaluate_translation(model, loader, tgt_vocab):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = total_tokens = 0
    references, hypotheses, sources = [], [], []
    with torch.no_grad():
        for src, tgt, src_text, tgt_text in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            tokens = tgt[:, 1:].ne(0).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            pred = model.generate(src, tgt_vocab.sos_idx, tgt_vocab.eos_idx, max_len=30)
            hypotheses.extend(tgt_vocab.decode(row) for row in pred.cpu())
            references.extend(tgt_text)
            sources.extend(src_text)
    loss = total_loss / max(1, total_tokens)
    return {
        "valid_loss": loss,
        "valid_perplexity": math.exp(min(20, loss)),
        "sequence_accuracy": sequence_accuracy(references, hypotheses),
        "bleu4": corpus_bleu(references, hypotheses),
        "sources": sources,
        "references": references,
        "hypotheses": hypotheses,
    }


def evaluate_translation_loss(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = total_tokens = 0
    with torch.no_grad():
        for src, tgt, _, _ in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            tokens = tgt[:, 1:].ne(0).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    loss = total_loss / max(1, total_tokens)
    return {"valid_loss": loss, "valid_perplexity": math.exp(min(20, loss))}


def train_translation_run(direction, num_layers, nhead, epochs=8, d_model=64):
    set_seed(SEED + num_layers * 23 + nhead * 31 + (0 if direction == "en_fr" else 1000))
    pairs = load_pairs(DATA_DIR / "vast_english_french.txt")
    train_pairs, valid_pairs = split_pairs_hw3(pairs)
    if direction == "fr_en":
        train_pairs = [(fr, en) for en, fr in train_pairs]
        valid_pairs = [(fr, en) for en, fr in valid_pairs]
        direction_label = "French-to-English"
    else:
        direction_label = "English-to-French"
    src_vocab = Vocab([src for src, _ in train_pairs])
    tgt_vocab = Vocab([tgt for _, tgt in train_pairs])
    train_loader = DataLoader(TranslationDataset(train_pairs, src_vocab, tgt_vocab), batch_size=32, shuffle=True, collate_fn=collate_translation)
    valid_loader = DataLoader(TranslationDataset(valid_pairs, src_vocab, tgt_vocab), batch_size=64, shuffle=False, collate_fn=collate_translation)
    model = Seq2SeqTransformer(len(src_vocab), len(tgt_vocab), d_model=d_model, nhead=nhead, num_layers=num_layers).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_tokens = 0
        for src, tgt, _, _ in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = tgt[:, 1:].ne(0).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
        metrics = evaluate_translation_loss(model, valid_loader)
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, total_tokens), **metrics})
    seconds = time.perf_counter() - start
    final_metrics = evaluate_translation(model, valid_loader, tgt_vocab)
    sample_rows = []
    for src, ref, hyp in list(zip(final_metrics["sources"], final_metrics["references"], final_metrics["hypotheses"]))[:8]:
        sample_rows.append({
            "direction": direction_label,
            "num_layers": num_layers,
            "num_heads": nhead,
            "source": src,
            "reference": ref,
            "prediction": hyp,
            "exact_match": ref.strip() == hyp.strip(),
            "sentence_bleu4": sentence_bleu(ref, hyp),
        })
    prefix = f"problem{'3' if direction == 'en_fr' else '4'}_{direction}_L{num_layers}_H{nhead}"
    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{prefix}_history.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(RESULTS_DIR / f"{prefix}_samples.csv", index=False)
    src_len = 12
    tgt_len = 12
    return {
        "direction": direction_label,
        "num_layers": num_layers,
        "num_heads": nhead,
        "d_model": d_model,
        "train_loss": history[-1]["train_loss"],
        "valid_loss": final_metrics["valid_loss"],
        "valid_perplexity": final_metrics["valid_perplexity"],
        "sequence_accuracy": final_metrics["sequence_accuracy"],
        "bleu4": final_metrics["bleu4"],
        "parameter_count": count_parameters(model),
        "approx_attention_ops": num_layers * (src_len * src_len * d_model + tgt_len * tgt_len * d_model + src_len * tgt_len * d_model + 24 * src_len * d_model * d_model),
        "train_seconds": seconds,
        "source_vocab": len(src_vocab),
        "target_vocab": len(tgt_vocab),
    }


def plot_histories(pattern, title, out_name):
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for file in files:
        df = pd.read_csv(file)
        label = file.stem.replace("_history", "").replace("problem", "p")
        ax.plot(df["epoch"], df["train_loss"], linestyle="-", label=f"{label} train")
        if "valid_loss" in df:
            ax.plot(df["epoch"], df["valid_loss"], linestyle="--", label=f"{label} val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=180)
    plt.close(fig)


def bar_plot(csv_name, x_col, y_col, title, out_name):
    df = pd.read_csv(RESULTS_DIR / csv_name)
    labels = [f"L{int(r.num_layers)} H{int(r.num_heads)}" if "num_layers" in df.columns else str(getattr(r, x_col)) for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, df[y_col])
    ax.set_title(title)
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=180)
    plt.close(fig)


def run_all():
    print(f"Using device: {DEVICE}")
    p1_rows = [train_char_run(CHAR_TEXT, seq, "problem1_transformer", epochs=30, d_model=64, num_layers=2, nhead=2) for seq in (10, 20, 30)]
    pd.DataFrame(p1_rows).to_csv(RESULTS_DIR / "problem1_transformer_summary.csv", index=False)

    tiny = (DATA_DIR / "tiny_shakespeare.txt").read_text(encoding="utf-8")
    p2_rows = []
    for seq in (20, 30):
        p2_rows.append(train_char_run(tiny, seq, "problem2_baseline", epochs=5, max_chars=10000, d_model=64, num_layers=2, nhead=2))
    for layers in (1, 2, 4):
        for heads in (2, 4):
            p2_rows.append(train_char_run(tiny, 30, "problem2_hyper", epochs=5, max_chars=10000, d_model=64, num_layers=layers, nhead=heads))
    p2_rows.append(train_char_run(tiny, 50, "problem2_seq50", epochs=5, max_chars=10000, d_model=64, num_layers=2, nhead=2))
    pd.DataFrame(p2_rows).drop_duplicates(["experiment", "sequence_length", "num_layers", "num_heads"]).to_csv(RESULTS_DIR / "problem2_transformer_summary.csv", index=False)

    p3_rows = [train_translation_run("en_fr", layers, heads, epochs=35, d_model=64) for layers in (1, 2, 4) for heads in (2, 4)]
    pd.DataFrame(p3_rows).to_csv(RESULTS_DIR / "problem3_transformer_summary.csv", index=False)
    p4_rows = [train_translation_run("fr_en", layers, heads, epochs=35, d_model=64) for layers in (1, 2, 4) for heads in (2, 4)]
    pd.DataFrame(p4_rows).to_csv(RESULTS_DIR / "problem4_transformer_summary.csv", index=False)

    plot_histories("problem1_transformer_history*.csv", "Problem 1 Transformer Loss Curves", "problem1_transformer_loss.png")
    plot_histories("problem2_*history*.csv", "Problem 2 Transformer Loss Curves", "problem2_transformer_loss.png")
    plot_histories("problem3_*history*.csv", "Problem 3 English-to-French Loss Curves", "problem3_transformer_loss.png")
    plot_histories("problem4_*history*.csv", "Problem 4 French-to-English Loss Curves", "problem4_transformer_loss.png")
    bar_plot("problem3_transformer_summary.csv", "config", "bleu4", "Problem 3 BLEU-4 by Transformer Configuration", "problem3_bleu.png")
    bar_plot("problem4_transformer_summary.csv", "config", "bleu4", "Problem 4 BLEU-4 by Transformer Configuration", "problem4_bleu.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()
    if args.run_all:
        run_all()


if __name__ == "__main__":
    main()
