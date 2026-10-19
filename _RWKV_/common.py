"""Wspólne narzędzia dla notebooków tutorialu.

Wszystko, co NIE jest tematem danego rozdziału (pobieranie danych, tokenizer znakowy,
pętla treningowa, generowanie tekstu, zapis wyników), siedzi tutaj, żeby notebooki
mogły skupić się na samych architekturach: RNN, LSTM, Transformer, RWKV.

Każdy model w tutorialu ma ten sam interfejs:  model(idx) -> logits
    idx:    LongTensor (B, T)     - indeksy tokenów
    logits: FloatTensor (B, T, V) - nieznormalizowane log-prawdopodobieństwa następnego tokenu
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"

PAN_TADEUSZ_URL = "https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt"


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------------------
# Dane: "Pan Tadeusz" (domena publiczna, Wolne Lektury) + tokenizer znakowy
# --------------------------------------------------------------------------------------

def clean_wolne_lektury(raw: str) -> str:
    """Wycina metryczkę i stopkę licencyjną, normalizuje końce linii i wcięcia."""
    raw = raw.replace("\r\n", "\n")
    text = raw[raw.index("Księga pierwsza") : raw.index("\n-----\n")]
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def load_pan_tadeusz() -> str:
    path = DATA_DIR / "pan_tadeusz.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        raw = urllib.request.urlopen(PAN_TADEUSZ_URL, timeout=60).read().decode("utf-8")
        path.write_text(clean_wolne_lektury(raw), encoding="utf-8")
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    """Najprostszy możliwy tokenizer: jeden znak = jeden token."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def prepare_data(val_fraction: float = 0.1):
    """Zwraca (tokenizer, dane_treningowe, dane_walidacyjne). Walidacja = końcówka książki."""
    text = load_pan_tadeusz()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return tok, data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, generator=None):
    """Losowe okna długości block_size; cel y to wejście x przesunięte o jeden token w prawo."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# --------------------------------------------------------------------------------------
# Trening
# --------------------------------------------------------------------------------------

@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 32
    block_size: int = 128
    lr: float = 2e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 100
    betas: tuple = (0.9, 0.99)
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    eval_every: int = 100
    eval_iters: int = 16
    seed: int = 1337


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Liniowa rozgrzewka, potem cosinusowe wygaszanie do min_lr_frac * lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * cosine)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, cfg: TrainConfig) -> float:
    """Średnia strata na stałym zestawie batchy (ten sam seed = porównywalne liczby między modelami)."""
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(cfg.seed + 1)
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.block_size, cfg.batch_size, g)
        losses.append(lm_loss(model(x), y).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def train(model, train_data, val_data, cfg: TrainConfig, loss_fn=lm_loss, log: bool = True) -> dict:
    """Pętla treningowa wspólna dla wszystkich modeli: AdamW + warmup/cosinus + przycinanie gradientu.

    Zwraca słownik z historią (strata, norma gradientu, czas), gotowy do wykresów i zapisu do JSON.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed)
    hist = {"train_loss": [], "grad_norm": [], "eval_step": [], "val_loss": [], "eval_time": []}
    clip = cfg.grad_clip if cfg.grad_clip is not None else float("inf")
    train_time = 0.0
    model.train()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            val = estimate_loss(model, val_data, cfg)
            hist["eval_step"].append(step)
            hist["val_loss"].append(val)
            hist["eval_time"].append(train_time)
            if log:
                recent = hist["train_loss"][-cfg.eval_every:]
                tr = f"{sum(recent) / len(recent):.3f}" if recent else "  -  "
                print(f"krok {step:5d} | train {tr} | val {val:.3f} | {train_time:6.1f}s")
        if step == cfg.steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        t0 = time.perf_counter()
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, g)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        train_time += time.perf_counter() - t0
        hist["train_loss"].append(loss.item())
        hist["grad_norm"].append(grad_norm.item())
    hist["train_time"] = train_time
    hist["sec_per_step"] = train_time / max(1, cfg.steps)
    hist["config"] = asdict(cfg)
    hist["n_params"] = count_params(model)
    return hist


# --------------------------------------------------------------------------------------
# Generowanie tekstu
# --------------------------------------------------------------------------------------

def sample_from_logits(logits: torch.Tensor, temperature: float = 0.8, top_k: int | None = None,
                       generator=None) -> torch.Tensor:
    """logits: (B, V) -> indeksy (B, 1). temperature<1 = bardziej zachowawczo, >1 = bardziej losowo."""
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate_naive(model, tok: CharTokenizer, prompt: str, max_new_tokens: int = 300,
                   block_size: int = 128, temperature: float = 0.8, top_k: int | None = None,
                   seed: int = 0) -> str:
    """Generowanie 'na siłę': przy każdym nowym tokenie przeliczamy CAŁY kontekst (ostatnie block_size).

    Działa dla każdego modelu z interfejsem model(idx) -> logits. Rozdziały 1-4 pokazują,
    jak każda architektura robi to sprytniej (stan RNN, KV cache, tryb RNN w RWKV).
    """
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])[:, -1, :]
        idx = torch.cat([idx, sample_from_logits(logits, temperature, top_k, g)], dim=1)
    model.train(was_training)
    return tok.decode(idx[0].tolist())


# --------------------------------------------------------------------------------------
# Zadanie pamięciowe: czy model pamięta pierwszy token po L krokach szumu?
# --------------------------------------------------------------------------------------

MEMORY_N_KEYS = 8
MEMORY_N_NOISE = 8
MEMORY_VOCAB = 1 + MEMORY_N_KEYS + MEMORY_N_NOISE  # 0 = znak zapytania


def memory_task_batch(batch_size: int, seq_len: int, generator=None):
    """Sekwencja [KLUCZ, szum, szum, ..., szum, ?]; poprawna odpowiedź na pozycji '?' to KLUCZ.

    Tokeny: 0 = '?', 1..8 = klucze, 9..16 = szum. Losowa odpowiedź trafia w 1/8 = 12.5% przypadków.
    """
    keys = torch.randint(1, 1 + MEMORY_N_KEYS, (batch_size, 1), generator=generator)
    noise = torch.randint(1 + MEMORY_N_KEYS, MEMORY_VOCAB, (batch_size, seq_len - 2), generator=generator)
    query = torch.zeros(batch_size, 1, dtype=torch.long)
    return torch.cat([keys, noise, query], dim=1), keys.squeeze(1)


def train_memory_task(model, seq_len: int, max_steps: int = 1500, batch_size: int = 64, lr: float = 3e-3,
                      grad_clip: float = 1.0, seed: int = 0, solved_acc: float = 0.99) -> dict:
    """Uczy model odpowiadać kluczem na ostatniej pozycji.

    Każdy model dostaje ten sam limit kroków; trening kończy się wcześniej, gdy średnia trafność
    z ostatnich 50 batchy przekroczy solved_acc (zadanie rozwiązane). Wynik końcowy liczymy na 1000 nowych przykładach.
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    accs = []
    model.train()
    for _ in range(max_steps):
        x, y = memory_task_batch(batch_size, seq_len, g)
        logits = model(x)[:, -1, :]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        accs.append((logits.argmax(-1) == y).float().mean().item())
        if len(accs) >= 50 and sum(accs[-50:]) / 50 >= solved_acc:
            break
    model.eval()
    with torch.no_grad():
        x, y = memory_task_batch(1000, seq_len, torch.Generator().manual_seed(seed + 123))
        final_acc = (model(x)[:, -1, :].argmax(-1) == y).float().mean().item()
    return {"seq_len": seq_len, "train_acc": accs, "steps": len(accs), "final_acc": final_acc}


def run_memory_experiment(make_model, lengths, name: str, seeds=(0, 1), max_steps: int = 1000,
                          lr: float = 3e-3) -> dict:
    """Zadanie pamięciowe dla kilku odległości i kilku seedów (świeży model za każdym razem)."""
    out = {"name": name, "lengths": list(lengths), "per_seed": [], "steps": [], "final_acc": []}
    for L in lengths:
        accs, steps = [], []
        t0 = time.perf_counter()
        for seed in seeds:
            set_seed(seed)
            r = train_memory_task(make_model(), L, max_steps=max_steps, lr=lr, seed=seed)
            accs.append(r["final_acc"])
            steps.append(r["steps"])
        out["per_seed"].append(accs)
        out["steps"].append(steps)
        out["final_acc"].append(sum(accs) / len(accs))
        print(f"L = {L:4d}: trafność {' / '.join(f'{a:4.0%}' for a in accs)}"
              f"   kroki treningu: {' / '.join(f'{s:4d}' for s in steps)}   ({time.perf_counter() - t0:.0f} s)")
    return out


def plot_memory(results: list, ax=None):
    """Średnia trafność (linia) i wyniki poszczególnych seedów (kropki) w funkcji odległości."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.8))
    for i, r in enumerate(r for r in results if r):
        color = f"C{i}"
        ax.plot(r["lengths"], r["final_acc"], "o-", color=color, label=r["name"])
        for L, accs in zip(r["lengths"], r.get("per_seed", [])):
            ax.scatter([L] * len(accs), accs, color=color, s=12, alpha=0.4)
    ax.axhline(1 / MEMORY_N_KEYS, color="gray", ls="--", label="zgadywanie (1/8)")
    ax.set_xscale("log", base=2)
    all_lengths = sorted({L for r in results if r for L in r["lengths"]})
    ax.set_xticks(all_lengths, all_lengths)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("odległość klucz → pytanie (L)")
    ax.set_ylabel("trafność")
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------------------
# Pomiary czasu i zapis wyników
# --------------------------------------------------------------------------------------

def time_fn(fn, repeats: int = 10, warmup: int = 2) -> float:
    """Mediana czasu wywołania fn() w sekundach."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def save_results(name: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_results(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
