"""Generuje ilustracje koncepcyjne używane w plikach .md.

Uruchomienie (z katalogu RWKV):  python assets/make_figures.py
Wykresy z wynikami eksperymentów powstają w notebookach, tutaj są tylko obrazki "do teorii".
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("zapisano", name)


def fig_embedding():
    rng = np.random.default_rng(3)
    vocab = ["a", "b", "c", "d", "e", "k", "o", "t"]
    E = rng.normal(size=(len(vocab), 6))
    idx = vocab.index("k")
    onehot = np.zeros((1, len(vocab)))
    onehot[0, idx] = 1

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), gridspec_kw={"width_ratios": [8, 6, 6]})
    axes[0].imshow(onehot, cmap="Greys", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(vocab)), vocab)
    axes[0].set_yticks([])
    axes[0].set_title("one-hot tokenu 'k'\n(wektor długości V)")
    axes[1].imshow(E, cmap="RdBu", vmin=-2.5, vmax=2.5)
    axes[1].add_patch(Rectangle((-0.5, idx - 0.5), 6, 1, fill=False, ec="black", lw=2.5))
    axes[1].set_yticks(range(len(vocab)), vocab)
    axes[1].set_xticks(range(6), [f"d{i}" for i in range(6)])
    axes[1].set_title("macierz embeddingów E\n(V × d, uczona)")
    axes[2].imshow(E[idx : idx + 1], cmap="RdBu", vmin=-2.5, vmax=2.5)
    axes[2].set_yticks([0], ["k"])
    axes[2].set_xticks(range(6), [f"d{i}" for i in range(6)])
    axes[2].set_title("one-hot · E = wiersz 'k'\n(gęsty wektor długości d)")
    save(fig, "00_embedding.png")


def fig_tanh():
    x = np.linspace(-5, 5, 400)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(x, np.tanh(x), label="tanh(x)")
    ax.plot(x, 1 - np.tanh(x) ** 2, label="tanh'(x) = 1 - tanh²(x)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.fill_between(x, 0, 1 - np.tanh(x) ** 2, where=np.abs(x) > 2, color="C3", alpha=0.15, label="saturacja: pochodna ≈ 0")
    ax.set_xlabel("pre-aktywacja")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Pochodna tanh nigdy nie przekracza 1 i szybko gaśnie")
    save(fig, "01_tanh.png")


def fig_gradient_decay():
    n = np.arange(0, 101)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for factor, label in [(0.5, "0.5"), (0.9, "0.9"), (1.0, "1.0"), (1.05, "1.05"), (1.2, "1.2")]:
        ax.semilogy(n, factor ** n, label=f"γ·σ_max(W_hh) = {label}")
    ax.axhline(1e-6, color="gray", ls=":", lw=1)
    ax.text(45, 2e-6, "praktycznie zero", color="gray", fontsize=8)
    ax.set_xlabel("odległość w czasie  T - t")
    ax.set_ylabel("skala ‖∂h_T / ∂h_t‖ ~ (γ·σ)^(T−t)")
    ax.set_ylim(1e-12, 1e8)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Gradient przez n kroków RNN ≈ (γ·σ)^n: zanika albo eksploduje wykładniczo")
    save(fig, "01_gradient_decay.png")


def fig_memory_products():
    n = np.arange(0, 501)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.semilogy(n, 0.7 ** n, "k--", label="RNN: typowe γ·σ = 0.7")
    for f in (0.9, 0.99, 0.999):
        ax.semilogy(n, f ** n, label=f"LSTM: bramka zapominania f = {f}")
    ax.set_ylim(1e-8, 2)
    ax.set_xlabel("odległość w czasie (kroki)")
    ax.set_ylabel("ile sygnału/gradientu przetrwa")
    ax.legend(fontsize=8)
    ax.set_title("Ścieżka przez stan komórki LSTM: mnożymy tylko przez f")
    save(fig, "02_memory_products.png")


def fig_attention():
    T = 12
    rng = np.random.default_rng(0)
    mask = np.tril(np.ones((T, T)))
    scores = rng.normal(size=(T, T)) * 1.5
    scores += 3.0 * np.eye(T, k=-1)
    scores[:, 0] += 1.5
    scores = np.where(mask > 0, scores, -np.inf)
    att = np.exp(scores - scores.max(1, keepdims=True))
    att /= att.sum(1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].imshow(mask, cmap="Greens", vmin=0, vmax=1.4)
    axes[0].set_title("maska przyczynowa\n(token t widzi tylko i ≤ t)")
    im = axes[1].imshow(att, cmap="viridis")
    axes[1].set_title("przykładowe wagi uwagi (softmax po wierszu)\nkażdy wiersz sumuje się do 1")
    for ax in axes:
        ax.set_xlabel("pozycja źródłowa i (klucz)")
        ax.set_ylabel("pozycja docelowa t (zapytanie)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    save(fig, "03_attention.png")


def fig_inference_cost():
    t = np.arange(1, 32769)
    n_layer, d = 32, 4096
    kv_gb = 2 * n_layer * d * 2 * t / 1e9          # K i V, fp16, model klasy 7B
    rwkv_gb = np.full_like(t, 5 * n_layer * d * 4 / 1e9, dtype=float)  # 5 wektorów na warstwę, fp32

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    axes[0].plot(t, 1 + t / 4096, label="Transformer + KV cache: O(t)")
    axes[0].plot(t, np.ones_like(t), label="RNN / RWKV: O(1)")
    axes[0].set_title("koszt wygenerowania\nJEDNEGO tokenu")
    axes[0].set_xlabel("pozycja w kontekście t")
    axes[0].set_ylabel("koszt (jednostki umowne)")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, np.cumsum(1 + t / 4096), label="Transformer: O(T²)")
    axes[1].plot(t, t, label="RNN / RWKV: O(T)")
    axes[1].set_title("łączny koszt wygenerowania\nT tokenów")
    axes[1].set_xlabel("T")
    axes[1].legend(fontsize=8)
    axes[2].semilogy(t, kv_gb, label="KV cache (32 warstwy, d=4096, fp16)")
    axes[2].semilogy(t, rwkv_gb, label="stan RWKV-4 (5·L·d liczb, fp32)")
    axes[2].set_title("pamięć na kontekst\n(model klasy 7B)")
    axes[2].set_xlabel("długość kontekstu")
    axes[2].set_ylabel("GB")
    axes[2].legend(fontsize=8)
    save(fig, "03_koszt_inferencji.png")


def fig_wkv_matrix():
    T = 14
    rng = np.random.default_rng(1)
    k = rng.normal(size=T) * 1.2
    u = 0.5
    t_idx, i_idx = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")

    def rwkv_weights(w):
        bias = np.where(i_idx < t_idx, -(t_idx - 1 - i_idx) * w, np.where(i_idx == t_idx, u, -np.inf))
        s = k[None, :] + bias
        p = np.exp(s - s.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    bias_show = np.where(i_idx < t_idx, -(t_idx - 1 - i_idx) * 0.3, np.where(i_idx == t_idx, u, np.nan))
    q = rng.normal(size=(T, 8))
    kk = rng.normal(size=(T, 8))
    sc = np.where(i_idx <= t_idx, q @ kk.T / np.sqrt(8) * 2, -np.inf)
    soft = np.exp(sc - sc.max(1, keepdims=True))
    soft /= soft.sum(1, keepdims=True)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0))
    im = axes[0].imshow(bias_show, cmap="magma")
    axes[0].set_title("bias pozycyjny (jeden kanał)\n−(t−1−i)·w pod przekątną\nu na przekątnej")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for ax, w, name in zip(axes[1:4], [3.0, 0.4, 0.02], ["szybki zanik", "średni zanik", "prawie bez zaniku"]):
        ax.imshow(rwkv_weights(w), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"RWKV-4: {name}\nw = {w}\nwagi = softmax_i(k_i + bias)")
    axes[4].imshow(soft, cmap="viridis", vmin=0, vmax=1)
    axes[4].set_title("Transformer\nwagi = softmax_i(q_t·k_i)\ndowolny wzór zależny od treści")
    for ax in axes:
        ax.set_xticks(range(0, T, 2))
        ax.set_yticks(range(0, T, 2))
        ax.set_xlabel("źródło i")
        ax.set_ylabel("cel t")
    save(fig, "04_wkv_macierz.png")


def fig_decay_spectrum():
    n_layer, d = 12, 768
    h = np.arange(d)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for layer in range(n_layer):
        r01 = layer / (n_layer - 1)
        raw = -5 + 8 * (h / (d - 1)) ** (0.7 + 1.3 * r01)
        w = np.exp(raw)
        color = plt.cm.viridis(layer / (n_layer - 1))
        axes[0].plot(h, np.exp(-w), color=color, lw=1.2, label=f"warstwa {layer}" if layer in (0, n_layer - 1) else None)
        axes[1].semilogy(h, np.log(2) / w, color=color, lw=1.2)
    axes[0].set_xlabel("kanał")
    axes[0].set_ylabel("e^(−w): ile stanu zostaje po 1 kroku")
    axes[0].set_title("Inicjalizacja zaniku w RWKV-4 (12 warstw, d=768)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("kanał")
    axes[1].set_ylabel("czas połowicznego zaniku [tokeny]")
    axes[1].set_title("Każdy kanał pamięta inaczej długo: od <1 do ~100 tokenów")
    save(fig, "04_zanik_init.png")


def fig_chunks():
    T, L = 16, 4
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for t in range(T):
        for i in range(T):
            if i > t:
                color = "#f2f2f2"
            elif t // L == i // L:
                color = "#4c72b0" if i < t else "#dd8452"
            else:
                color = "#a9c8a0"
            ax.add_patch(Rectangle((i, T - 1 - t), 1, 1, facecolor=color, edgecolor="white", lw=0.8))
    for c in range(T // L):
        ax.add_patch(Rectangle((c * L, T - (c + 1) * L), L, L, fill=False, ec="black", lw=1.6))
    for c in range(T // L - 1):
        y = T - (c + 1) * L - 0.5
        ax.add_patch(FancyArrowPatch((c * L + L - 0.2, y + 0.3), ((c + 1) * L + 0.3, y - 0.3),
                                     arrowstyle="-|>", mutation_scale=14, color="black"))
    ax.set_xlim(0, T)
    ax.set_ylim(0, T)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("źródło i  →")
    ax.set_ylabel("←  cel t")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ("#4c72b0", "#dd8452", "#a9c8a0", "#f2f2f2")]
    labels = ["w bloku: liczone jak attention (równolegle)", "przekątna: bonus u dla bieżącego tokenu",
              "przed blokiem: streszczone w stanie (a, b, p)", "przyszłość: zamaskowane"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, ncol=1, frameon=False)
    ax.set_title(f"WKV liczony blokami (T={T}, blok L={L})\nstrzałki = stan przekazywany między blokami")
    save(fig, "04_chunki.png")


if __name__ == "__main__":
    fig_embedding()
    fig_tanh()
    fig_gradient_decay()
    fig_memory_products()
    fig_attention()
    fig_inference_cost()
    fig_wkv_matrix()
    fig_decay_spectrum()
    fig_chunks()
