# Od RNN do RWKV: tutorial

Tutorial prowadzi od klasycznych sieci rekurencyjnych, przez LSTM i Transformera, do **RWKV**:
"bardzo nowoczesnego RNN", który trenuje się równolegle jak Transformer, a generuje tekst ze stałym kosztem
i stałą pamięcią jak RNN. Wszystkie modele są zaimplementowane **od zera w PyTorch** i trenowane na
"Panu Tadeuszu" na zwykłym CPU.

Zakres RWKV: **pierwsze wersje projektu (RWKV-1 → RWKV-4)**, czyli to, co opisuje paper
[*RWKV: Reinventing RNNs for the Transformer Era* (Peng i in., 2023)](https://arxiv.org/abs/2305.13048).
Późniejsze wersje (5–7) celowo pomijamy.

## Mapa tutorialu

Każdy rozdział ma dwie części: **plik `.md`** (teoria, diagramy, historia, źródła) i **notebook**
(implementacja, eksperymenty, wykresy). Najlepiej czytać najpierw `.md`, potem uruchamiać notebook.

| # | teoria | praktyka | o czym |
|---|---|---|---|
| 0 | [`00_tekst_tokeny_embeddingi.md`](00_tekst_tokeny_embeddingi.md) | [`00_tekst_tokeny_embeddingi.ipynb`](00_tekst_tokeny_embeddingi.ipynb) | jak sieć w ogóle czyta tekst: tokeny, embeddingi, model języka, model bigramowy |
| 1 | [`01_rnn.md`](01_rnn.md) | [`01_rnn.ipynb`](01_rnn.ipynb) | RNN, BPTT, znikający/eksplodujący gradient, sekwencyjność, zastosowania |
| 2 | [`02_lstm.md`](02_lstm.md) | [`02_lstm.ipynb`](02_lstm.ipynb) | bramki i stan komórki, dlaczego gradient przestaje znikać, jak się trenuje LSTM |
| 3 | [`03_transformer.md`](03_transformer.md) | [`03_transformer.ipynb`](03_transformer.ipynb) | attention, równoległy trening, KV cache, czego RNN nigdy nie dostaną |
| 4 | [`04_rwkv.md`](04_rwkv.md) | [`04_rwkv.ipynb`](04_rwkv.ipynb) | **RWKV-4 od zera**: ewolucja v1→v4, WKV na 5 sposobów, tryb GPT vs tryb RNN, stabilizacja treningu, wydajność |
| 5 | [`05_podsumowanie.md`](05_podsumowanie.md) | [`05_porownanie.ipynb`](05_porownanie.ipynb) | oś czasu, porównanie wszystkich modeli, wszystkie triki stabilizujące w jednej tabeli |

### Gdzie szukać odpowiedzi na konkretne pytania

- **Jak RNN-y mogą operować na tekście tak jak Transformery? Czy też mają embedding?**
  → [0.3 i 0.5](00_tekst_tokeny_embeddingi.md#03-embedding-z-identyfikatora-w-wektor): tak, cała "kanapka" (tokenizer, embedding,
  warstwa wyjściowa, strata) jest identyczna, różni się tylko środek.
- **Jakie problemy mają RNN-y i gdzie się je stosuje?** → [rozdział 1](01_rnn.md).
- **Gdzie pomaga LSTM i jak wygląda jego trening?** → [2.3 i 2.5](02_lstm.md#25-jak-wygląda-trening-lstm-w-praktyce).
- **Dlaczego Transformer działa tak dobrze i czego RNN nigdy nie będą miały?**
  → [3.5 i 3.7](03_transformer.md#37-czego-rnn-y-nigdy-nie-dostaną-przez-swoją-architekturę).
- **Skąd biorą się "kosmiczne" optymalizacje wydajności RWKV (wektoryzacja)?**
  → [4.6–4.8](04_rwkv.md#47-tryb-gpt-trening-wszystkie-tokeny-naraz) + część A i D notebooka 04.
- **Jak ustabilizowano i usprawniono trening takich sieci?**
  → [1.7](01_rnn.md#17-jak-łatano-trening-rnn-zanim-przyszło-lstm-i-po-nim), [2.5](02_lstm.md#25-jak-wygląda-trening-lstm-w-praktyce),
  [4.9–4.10](04_rwkv.md#410-jak-ustabilizowano-i-przyspieszono-trening) i zbiorczo [5.3](05_podsumowanie.md#53-jak-stabilizowano-trening-wszystkie-triki-w-jednym-miejscu).

## Uruchomienie

Wymagany Python 3.10+ i ~1 GB miejsca na PyTorch (wersja CPU wystarczy).

```bash
cd RWKV
uv venv .venv --python 3.12                         # albo: python -m venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

W Cursorze / VS Code otwórz notebook i wybierz kernel z `.venv` ("Select Kernel" → Python Environments → `.venv`).
Notebooki trzeba uruchamiać **po kolei** (00 → 05), bo każdy zapisuje wyniki w `results/`, a kolejne
rysują porównania. Notebooki są zapisane razem z wynikami, więc można je też po prostu czytać bez uruchamiania.

Zmierzone czasy wykonania na CPU (6 rdzeni, bez GPU): 00 ≈ 15 s, 01 ≈ 2 min, 02 ≈ 4 min, 03 ≈ 4 min, 04 ≈ 8 min,
05 kilka sekund. GPU nie jest potrzebne.

### Najważniejsze wyniki (z notebooków)

| model | parametry | strata walidacyjna (1000 kroków) | zadanie pamięciowe: najdłuższa rozwiązana odległość | koszt tokenu przy kontekście 32k |
|---|---|---|---|---|
| bigram (bez pamięci) | 3 tys. | 2,55 | – | – |
| RNN | 346 tys. | 1,79 | ~10 (niestabilnie) | stały |
| LSTM | 526 tys. | 1,76 | ~20 (bias f = 1) / 40 (bias f = 3) | stały |
| Transformer | 832 tys. | 1,86 | ≥ 160 | rośnie (~13× droższy niż przy 128) |
| **RWKV-4** | 880 tys. | **1,70** | **≥ 160** | **stały** |

To mikroskala (milion parametrów, 4 mln znaków), więc różnice w stracie nie przenoszą się 1:1 na duże modele.
Paper pokazuje, że w dużej skali RWKV jest porównywalny z Transformerami tej samej wielkości.

### Diagramy

Diagramy w plikach `.md` są w formacie [mermaid](https://mermaid.js.org/). Renderują się automatycznie na GitHubie
i w JupyterLab 4.1+. W podglądzie Markdown w Cursorze / VS Code potrzebne jest rozszerzenie
*Markdown Preview Mermaid Support* (`bierner.markdown-mermaid`). Ilustracje w `assets/` generuje
[`assets/make_figures.py`](assets/make_figures.py).

## Struktura katalogu

```
RWKV/
├── 0X_*.md / 0X_*.ipynb     teoria + praktyka dla każdego rozdziału
├── common.py                dane, tokenizer znakowy, pętla treningowa, generowanie, zadanie pamięciowe
├── assets/                  ilustracje do plików .md (+ skrypt, który je generuje)
├── data/pan_tadeusz.txt     tekst treningowy (pobierany automatycznie przy pierwszym uruchomieniu)
├── results/                 wyniki zapisywane przez notebooki (JSON), używane do porównań
├── requirements.txt
└── info                     opis tematu od prowadzącego
```

## Najważniejsze źródła

RWKV:
- [Peng i in. (2023), *RWKV: Reinventing RNNs for the Transformer Era*](https://arxiv.org/abs/2305.13048), Findings of EMNLP 2023
- [Charles Frye, *RWKV, Explained* (Full Stack Deep Learning)](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/)
- [Johan Wind, *How the RWKV language model works*](https://johanwind.github.io/2023/03/23/rwkv_details.html)
- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM): kod wszystkich wersji (v1–v4 użyte do opisu ewolucji) i README z historią trików
- [Hugging Face, *Introducing RWKV*](https://huggingface.co/blog/rwkv)
- [Zhai i in. (2021), *An Attention Free Transformer*](https://arxiv.org/abs/2105.14103), [Katharopoulos i in. (2020), *Transformers are RNNs*](https://arxiv.org/abs/2006.16236)

Tło:
- [Karpathy, *The Unreasonable Effectiveness of RNNs*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Pascanu i in. (2013), *On the difficulty of training RNNs*](https://arxiv.org/abs/1211.5063)
- [Olah, *Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Hochreiter i Schmidhuber (1997), *Long Short-Term Memory*](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Vaswani i in. (2017), *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- [Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)

Tekst "Pana Tadeusza" jest w domenie publicznej. Źródło: [Wolne Lektury](https://wolnelektury.pl/katalog/lektura/pan-tadeusz/).
