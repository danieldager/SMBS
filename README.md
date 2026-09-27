<h1 align="center">SMBS</h1>
<p align="center"><b>Speech Model Benchmarking Suite: encode audio to discrete units, train unit language models, score them on sWuggy.</b></p>
<p align="center"><img src="https://img.shields.io/badge/python-3.12%2B-3776ab" alt="Python 3.12+"> <a href="LICENSE"><img src="https://img.shields.io/badge/licence-MIT-green" alt="Licence: MIT"></a> <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status: active"></p>

A speech language model is trained on sound rather than text: the audio is converted to a sequence of discrete units, and a language model learns to predict the next unit. Whether such a model has learned word-like structure is measured with sWuggy, which presents a real word and a matched pseudo-word and asks whether the model assigns the real word the higher probability. SMBS is a command-line tool covering this whole loop. It encodes audio into units with one of three encoders, trains an LSTM or a GPT-2 on the units across several GPUs from streamed shards, and scores the result, so that every model in the lab is trained and evaluated in the same way. It was built at the Cognitive Machine Learning lab at ENS Paris.

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/pipeline-dark.svg"><img src="docs/figures/pipeline.svg" width="100%" alt="Pipeline: Audio (smbs scan) → Encoder: HuBERT-500, mHuBERT or SpidR (smbs encode) → Unit shards, WebDataset, streamed to GPUs → LSTM or GPT-2 (smbs train) → sWuggy score, real word vs pseudo-word (smbs evaluate)"></picture></p>

## Results so far

<p align="center"><img src="docs/figures/swuggy.png" width="680" alt="sWuggy accuracy by encoder and model, February to March 2026"></p>
<p align="center"><sub>sWuggy accuracy; chance is 0.5. Two rounds of the same suite: the first runs scored between 0.55 and 0.58; the best current model, a GPT-2 on SpidR units, reaches 0.66. Every point was trained and scored with the same commands.</sub></p>

## Design notes

- **One recipe per model family.** About thirty short LSTM runs remained on a loss plateau near 5.1. A smaller model with a hundred times the learning rate, gradient clipping and no weight decay converged within 40 steps. `smbs grid` compares that recipe with the published one, changing one factor at a time; the runs are documented in [docs/training_notes.md](docs/training_notes.md).
- **Streaming.** Unit shards are streamed to every GPU worker, each drawing shards at random from the corpus, so memory use does not grow with corpus size.
- **Interchangeable encoders.** Switching from HuBERT to SpidR is a command-line flag; the training and scoring code is unchanged.

## Usage

After `uv sync` and `smbs scan /path/to/audio` (which writes the file manifest), each step is one command. Each submits a SLURM job; add `--local` to run it on the current machine.

```bash
# audio → unit shards
smbs encode --encoder spidr_base --dataset chunks30
# train a unit language model (or --arch lstm)
smbs train --encoder spidr_base --arch gpt2
# encode the sWuggy audio, once per encoder
smbs prepare-swuggy --encoder spidr_base --parquet-pattern '/path/to/swuggy/*.parquet'
# score a trained model
smbs evaluate --encoder spidr_base --model gpt2_e768_l12_h12_feb12
```

The evaluation ends with a summary in this format (value shown: GPT-2 on SpidR units, from the March results, to three decimals):

```
  Normalized: 0.658
```

Raw and length-normalised accuracy are both printed, with a per-voice breakdown. Full options, the evaluation schema for other lexical benchmarks, and the project layout are in [docs/USAGE.md](docs/USAGE.md).

## Scope and credit

- Training audio is LibriVox audiobooks, public domain. sWuggy follows the Zero Resource Speech Benchmark definition.
- Encoders: HuBERT and mHuBERT (Meta), SpidR (Meta). GPT-2 via Hugging Face.
- [DL++](https://github.com/danieldager/DLplusplus) produces training shards from daylong child recordings in the same format.

Issues and pull requests are welcome.
