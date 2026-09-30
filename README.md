# Homebrew-LLM

A GPT language model built, pretrained and chat-tuned from scratch on a single RTX 3090 (24 GB).

I write the model, tokenizer, data pipeline and training code myself, following a validation-first
approach: every stage ends with a measurable gate that must pass before the next one starts.

## Targets

| | Stage 1 (validation) | Stage 2 (final) |
|---|---|---|
| Parameters | 124M (GPT-2 small) | ~350M (GPT-2 medium class) |
| Shape | 12 layers, d=768, 12 heads | 24 layers, d=1024, 16 heads |
| Training tokens | 10B | ~7B (~20 tokens/parameter) |
| Compute | ~7.4e18 FLOPs | ~1.5e19 FLOPs |
| Wall time | ~3 days | ~6 days |
| Gate | HellaSwag >= 29% | HellaSwag >= 33%, val loss below Stage 1 |

After pretraining, the Stage 2 model is supervised fine-tuned into a simple chat model. Its sampler
embeds a keyed statistical watermark (hash-seeded green-list logit bias), and a standalone detector
identifies watermarked text from the key and tokenizer alone.

## Scope and expectations

The final model should produce fluent short text and follow simple instructions, with frequent
factual errors. It is not intended for coding, reasoning or long documents. The purpose of the
project is to understand and validate the full stack: tokenizer, data, architecture, pretraining,
evaluation and chat tuning.

## Stack

- Python, PyTorch + CUDA, bf16, `torch.compile`, `F.scaled_dot_product_attention`
- Byte-level BPE tokenizer (32k vocabulary) trained on FineWeb-Edu; encoding via `tiktoken` with my merges
- Data: FineWeb-Edu 10B-token sample
- Evaluation: HellaSwag (acc_norm), validation loss and bits-per-byte

## Repository layout

```
model/    gpt.py, tokenizer.py, watermark.py
train/    pretrain.py, sft.py, data.py
eval/     hellaswag.py, sample.py, detect_watermark.py
tests/    pytest suite
docs/     DESIGN.md (plan, gates, decisions, limitations), run_log.csv
```

## Status

See `docs/DESIGN.md` for phases, gates and progress, and `docs/run_log.csv` for run history.

## References

- A. Karpathy: "Let's build GPT", build-nanogpt, llm.c, minbpe, nanochat
- J. Kirchenbauer et al., "A Watermark for Large Language Models" (ICML 2023)
- FineWeb-Edu (HuggingFace), SmolTalk (HuggingFace), HellaSwag (Zellers et al., 2019)
