# Homebrew-LLM design

## 1. Goal

Build, pretrain and chat-tune my own GPT on one RTX 3090 (24 GB), writing every load-bearing
component myself, and validate each stage against a measurable gate or published reference.

## 2. Compute budget

- Peak bf16 tensor throughput (FP32 accumulate): ~71 TFLOPS.
- Planning assumption: ~28 TFLOPS sustained (40% MFU), about 2.4e18 FLOPs/day.
- Training cost estimate: 6 * N * D FLOPs (N parameters, D tokens); attention adds ~15% at context 1024.
- Beyond ~500M parameters a Chinchilla-scale run exceeds ~2 weeks, so ~350M is the ceiling.
- Long runs: GPU power limit ~300 W (about 5% slower, lower thermal risk), temperatures logged.

## 3. Targets

| | Stage 1 | Stage 2 |
|---|---|---|
| Parameters | 124M | ~350M |
| Shape | 12 layers, d=768, 12 heads | 24 layers, d=1024, 16 heads |
| Tokens | 10B | ~7B |
| Compute | ~7.4e18 FLOPs | ~1.5e19 FLOPs |
| Wall time | ~3 days | ~6 days |
| Gate | HellaSwag >= 29% | HellaSwag >= 33%, val loss below Stage 1 |

## 4. Metrics

- Validation loss (nats/token) on a fixed held-out shard.
- Bits-per-byte (bpb): loss summed over tokens / (ln 2 * UTF-8 bytes). Tokenizer-independent, so it
  compares across vocabularies and against GPT-2. Logged next to every val loss.
- HellaSwag acc_norm on the full validation set (10,042 items).
- MFU: achieved FLOPs/s over 71 TFLOPS.

## 5. Phases and gates

Status: [ ] open, [x] passed. A phase starts only after the previous gate passes.

### P0: Setup (1-2 days)
- [ ] Environment on Fedora; bf16, `torch.compile` and SDPA flash path verified.
- [ ] Throughput benchmark: one transformer block, then a full 124M forward + backward step.
- [ ] HellaSwag harness reproduces HF GPT-2 124M.
- **Gate:** block MFU >= 35%; full 124M throughput projects Stage 1 at <= 3.5 days;
  GPT-2 124M HellaSwag acc_norm 29.5% +/- 0.5.

### P1: Toy GPT (~1 week)
- [ ] Character-level GPT on TinyShakespeare: bigram -> attention -> multi-head -> blocks -> norms and residuals.
- **Gate:** val loss <= 1.5; I can explain every tensor shape.

### P2: Tokenizer (3-4 days)
- [ ] Byte-level BPE trainer (Python), 32k vocabulary, trained on a FineWeb-Edu sample.
- [ ] Special tokens reserved now: `<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`,
  `<|assistant_end|>` (chat tokens unused until P7, so the embedding never resizes).
- [ ] Fast encoder: my merges loaded into `tiktoken` as custom ranks.
- [ ] Compression compared with tiktoken GPT-2 on held-out FineWeb-Edu.
- **Gate:** lossless round-trip on arbitrary text (incl. non-ASCII and emoji); my encoder and the
  tiktoken encoder give identical ids on a held-out sample; bytes/token within 5% of GPT-2.

### P3: Data pipeline (2-3 days)
- [ ] Download FineWeb-Edu (10B sample), tokenize into uint16 shards, `<|bos|>` between documents.
- [ ] Memory-mapped loader; held-out validation shard.
- **Gate:** loader sustains more tokens/s than the model consumes; decoding a random shard slice
  reproduces the source text byte for byte.

### P4: Stage 1 pretrain, 124M (~3 days)
- [ ] GPT-2 architecture; AdamW; cosine LR with warmup; grad accumulation to ~0.5M tokens/step; clip 1.0.
- [ ] Checkpoint ~hourly; resume tested (bit-identical next-step loss) before the long run.
- [ ] HellaSwag, val loss/bpb and text samples at every checkpoint.
- **Gate:** HellaSwag >= 29%; loss curve smooth (no unrecovered spikes).

### P5: Architecture ablations (~1 week, short runs)
Each change alone at 124M for ~300M tokens (~2.3 h each), against a baseline run twice (2 seeds):
- [ ] Baseline x2 (noise band)
- [ ] RoPE
- [ ] RMSNorm
- [ ] SwiGLU (hidden = 8/3 d, parameter matched)
- [ ] QK-norm
- [ ] Grouped-query attention
- [ ] Untied embeddings
- **Gate:** keep a change if val loss improves beyond the seed spread at equal compute, or is
  loss-neutral with a measured speed or memory gain.

### P6: Stage 2 pretrain, ~350M (~6 days)
- [ ] Final architecture from P5; 24 layers, d=1024; ~7B tokens, same data mix.
- [ ] Micro-batch sized to fill 24 GB; same checkpoint and eval cadence as P4.
- **Gate:** HellaSwag >= 33%; val bpb below Stage 1.

### P7: Chat tuning (~1 week)
- [ ] Chat template with the reserved tokens; loss masked on user turns.
- [ ] SFT on a SmolTalk subset (~100k conversations).
- [ ] Fixed set of 50 prompts, blind-graded against the base model.
- **Gate:** >= 70% of the 50 prompts pass (follows the instruction and format).

### P8: Write-up
- [ ] Loss curves, ablation table, eval table, lessons learned.
- **Deliverable:** portfolio write-up alongside the Homebrew CFD solver.

## 6. Decisions

- 2026-09-30: BPE training in Python; encoding through tiktoken with my merges (pure-Python encoding
  of 10B tokens would take days to weeks). Identity of both encoders is part of the P2 gate.
- 2026-09-30: bits-per-byte logged alongside val loss so results compare across tokenizers.
- 2026-09-30: code edited on Windows, training on Fedora; repo synced through GitHub.

## 7. Known limitations

- Small model and data budget: frequent factual errors; no real coding, reasoning or long context.
- Single training run per stage; no hyperparameter sweeps beyond P5.
- Ablations are one-at-a-time at 124M and short horizon; interactions and scale effects are not measured.

## 8. Progress notes

(none yet)
