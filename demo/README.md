# Demo

A small Streamlit app with two pages for showing how an LLM's uncertainty
relates to hallucination:

| Page | What it shows | Needs a model at demo time? |
|---|---|---|
| **Entropy Lab** (`demo.py`) | Token-by-token softmax, entropy and surprisal for a prompt, with the math worked out for any step. | Only in "Live Generation" mode |
| **Circuit Tracer** (`pages/1_Circuit_Tracer.py`) | 10 HaluEval questions answered by Qwen2.5-0.5B-Instruct: every hallucination detector from the project, plus an attribution graph explaining the most suspicious tokens. | No (reads pre-computed JSON) |

## Run it

From this folder:

```bash
pip install -r requirements.txt
```
```bash
streamlit run demo.py
```

Both pages appear in the sidebar. The Circuit Tracer page needs
`circuit_examples.json` in this folder (see below).

## Files

| File | Purpose |
|---|---|
| `demo.py` | Entropy Lab page (main entry point) |
| `pre_save.py` | Generates `pre_saved_results.json` for Entropy Lab's "Pre-saved" mode |
| `pre_saved_results.json` | Pre-computed token statistics for the 5 Entropy Lab scenarios |
| `pages/1_Circuit_Tracer.py` | Circuit Tracer page |
| `build_circuit_examples.py` | Generates `circuit_examples.json` for the Circuit Tracer (GPU) |
| `circuit_examples.json` | Pre-computed examples, detector scores and attribution graphs |
| `requirements.txt` | Packages for running the app |

## Circuit Tracer

### What you see for each example

- **Prompt, reference answer and the model's answer**, coloured per token by
  surprisal and per sentence by its SelfCheck-NLI score.
- **Detector scorecard**, each with its percentile among all 24,652 answers:
  - SelfCheckGPT-NLI (B3)
  - semantic entropy and number of meaning clusters (B4)
  - semantic-entropy probe (B5)
  - naive log-probability (B6)
  - accuracy probe (B5 baseline)
- **Attribution graph** for the 3 most suspicious answer tokens: why the model
  wrote this token instead of its strongest alternative, split across the
  token embedding, every layer's MLP, and every attention head × the prompt
  token it read from. Red nodes push toward the generated token, blue toward
  the alternative. Hover a node to see its heads and the tokens it promotes.
- **Logit lens**: at which layer the model commits to the token.
- **Meaning clusters** of the 15 re-sampled answers.

The 10 examples are picked automatically from the B4–B6 results: per subset,
some hallucinations the detectors flag, some confidently correct answers, and
one confident hallucination the detectors miss.

### Rebuilding `circuit_examples.json`

It needs the GPU server, the project's models and the B4–B6 outputs
(`data/b4`, `data/b5`, `data/b6`, the labels and the 15 extra samples per
prompt). From the repository root on the server:

```bash
sbatch slurm/demo_circuits.sbatch
```

To choose the examples yourself, pass their uids:

```bash
sbatch --export=ALL,DEMO_UIDS="halueval-qa-00012 halueval-dialogue-00451" slurm/demo_circuits.sbatch
```

The log prints an "attribution reconstruction error" per example. It should
be close to 0, which confirms the contributions add up to the model's real
logit difference. Then copy `demo/circuit_examples.json` back into this folder.

### How it relates to Anthropic's circuit-tracer

The graph follows the style of Anthropic's
[circuit-tracer](https://github.com/safety-research/circuit-tracer)
attribution graphs, but it is not their method:

- circuit-tracer replaces MLPs with **transcoders** so that nodes are
  interpretable *features*. No transcoders exist for Qwen2.5-0.5B-Instruct,
  the model behind every answer in this project.
- Here the logit difference is decomposed **exactly** (final RMSNorm scale
  frozen) into model components: embedding, MLPs, and attention heads split
  by source token. Nodes are layers and heads, not features.
- Only **direct effects** on the output are shown. Indirect paths
  (component → component → output) are not traced.

## Caveats

- **Labels are provisional.** The hallucinated / faithful badge comes from an
  NLI check against the dataset references, which marks about 89% of answers
  as hallucinated. Treat it as a hint and read the reference answer yourself.
- **The model never saw the knowledge.** HaluEval provides a knowledge
  passage per question, but answers were generated from the question alone,
  so these are closed-book answers.
- **Entropy Lab's risk score is illustrative.** It is a hand-weighted
  combination of raw signals and has not been validated against labels.
