# Prompt Repetition for LLM Retrieval Tasks

Reproducing findings from [Repeat to Recall: Prompt Repetition Improves LLM Recall of Long Contexts](https://arxiv.org/html/2512.14982v1) using Google Vertex AI.

## TL;DR

**Expected answer: Paul Sanchez**

| Condition | Model Response | Correct |
|---|---|---|
| No Repetition | Samuel Curtis (26th name, off by 1) | No |
| Prefill (Counting) | Counts 1-25 correctly, returns Paul Sanchez | Yes |
| Vanilla Repetition | Paul Sanchez | Yes |

Above results are for `gemini-2.0-flash-lite`.  The new version `gemini-3-flash-preview` fixed the repetition bug.

### Prefill Counting Output (intermediate steps)

Without explicit counting, the model returns **Samuel Curtis** — the 26th name. The prefill condition forces the model to count step-by-step, revealing that it *can* map positions correctly when tracking is made explicit:

```
No Repetition (implicit counting):

  "The 25th name on the list is Samuel Curtis."
  Samuel Curtis is actually #26 — off by one.

Prefill Counting (explicit counting):

  1. Dale Lopez
  2. Peter Sanchez
  3. Allen Harris        <-- prefilled by us
  4. Scott Davis
  5. Hudson Leviathan
  ...                    <-- model continues counting
  23. James Harris
  24. Craig Chavez
  25. **Paul Sanchez**   <-- correct
  26. Samuel Curtis

  The 25th name is Paul Sanchez.
```

The off-by-one error in the No Repetition condition shows that the model's implicit positional encoding is imprecise for items in the middle of the list. When forced to count explicitly (Prefill) or given a second pass at the context (Repetition), the model retrieves the correct answer.

## Background

The paper demonstrates that simply repeating a prompt improves LLM accuracy on retrieval tasks, particularly for information located in the middle of long contexts — the "lost in the middle" phenomenon. This effect is especially pronounced on smaller/lighter models.

## Why It Works

*Insight from [Cris Benge](https://github.com/cbenge509):*

During prefill encoding, the model processes tokens left-to-right. On the first pass, it encounters the list of names **before** it reaches the question "What's the 25th name?". Without knowing what will be asked, the encoder cannot attend specifically to the 25th item — it encodes the list generically, tracking relationships like "this name comes between item Y and item Z" rather than precise positional indices.

When the question finally arrives, the model must retroactively figure out positional indexing from a fuzzy encoding that didn't prioritize position 25. This is why it gets it wrong (off by one — returning the 26th name instead of the 25th).

With repetition, the model has already seen the question once. The second time the list appears during prefill, the encoder **knows** to attend to the 25th item, producing a more targeted encoding and more reliable retrieval.

The **Prefill (Counting)** condition in this benchmark makes this visible: by seeding the model's response with explicit numbering (`1. Dale Lopez, 2. Peter Sanchez, 3. Allen Harris...`), the model is forced to map each name to its position step-by-step. The output shows every intermediate step — and the model counts correctly all the way to `25. Paul Sanchez`. This confirms that the model *can* retrieve the right answer when positional tracking is made explicit, but fails when it must do so implicitly from a single-pass generic encoding.

## Setup

- **Models**: `gemini-2.0-flash-lite`, `gemini-3-flash-preview` (via Vertex AI)
- **SDK**: `google-genai` v1.64.0
- **Python**: 3.13
- **Task**: NameIndex (N=50, i=25)
- **Temperature**: 0.0

### NameIndex Task

The model receives a list of 50 human names and is asked to identify the 25th name. The target (`Paul Sanchez`) sits at the midpoint of the list — the position where models are most prone to errors.

### Prompt Conditions

| Condition | Format | Purpose |
|---|---|---|
| No Repetition | `<QUERY>` | Baseline — single pass |
| Prefill (Counting) | Multi-turn: user sends `<QUERY>`, model response seeded with first 3 names numbered | Shows intermediate counting steps |
| Vanilla Repetition | `<QUERY>\n<QUERY>` | Repeat without bridging text |

## Usage

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_benchmark.py
```

### Model Selection

By default the benchmark runs against `gemini-2.0-flash-lite`. Use `--models` to select one or more models:

```bash
# Single model
.venv/bin/python run_benchmark.py --models gemini-3-flash-preview

# Multiple models (runs all conditions against each)
.venv/bin/python run_benchmark.py --models gemini-2.0-flash-lite gemini-3-flash-preview
```

You can also set the `MODEL` environment variable (CLI `--models` takes precedence):

```bash
MODEL=gemini-3-flash-preview .venv/bin/python run_benchmark.py
```

Configure your GCP project in `.env`:

```
VERTEXAI=true
PROJECT=your-project-id
LOCATION=us-central1
```

## Reference

> Kwon, S. J., Jiang, Y., & Kolter, J. Z. (2024). *Repeat to Recall: Prompt Repetition Improves LLM Recall of Long Contexts*. arXiv:2512.14982. https://arxiv.org/html/2512.14982v1
