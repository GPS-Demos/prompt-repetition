import argparse
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# 1. Initialize the client for Vertex AI
client = genai.Client(
    vertexai=os.getenv("VERTEXAI", "true").lower() == "true",
    project=os.getenv("PROJECT"),
    location=os.getenv("LOCATION"),
)

SUPPORTED_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-3-flash-preview",
]

DEFAULT_MODEL = "gemini-2.0-flash-lite"

parser = argparse.ArgumentParser(description="NameIndex Benchmark — Prompt Repetition for LLM Recall")
parser.add_argument(
    "--models",
    nargs="+",
    metavar="MODEL",
    help=f"Model(s) to benchmark (default: MODEL env var or {DEFAULT_MODEL})",
)
args = parser.parse_args()

# Resolve models: CLI --models > MODEL env var > DEFAULT_MODEL
if args.models:
    models: list[str] = args.models
else:
    env_model = os.getenv("MODEL")
    models = [env_model] if env_model else [DEFAULT_MODEL]

for m in models:
    if m not in SUPPORTED_MODELS:
        print(f"Warning: '{m}' is not in the supported models list {SUPPORTED_MODELS}", file=sys.stderr)

# 2. Setup the NameIndex task (N=50, i=25)
names = [
    "Dale Lopez", "Peter Sanchez", "Allen Harris", "Scott Davis",
    "Hudson Leviathan", "Daphne Kalman", "Dennis Davis", "Henry King",
    "Alfred Cooper", "Bruce Usher", "Travis Ramirez", "Rafael Jennings",
    "Richard Rogers", "Walter Young", "Caleb Harris", "Ben Kalman",
    "Donald Carter", "Richard Sterling", "Mark Nightingale", "Steven Carter",
    "Talia Kalman", "Dennis Hanson", "James Harris", "Craig Chavez",
    "Paul Sanchez", "Samuel Curtis", "Jacob James", "Allen Thomas",
    "Dale Evans", "James Fox", "Douglas Allen", "Orion Johnson",
    "Alexander Wright", "Eugene Morrison", "Nelson Lee", "Alan Young",
    "Caleb Ward", "Alberto Robinson", "Robert McCarthy", "Mark Price",
    "Kenneth Ramirez", "Jeffrey White", "Chad Cooper", "Arthur Waters",
    "Bruce Callahan", "Liam Leviathan", "Steven Robinson", "Alberto Murphy",
    "Leonard Johnson", "Robert Murphy"
]

expected_answer = "Paul Sanchez"
names_list_str = ", ".join(names)
query = f"Here's a list of names: {names_list_str}\nWhat's the 25th name?"

# 3. Define conditions
#
# Prefill condition: use multi-turn content to seed the model's response with
# the start of an explicit count, so we can see intermediate steps.
prefill_contents = [
    types.Content(
        role="user",
        parts=[types.Part(text=query + "\n\nCount through the names one by one to find the answer.")]
    ),
    types.Content(
        role="model",
        parts=[types.Part(text="Let me count through the list:\n1. Dale Lopez\n2. Peter Sanchez\n3. Allen Harris\n")]
    ),
]

conditions = [
    ("No Repetition",       query,                                           False),
    ("Prefill (Counting)",  prefill_contents,                                True),
    ("Vanilla Repetition",  f"{query}\n{query}",                             False),
]

# 4. Run each condition and collect results
print("=== NameIndex Benchmark (N=50, i=25) ===")
print(f"Expected answer: {expected_answer}")
print(f"Models: {', '.join(models)} | temperature=0.0")
print()

# results_by_model: {model_name: [(condition_name, answer, correct), ...]}
results_by_model = {}

for model in models:
    print(f"=== Model: {model} ===")
    print()

    model_results = []
    for idx, (name, prompt, is_prefill) in enumerate(conditions, 1):
        print(f"[{idx}/{len(conditions)}] {name}")

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        answer = response.text.strip()

        # For prefill, reconstruct full output (prefilled text + model continuation)
        if is_prefill:
            prefilled_text = "Let me count through the list:\n1. Dale Lopez\n2. Peter Sanchez\n3. Allen Harris\n"
            full_answer = prefilled_text + answer
            correct = expected_answer.lower() in full_answer.lower()
            model_results.append((name, full_answer, correct))
            print(f"  Response (with prefill):\n")
            for line in full_answer.split("\n"):
                print(f"    {line}")
            print(f"\n  Correct: {correct}")
        else:
            correct = expected_answer.lower() in answer.lower()
            model_results.append((name, answer, correct))
            print(f"  Response: {answer}")
            print(f"  Correct:  {correct}")

        print()

    results_by_model[model] = model_results

# 5. Summary
print("=== Summary ===")
for model, model_results in results_by_model.items():
    print(f"\n  Model: {model}")
    for name, answer, correct in model_results:
        mark = "Y" if correct else "X"
        # For summary, show just the final line or first 80 chars
        short = answer.split("\n")[-1].strip() if "\n" in answer else answer
        print(f"    {name:25s} [{mark}]  {short[:80]}")
