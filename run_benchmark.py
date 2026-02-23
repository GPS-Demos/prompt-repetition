import os
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

MODEL = "gemini-2.0-flash-lite"

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
    ("Verbose Repetition",  f"{query}\nLet me repeat that:\n{query}",        False),
]

# 4. Run each condition and collect results
print("=== NameIndex Benchmark (N=50, i=25) ===")
print(f"Expected answer: {expected_answer}")
print(f"Model: {MODEL} | temperature=0.0")
print()

results = []
for idx, (name, prompt, is_prefill) in enumerate(conditions, 1):
    print(f"[{idx}/{len(conditions)}] {name}")

    response = client.models.generate_content(
        model=MODEL,
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
        results.append((name, full_answer, correct))
        print(f"  Response (with prefill):\n")
        for line in full_answer.split("\n"):
            print(f"    {line}")
        print(f"\n  Correct: {correct}")
    else:
        correct = expected_answer.lower() in answer.lower()
        results.append((name, answer, correct))
        print(f"  Response: {answer}")
        print(f"  Correct:  {correct}")

    print()

# 5. Summary
print("=== Summary ===")
for name, answer, correct in results:
    mark = "Y" if correct else "X"
    # For summary, show just the final line or first 80 chars
    short = answer.split("\n")[-1].strip() if "\n" in answer else answer
    print(f"  {name:25s} [{mark}]  {short[:80]}")
