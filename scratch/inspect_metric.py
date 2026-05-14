from ragas.metrics import answer_relevancy
print("--- Instance Variables ---")
for key, value in vars(answer_relevancy).items():
    print(f"{key}: {value}")

print("\n--- Class Dictionary ---")
for key, value in answer_relevancy.__class__.__dict__.items():
    if not key.startswith("__"):
        print(f"{key}: {value}")
