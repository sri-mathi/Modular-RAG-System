from ragas.metrics import answer_relevancy
print(dir(answer_relevancy))
if hasattr(answer_relevancy, 'n'):
    print(f"Current n: {answer_relevancy.n}")
elif hasattr(answer_relevancy, 'n_samples'):
    print(f"Current n_samples: {answer_relevancy.n_samples}")
