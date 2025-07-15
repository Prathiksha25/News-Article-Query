from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

expected = input("Enter the expected output: ")
output = input("Enter the Generated output : ")

emb1 = model.encode(expected, convert_to_tensor=True)
emb2 = model.encode(output, convert_to_tensor=True)

score = util.pytorch_cos_sim(emb1, emb2).item()

def evaluate_answer(score):
    if score > 0.75:
        return "Correct"
    elif score > 0.4:
        return "Partially Correct"
    else:
        return "Incorrect"

verdict = evaluate_answer(score)

print(f"\nSimilarity Score: {score:.2f}")
print(f"Evaluation Verdict: {verdict}")