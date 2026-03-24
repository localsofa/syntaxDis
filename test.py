import spacy
from spacy.tokens import Doc
import numpy as np
# original prompt: Paraphrase the following sentence: The book that the teacher gave me was long.


nlp = spacy.load("en_core_web_sm")

with open("pilotParaphrases.txt", "r") as f:
    paraphrases = [line.strip() for line in f.readlines()]

print(f"Loaded {len(paraphrases)} paraphrases:")
for i, p in enumerate(paraphrases):
    print(f"{i}: {p}")

# dependency parsing

def get_dependency_tree(sentence): 
    doc = nlp(sentence)
    arcs = [(t.head.i, t.i, t.dep_) for t in doc]
    return arcs

trees = [get_dependency_tree(p) for p in paraphrases]

# tree edit distance
def tree_edit_distance(tree1, tree2):
    arcs1 = set(tree1)
    arcs2 = set(tree2)
    return len(arcs1 - arcs2) + len(arcs2 - arcs1)  # Symmetric difference

# distance matrix
distance_matrix = np.zeros((len(trees), len(trees)))
for i in range(len(trees)):
    for j in range(len(trees)):
        distance_matrix[i, j] = tree_edit_distance(trees[i], trees[j])

print("Tree Edit Distance Matrix:")
print(distance_matrix)

# later
# import matplotlib.pyplot as plt
# plt.imshow(distance_matrix, cmap="viridis")
# plt.colorbar()
# plt.title("Tree Edit Distance Between Paraphrases")
# plt.show()