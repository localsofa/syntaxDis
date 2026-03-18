import spacy
from spacy.tokens import Doc
import numpy as np
# import skilearn
# import matplotlib

nlp = spacy.load("en_core_web_sm")

with open("pilotParaphrases.txt", "r") as f:
    paraphrases = [line.strip() for line in f.readlines()]

print(f"Loaded {len(paraphrases)} paraphrases:")
for i, p in enumerate(paraphrases):
    print(f"{i}: {p}")

# WORKS
# original prompt: Paraphrase the following sentence: The book that the teacher gave me was long.


# DEPENDENCY PARSING

def get_dependency_tree(sentence): 
    doc = nlp(sentence)
    # Extract dependency arcs: (head_index, dependent_index, dep_label)
    arcs = [(t.head.i, t.i, t.dep_) for t in doc]
    return arcs

# Get trees for all paraphrases
trees = [get_dependency_tree(p) for p in paraphrases]

# Compute tree edit distance (simplified)
def tree_edit_distance(tree1, tree2):
    arcs1 = set(tree1)
    arcs2 = set(tree2)
    return len(arcs1 - arcs2) + len(arcs2 - arcs1)  # Symmetric difference

# Build a distance matrix
distance_matrix = np.zeros((len(trees), len(trees)))
for i in range(len(trees)):
    for j in range(len(trees)):
        distance_matrix[i, j] = tree_edit_distance(trees[i], trees[j])

print("Tree Edit Distance Matrix:")
print(distance_matrix)


# WORKS
# FOR LATER VISUALIZATION
# import matplotlib.pyplot as plt
# plt.imshow(distance_matrix, cmap="viridis")
# plt.colorbar()
# plt.title("Tree Edit Distance Between Paraphrases")
# plt.show()