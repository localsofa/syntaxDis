import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# DATA
# -----------------------

WH_WORDS = ["what", "which", "who", "whom"]
CLAUSE_MARKERS = ["that", "which", "who", "because"]
VERBS = ["read"]
AUX = ["was", "did", "is"]
SUBJECT = ["Mary"]
OBJECT = ["book"]

ONE = ["which", "what"]
NOUN1 = ["book"]
NOUN2 = ["student"]
VERBONE = ["say", "claim", "state", "according", "stated", "said", "claimed"]
SAID = ["said", "say"]
CLAIMED = ["claimed", "claim"]  
VERBTWO = ["read"]
MARY = ["mary"]
JOHN = ["john"]

paraphrases = [
"Which book did Mary say that the student claimed John read?",
"Which book was it that Mary claimed the student said John read?",
"Mary claimed that the student said John read which book?",
"What book does Mary claim the student said John read?",
"Which book did Mary report that the student said the student said John read?",
"According to Mary, which book did the student say John read?",
"Which book is Mary claimed to have been the one the student said John read?",
"Which book did Mary claim was said by the student to have been read by John?",
"It was which book that Mary claimed the student said John read?"
]

print(f"Loaded {len(paraphrases)} paraphrases:")
for i, p in enumerate(paraphrases):
    print(f"{i}: {p}")

# -----------------------
# MODEL
# -----------------------

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.output_attentions = True
model.eval()


# -----------------------
# ATTENTION EXTRACTION
# -----------------------

def get_attention(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = torch.stack(outputs.attentions)
    avg_attention = attentions.mean(dim=(0, 1, 2))  # full average
    #avg_attention = attentions.mean(dim=1)  # averages heads -- this would be good to do, but honestly i'd have to change so much of the script to accodmodate the vector 
    return avg_attention, inputs


# -----------------------
# TOKEN UTILITIES
# -----------------------

def get_tokens(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens


def find_indices(tokens, word_list):
    indices = []
    for i, tok in enumerate(tokens):
        for w in word_list:
            if w in tok.lower():
                indices.append(i)
    return indices


# -----------------------
# HEURISTIC STRUCTURE
# -----------------------

def guess_structure(tokens):
    wh = find_indices(tokens, WH_WORDS)
    comp = find_indices(tokens, CLAUSE_MARKERS)
    subj = find_indices(tokens, SUBJECT)
    verb = find_indices(tokens, VERBS)
    obj = find_indices(tokens, OBJECT)
    aux = find_indices(tokens, AUX)
    verbone = find_indices(tokens, VERBONE)
    verbtwo = find_indices(tokens, VERBTWO)
    nounone = find_indices(tokens, NOUN1)
    nountwo = find_indices(tokens, NOUN2)
    mary = find_indices(tokens, MARY)
    john = find_indices(tokens, JOHN)
    one = find_indices(tokens, ONE)
    said = find_indices(tokens, SAID)
    claimed = find_indices(tokens, CLAIMED)

    return {
        "subj": subj,
        "verb": verb,
        "obj": obj,
        "wh": wh,
        "comp": comp,
        "aux": aux,
        "verbone": verbone,
        "verbtwo": verbtwo,
        "nounone": nounone,
        "mary": mary,
        "john": john,
        "one": one,
        "said": said,
        "claimed": claimed,
        "nountwo": nountwo
    }


# -----------------------
# ATTENTION RELATIONS
# -----------------------

def extract_relations(sentence):
    attn, inputs = get_attention(sentence)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    structure = guess_structure(tokens)

    # computes averages in vP submatrix (verb & object region) and COMP submatrix (WH & complement region)
    # if stable across all forms --> structural dependency
    def avg_attention(src, tgt):
        vals = []
        for s in src:
            for t in tgt:
                if s < attn.shape[0] and t < attn.shape[0]:
                    vals.append(attn[s, t].item())
        return np.mean(vals) if vals else np.nan

    return {
        "wh→verbone": avg_attention(structure["one"], structure["verbone"]),
        "wh→verbtwo": avg_attention(structure["one"], structure["verbtwo"]),
        "wh→comp": avg_attention(structure["one"], structure["comp"]),
        "aux→verb": avg_attention(structure["aux"], structure["verb"]),
        "nounone→verb": avg_attention(structure["nounone"], structure["verb"]),
        "mary→verbone": avg_attention(structure["mary"], structure["verbone"]),
        "john→verbtwo": avg_attention(structure["john"], structure["verbtwo"]),
        "mary→verbtwo": avg_attention(structure["mary"], structure["verbtwo"]),
        "john→verbone": avg_attention(structure["john"], structure["verbone"]),
        "one→nounone": avg_attention(structure["one"], structure["nounone"]),
        "nountwo→said": avg_attention(structure["nountwo"], structure["said"]),
        "nountwo→claimed": avg_attention(structure["nountwo"], structure["claimed"]),
        "mary→said": avg_attention(structure["mary"], structure["said"]),
        "mary→claimed": avg_attention(structure["mary"], structure["claimed"]),
        "said→comp": avg_attention(structure["said"], structure["comp"]),
        "claimed→comp": avg_attention(structure["claimed"], structure["comp"]),
        "comp→mary": avg_attention(structure["comp"], structure["mary"]),
        "comp→nountwo": avg_attention(structure["comp"], structure["nountwo"]),
        "comp→john": avg_attention(structure["comp"], structure["john"])
    }


# -----------------------
# RUN ANALYSIS
# -----------------------

results = []

for sent in paraphrases:
    rels = extract_relations(sent)
    results.append(rels)

print("\n--- STRUCTURAL ATTENTION SERIES ---")

for key in results[0].keys():
    series = [r[key] for r in results]
    print(f"\n{key}:")
    print(series)


# -----------------------
# GLOBAL SIMILARITY
# -----------------------

attention_mats = [get_attention(p)[0] for p in paraphrases]

max_len = max(mat.shape[0] for mat in attention_mats)
padded = [torch.nn.functional.pad(mat, (0, max_len - mat.shape[1], 0, max_len - mat.shape[0])) for mat in attention_mats]

flattened = [mat.flatten().numpy() for mat in padded]
similarity_matrix = cosine_similarity(flattened)

print("\nAttention Similarity Matrix:")
print(similarity_matrix)


# -----------------------
# SIM METRICS
# -----------------------

sim_consecutive = [similarity_matrix[i][i+1] for i in range(len(similarity_matrix) - 1)]
sim_every_other = [similarity_matrix[i][i+2] for i in range(len(similarity_matrix) - 2)]

print("\nConsecutive:", sim_consecutive)
print("Every-other:", sim_every_other)


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled


print(f"Cohen's d: {cohen_d(sim_consecutive, sim_every_other):.3f}")

# INTERPRETATION:
# consecutive similarity ≈ every-other similarity = stable structure
# consecutive >> every-other = gradual drift
# sudden drops at specific steps = chunking effects
