import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_ind

# -----------------------
# DATA
# -----------------------

WH_WORDS = ["what", "which", "who", "whom"]
CLAUSE_MARKERS = ["that", "which", "who", "because"]

ONE = ["which", "what"]
NOUN1 = ["book"]
NOUN2 = ["student"]

VERBONE = ["say", "claim", "state", "report", "said", "claimed"]
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

print(f"Loaded {len(paraphrases)} paraphrases")

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

    # (layers, heads, seq, seq)
    attentions = torch.stack(outputs.attentions)

    # average over layers + heads → (seq, seq)
    avg_attention = attentions.mean(dim=(0, 1, 2))

    return avg_attention, inputs

# -----------------------
# TOKEN UTILITIES
# -----------------------

def find_indices(tokens, word_list):
    indices = []
    for i, tok in enumerate(tokens):
        for w in word_list:
            if w in tok.lower():
                indices.append(i)
    return indices

# -----------------------
# STRUCTURE GUESS
# -----------------------

def guess_structure(tokens):
    return {
        "wh": find_indices(tokens, WH_WORDS),
        "comp": find_indices(tokens, CLAUSE_MARKERS),
        "verbone": find_indices(tokens, VERBONE),
        "verbtwo": find_indices(tokens, VERBTWO),
        "nounone": find_indices(tokens, NOUN1),
        "nountwo": find_indices(tokens, NOUN2),
        "mary": find_indices(tokens, MARY),
        "john": find_indices(tokens, JOHN),
        "one": find_indices(tokens, ONE),
    }

# -----------------------
# DOMAIN DEFINITIONS
# -----------------------

def get_domains(structure):
    # vP = lower clause
    vp = structure["verbtwo"] + structure["nounone"] + structure["john"]

    # CP = higher clause
    cp = structure["comp"] + structure["verbone"] + structure["mary"]

    return vp, cp

# -----------------------
# DOMAIN ATTENTION
# -----------------------

def domain_attention(attn, A, B):
    vals = []
    for i in A:
        for j in B:
            if i < attn.shape[0] and j < attn.shape[0]:
                vals.append(attn[i, j].item())
    return np.mean(vals) if vals else np.nan

# -----------------------
# EXTRACT DOMAIN METRICS
# -----------------------

def extract_domains(sentence):
    attn, inputs = get_attention(sentence)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    structure = guess_structure(tokens)
    vp, cp = get_domains(structure)

    return {
        "vP_internal": domain_attention(attn, vp, vp),
        "CP_internal": domain_attention(attn, cp, cp),
        "cross_vP_CP": domain_attention(attn, vp, cp),
        "cross_CP_vP": domain_attention(attn, cp, vp)
    }

# -----------------------
# RUN ANALYSIS
# -----------------------

results = [extract_domains(s) for s in paraphrases]

print("\n--- DOMAIN ATTENTION ---")
for key in results[0].keys():
    series = [r[key] for r in results]
    print(f"{key}: {series}")

# -----------------------
# STABILITY METRIC
# -----------------------

def stability(series):
    return np.nanstd(series)

vp_series = [r["vP_internal"] for r in results]
cp_series = [r["CP_internal"] for r in results]
cross_series = [r["cross_vP_CP"] for r in results]

print("\n--- STABILITY ---")
print("vP stability:", stability(vp_series))
print("CP stability:", stability(cp_series))
print("Cross stability:", stability(cross_series))

# -----------------------
# DOMAIN SEPARATION SCORE
# -----------------------

def domain_separation(vp, cp, cross):
    return np.nanmean([vp, cp]) - cross

sep_scores = [
    domain_separation(r["vP_internal"], r["CP_internal"], r["cross_vP_CP"])
    for r in results
]

print("\n--- DOMAIN SEPARATION ---")
print("Scores:", sep_scores)
print("Mean separation:", np.nanmean(sep_scores))

# -----------------------
# STATISTICAL TEST
# -----------------------

within = vp_series + cp_series
cross = cross_series

t, p = ttest_ind(within, cross, nan_policy='omit')

print("\n--- T-TEST ---")
print("t =", t)
print("p =", p)