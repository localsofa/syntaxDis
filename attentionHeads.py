import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
# import matplotlib.pyplot as plt


# read .txt file
with open("pilotParaphrases.txt", "r") as f:
    paraphrases = [line.strip() for line in f.readlines()]

print(f"Loaded {len(paraphrases)} paraphrases:")
for i, p in enumerate(paraphrases):
    print(f"{i}: {p}")


# call model
model_name = "gpt2"  # or "mistralai/Mistral-7B-Instruct-v0.2" if you have GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
model.config.output_attentions = True

# get average attention across all heads/layers for each paraphrase
def get_attention(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = torch.stack(outputs.attentions)  
    avg_attention = attentions.mean(dim=(0, 1, 2))   
    return avg_attention

# assign attention values to variable
attention_mats = [get_attention(p) for p in paraphrases]

# pad attention matrices to the same size
max_len = max(mat.shape[0] for mat in attention_mats)
padded_mats = [torch.nn.functional.pad(mat, (0, max_len - mat.shape[1], 0, max_len - mat.shape[0])) for mat in attention_mats]

# flatten attention matrices; compute cosine similarity
flattened_attentions = [mat.flatten().numpy() for mat in padded_mats]
similarity_matrix = cosine_similarity(flattened_attentions)

print("Attention Similarity Matrix:")
print(similarity_matrix)


# specific attention layer analysis (in this case layer 8)
def get_layer_attention(sentence, layer=8):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    return outputs.attentions[layer].mean(dim=0)  

attention_mats_layer8 = [get_layer_attention(p, layer=8) for p in paraphrases]

# more padding
max_seq_len = max(mat.shape[1] for mat in attention_mats_layer8)
padded_mats_layer8 = [torch.nn.functional.pad(mat, (0, max_seq_len - mat.shape[2], 0, max_seq_len - mat.shape[1])) for mat in attention_mats_layer8]

flattened_layer8 = [mat.flatten().numpy() for mat in padded_mats_layer8]
similarity_matrix_layer8 = cosine_similarity(flattened_layer8)
print("Layer 8 Attention Similarity Matrix:")
print(similarity_matrix_layer8)


# similarities between consecutive paraphrases (n and n+1)
sim_consecutive = [similarity_matrix[i][i+1] for i in range(len(similarity_matrix) - 1)]

# similarities between every-other paraphrases (n and n+2)
sim_every_other = [similarity_matrix[i][i+2] for i in range(len(similarity_matrix) - 2)]

print("Consecutive similarities:", sim_consecutive)
mean1 = np.mean(sim_consecutive)
# [0.9687429, 0.9686376, 0.9686252, 0.9361479, 0.9658308, 0.9421689]
# mean = 0.9583588; std = 0.0139 


print("Every-other similarities:", sim_every_other)
mean2 = np.mean(sim_every_other) 
#[0.9998303, 0.99982417, 0.9666721, 0.9685287, 0.9065159, 0.9065159]
# mean = 0.9682743;  std = 0.0386


# effect size
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1) * np.std(x, ddof=1)**2 + (ny-1) * np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

d = cohen_d(sim_consecutive, sim_every_other)
print(f"Cohen's d: {d:.3f}")


# visualize later with BERTviz
# check if padding is done properly 

# matlab plot
# plt.imshow(similarity_matrix, cmap="viridis")
# plt.colorbar()
# plt.title("Attention Similarity Between Paraphrases")
# plt.show()