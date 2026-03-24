from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

with open("pilotParaphrases.txt", "r") as f:
    paraphrases = [line.strip() for line in f.readlines()]

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_log_prob(sentence, target_token):
    inputs = tokenizer(sentence, return_tensors="pt")
    tokenized = tokenizer.tokenize(sentence)
    print("Tokenized:", tokenized)  # debug

    # prepend "Ġ" to target token // match tokenizer's format
    target_token_with_prefix = "Ġ" + target_token if target_token not in ["'", ".", ",", "!", "?"] else target_token

    # find position of target token
    try:
        target_position = tokenized.index(target_token_with_prefix)
    except ValueError:
        return float('nan')

    # logits and log probs
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits[0]
    target_token_id = tokenizer.convert_tokens_to_ids([target_token_with_prefix])[0]
    log_prob = torch.log_softmax(logits[target_position], dim=-1)[target_token_id]
    return log_prob.item()

#controls
target_tokens = ["my", "I", "me"]
log_probs = {token: [] for token in target_tokens}

for p in paraphrases:
    for token in target_tokens:
        log_prob = get_log_prob(p, token)
        log_probs[token].append(log_prob)

for token in log_probs:
    print(f"Log probabilities for '{token}': {log_probs[token]}")