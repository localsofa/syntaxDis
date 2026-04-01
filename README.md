# syntaxDis
all scripts written and used for my syntax research extension on cycliclity in LLMs

# attentionHead.py
- calculates all attention weights across paraphrases
- compresses all heads and layers into one array
- caclulates differences in processing across sentences to determine 2-period cyclicity
- additional function: layer-specific attention analysis (set up for layer 8)

# dependencies.py
- utilizes sPaCy tokenizer to parse dependencies in each paraphrased sentence
- calculates and compares "arcs" in dependencies
- returns a matrix with comparison values, where 0 = identical

# logProbs.py
- calculates logprobs of specific words
- compares logprob results of word across all paraphrases
