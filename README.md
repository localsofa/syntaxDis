# Notice:
all scripts are written using the module and Python versions which comply with the UoY's Viking server. model calling may have to be changed when using more modern versions of "transformers" etc

# syntaxDis
all scripts written and used for my syntax research extension on cycliclity in LLMs; attentionHeads.py was the main one i ended up using

# attentionHeads.py
1. tokenize sentence
uses a pretrained HuggingFace tokenizer (default: gpt2) to convert text into subword tokens


2. extract attention matrices
retrieves full attention tensors across all layers and heads
averages across heads and layers and produces a global attention map


3. user input of syntactic roles
* wh-elements (e.g. what, which, who)
* verbs (matrix and embedded predicates)
* subjects / objects
* complementisers / clause markers 


4. compute structural attention relations
computes directed attention scores such as:

* WH → verb (matrix / embedded)
* subject → verb
* object → verb
* clause marker → predicate head

each value represents mean attention from source token(s) to target token(s)


5. tracks paraphrase dynamics
* attention similarity matrices
* consecutive similarity (n → n+1)
* skip similarity (n → n+2)
* effect size (Cohen’s d) 


# dependencies.py
- utilizes sPaCy tokenizer to parse dependencies in each paraphrased sentence
- calculates and compares "arcs" in dependencies
- returns a matrix with comparison values, where 0 = identical

# logProbs.py
- calculates logprobs of specific words
- compares logprob results of word across all paraphrases
