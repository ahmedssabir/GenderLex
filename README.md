# GenderLex 
[![arXiv](https://img.shields.io/badge/arXiv-2507.02679-b31b1b.svg)](https://arxiv.org/pdf/2507.02679.pdf) [![huggingface](https://img.shields.io/badge/🤗-GenderLex-yellow)](https://huggingface.co/datasets/AhmedSSabir/GenderLex)


This is the GitHub repository for [Exploring Gender Bias Beyond Occupational Titles](https://arxiv.org/pdf/2507.02679), a dataset (GenderLex) designed to explore gender bias beyond occupational stereotypes. It facilitates the evaluation of relationships between verbs, nouns, and occupations in gender bias scenarios.

The GenderLex dataset Includes:

- `Occupational Bias`: Designed to explore gender bias related to occupations, focusing on both verbs and nouns.
- `Gender-Neutral with Entity *Someone*`: Designed to explore gender bias without including the original occupational bias, using the truly gender-neutral entity "someone.",  focusing on both verbs and nouns.
- `Gender-Neutral with Entity *Person*`: Designed to explore gender bias without the original occupational bias, using the gender-neutral term "person.",  focusing on both verbs and nouns.



## The Dataset

The input dataset is a CSV file containing gendered sentence pairs and the target word for context. Each row should include the following columns:

- `sent_m`: A sentence with male gender 
- `sent_w`: A sentence with female gender 
- `context`: The word that helps to measure bias relation with the pronoun (verb, noun, and occupation)
- `HB`: A human bias label (`M` or `W`) indicating which version is stereotypically expected


## Quantifying Biases in LLMs with External Bias Context 

### Requirement

Use Python 3 (we use Python 3.10.12) and install the required packages.

```
pip install -r requirements.txt
```

The code for measuring biases in LLMs is available at [metric.py](https://github.com/ahmedssabir/GenderLex/tree/main/code). You can run the code using the following command:

```
python clozegender_metric.py \
  --input GenderLex_verb.csv \
  --output GenderLex_verb-fasttext.csv \
  --summary GenderLex_verb-fasttext-summary.txt \
  --emd fasttext --emd_path crawl-300d-2M-subword.vec \
  --model_name gpt2-xl
```
For benchmark datasets (e.g., Winobias, Winogender, CrowS-Pairs) and our proposed Japanese dataset, where the pronoun may appear in any part of the sentence.

```  
python meanpro_metric.py \
  --input winobias_occ.csv \
  --output winobias_occ-fasttext.csv \
  --summary winobias-fasttext-summary.txt \
  --emd fasttext --emd_path crawl-300d-2M-subword.vec \
  --model_name gpt2-xl
```


For `LLMs`, the code supports any LLMs on Huggingface (tested with `GPT2-XL`, `EleutherAI/gpt-j-6b`, `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-70B` and `DeepSeek-R1-8B`. For wordembedding `glove` [glove 300d 840b](https://nlp.stanford.edu/projects/glove/), `fasttext`[crawl-300d-2M-subword.vec](https://fasttext.cc/docs/en/english-vectors.html), `word2vec` [
word2vec-GoogleNews-vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors
), or `glove-GN` [GN-GloVe using 1-billion](https://drive.google.com/file/d/1g1QPqbIlQorwlfGShtPbZVk6mfwodQgE/view), `glove_dd` [vector_ddglove_gender](https://drive.google.com/drive/folders/1yqpBcqENLkPrzL1wfkw08GkO6VQ8m2tf) and our Race-Neutral `glove-RN` [1b-vectors300-RN](https://www.dropbox.com/scl/fi/2f45d6zpiqxdcdyc8ms3w/1b-vectors300-RN.txt?rlkey=a1opowg3g7585atpls01x7lk8&dl=0)



The `--output` file will store the sentence scores for each example. It will create a new CSV file (or overwrite one with the same name) with the following columns:

- `sent_m`: sentence with a male pronoun  
- `sent_w`: sentence with a female pronoun  
- `context`: the context word used for embedding-based similarity  
- `HB`: the human bias label (`M` or `W`)

And the following additional columns are computed by the script:

- `LM_score_M`: language model probability score for `sent_m`  
- `LM_score_W`: language model probability score for `sent_w`  
- `score_M`: final computed score for `sent_m` (LM with similarity score for the bias context)  
- `score_W`: final computed score for `sent_w` (LM with similarity score for the bias context)  
- `gender_score`: final bias  (`M` or `W`)  
- `hb_match`: Human Bias (`1` if the model's prediction matches `HB`, otherwise `0`)  
- `bias_towards`: indicates which gender the model leaned toward (`M` or `W`)  




# Acknowledgements

This work has received funding from the EU H2020 program under the SoBigData++ project (grant agreement No. 871042), by the CHIST-ERA grant No. CHIST-ERA-19-XAI-010, (ETAg grant No. SLTAT21096), and partially funded by HAMISON project.



