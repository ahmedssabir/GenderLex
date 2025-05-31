import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import log_softmax
import os
import csv
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model cache
model_cache = {}

def init_model(model_name):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

def load_embedding_model(emd, emd_path):
    global model_cache

    cache_key = (emd, emd_path)
    if cache_key in model_cache:
        return model_cache[cache_key]

    def load_glove(path):
        if 'bin' in path:
            model = KeyedVectors.load_word2vec_format(path, binary=True)
            words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
            vecs = [model[w] for w in words]
            vecs = np.array(vecs, dtype='float32')
            wv = vecs
            vocab = words
            w2i = {word: index for index, word in enumerate(words)}
        else:
            with open(path) as f:
                lines = f.readlines()
                wv = []
                vocab = []
                for line in lines:
                    tokens = line.strip().split(' ')
                    if not len(tokens) == 301:
                        continue
                    vocab.append(tokens[0])
                    wv.append([float(elem) for elem in tokens[1:]])
            w2i = {w: i for i, w in enumerate(vocab)}
            wv = np.array(wv).astype(float)
        return wv, w2i, vocab

    if emd == 'glove':
        glove_file = datapath(emd_path)
        tmp_file = get_tmpfile(f"test_word2vec_{os.path.basename(emd_path)}.txt")
        if not os.path.exists(tmp_file):
            glove2word2vec(glove_file, tmp_file)
        model_sim = KeyedVectors.load_word2vec_format(tmp_file)

    elif emd == 'fasttext':
        binary_model_path = emd_path + '.bin'
        if not os.path.exists(binary_model_path):
            embedding_dict = KeyedVectors.load_word2vec_format(emd_path, binary=False)
            embedding_dict.save_word2vec_format(binary_model_path, binary=True)
        model_sim = KeyedVectors.load_word2vec_format(binary_model_path, binary=True)

    elif emd == 'word2vec':
        model_sim = KeyedVectors.load_word2vec_format(emd_path, binary=True)

    elif emd == 'glove_dd':
        wv, w2i, vocab = load_glove(emd_path)
        model_sim = {'wv': wv, 'w2i': w2i, 'vocab': vocab}

    else:
        raise ValueError("Invalid model_flag. Choose from 'glove', 'fasttext', 'word2vec', or 'glove_dd'.")

    model_cache[cache_key] = model_sim
    return model_sim

def cloze_prob_last_word(text):
    whole_text_encoding = tokenizer.encode(text, return_tensors='pt').to(device)
    text_list = text.split()
    stem = ' '.join(text_list[:-1])
    stem_encoding = tokenizer.encode(stem, return_tensors='pt').to(device)

    cw_start_index = stem_encoding.size(1)
    cw_length = whole_text_encoding.size(1) - cw_start_index

    with torch.no_grad():
        outputs = model(whole_text_encoding)
        logits = outputs.logits

    log_probs = log_softmax(logits, dim=-1)

    cw_log_probs = []
    for i in range(cw_length):
        token_index = whole_text_encoding[0, cw_start_index + i]
        cw_log_probs.append(log_probs[0, cw_start_index + i - 1, token_index].item())

    total_log_prob = np.sum(cw_log_probs)
    return np.exp(total_log_prob)

def Context_score(sentence, context, pronoun, emd, emd_path):
    LM = cloze_prob_last_word(sentence)
    model_sim = load_embedding_model(emd, emd_path)

    try:
        if isinstance(model_sim, dict):
            wv = model_sim['wv']
            w2i = model_sim['w2i']
            if context in w2i and pronoun in w2i:
                vec1 = wv[w2i[context]]
                vec2 = wv[w2i[pronoun]]
                sim_score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            else:
                with open('out_of_dict.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([context, pronoun])
                sim_score = 0.0
        else:
            sim_score = model_sim.similarity(context, pronoun)
    except KeyError:
        with open('out_of_dict.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([context, pronoun])
        sim_score = 0.0

    score = np.power(float(LM), (1 - float(sim_score)))
    return score

def main(input, output, emd, emd_path, summary, model_name):
    init_model(model_name)

    output_data = []
    total_hb_matches = 0
    total_sentences = 0
    count_m = 0
    count_f = 0

    with open(input, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        total_rows = sum(1 for _ in open(input)) - 1
        csvfile.seek(0)

        for row in tqdm(csvreader, total=total_rows, desc="Processing rows"):
            sent_m = row['sent_m']
            sent_w = row['sent_w']
            context = row['context']
            HB = row['HB']

            LM_score_M = cloze_prob_last_word(sent_m)
            LM_score_W = cloze_prob_last_word(sent_w)

            score_M = Context_score(sent_m, context, 'him', emd, emd_path)
            score_W = Context_score(sent_w, context, 'her', emd, emd_path)

            if score_M > score_W:
                computed_gender = 'M'
                comparison_result = "Sentence M has a higher LM score."
            elif score_M < score_W:
                computed_gender = 'W'
                comparison_result = "Sentence W has a higher LM score."
            else:
                computed_gender = "Equal"
                comparison_result = "Scores are equal."

            hb_match = 1 if computed_gender == HB else 0
            bias_towards = 'M' if computed_gender == 'M' else 'W'

            total_hb_matches += hb_match
            if computed_gender == 'M':
                count_m += 1
            elif computed_gender == 'W':
                count_f += 1
            total_sentences += 1

            output_data.append({
                'sent_m': sent_m,
                'sent_w': sent_w,
                'context': context,
                'HB': HB,
                'LM_score_M': LM_score_M,
                'LM_score_W': LM_score_W,
                'score_M': score_M,
                'score_W': score_W,
                'gender_score': computed_gender,
                'hb_match': hb_match,
                'bias_towards': bias_towards
            })

    hb_match_ratio = total_hb_matches / total_sentences if total_sentences > 0 else 0
    m_ratio = count_m / (count_m + count_f) if (count_m + count_f) > 0 else 0
    f_ratio = count_f / (count_m + count_f) if (count_m + count_f) > 0 else 0

    print(f'Human Bias Match Ratio: {hb_match_ratio:.2f}')
    print(f'M Ratio: {m_ratio:.2f}')
    print(f'F Ratio: {f_ratio:.2f}')

    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['sent_m', 'sent_w', 'context', 'HB', 'LM_score_M', 'LM_score_W', 'score_M', 'score_W', 'gender_score', 'hb_match', 'bias_towards']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in output_data:
            writer.writerow(data)

    with open(summary, 'w') as txtfile:
        txtfile.write(f'Human Bias Match Ratio: {hb_match_ratio:.2f}\n')
        txtfile.write(f'M Ratio: {m_ratio:.2f}\n')
        txtfile.write(f'W Ratio: {f_ratio:.2f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process sentences to compute language model scores and gender bias.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output', type=str, default='result.txt', help='Path to the output CSV file.')
    parser.add_argument('--emd', type=str, default='glove', help="Embedding type: 'glove', 'fasttext', 'word2vec', or 'glove_dd'")
    parser.add_argument('--emd_path', type=str, default="", help="Path to the embedding model file")
    parser.add_argument('--summary', type=str, default='summary.txt', help='Path to the summary text file.')
    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='HuggingFace model name (e.g., gpt2-xl, EleutherAI/gpt-j-6b)')
    args = parser.parse_args()
    main(args.input, args.output, args.emd, args.emd_path, args.summary, args.model_name)
