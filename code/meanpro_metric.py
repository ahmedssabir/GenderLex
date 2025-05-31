import argparse
import os
import csv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cache = {}
tokenizer = None
model = None

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

    else:
        raise ValueError("Invalid model_flag. Choose from 'glove', 'fasttext', or 'word2vec'.")

    model_cache[cache_key] = model_sim
    return model_sim

def prob_mean(text):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    probs = softmax(shift_logits, dim=-1)
    gathered_probs = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    dot_token_id = tokenizer.convert_tokens_to_ids('.')
    non_punctuation_mask = (shift_labels != dot_token_id)

    filtered_probs = gathered_probs[non_punctuation_mask]
    mean_prob = torch.mean(filtered_probs).item()

    return mean_prob

def Context_score(sentence, context, pronoun, emd, emd_path):
    LM = prob_mean(sentence)

    model_sim = load_embedding_model(emd, emd_path)

    try:
        sim_score = model_sim.similarity(context, pronoun)
    except KeyError:
        print('out_of_dict', context, pronoun)
        sim_score = 0.0

    score = pow(float(LM), (1 - float(sim_score)))
    return score

def main(input, output, emd, emd_path, model_name, summary_file):
    init_model(model_name)

    output_data = []
    total_human_bias_matches = 0
    total_sentences = 0
    count_m = 0
    count_f = 0

    with open(input, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        for row in csvreader:
            sent_m = row['sent_m']
            sent_w = row['sent_w']
            context = row['context']
            pronoun_m = row['prounoun_m']
            pronoun_w = row['prounoun_w']
            human_bias = row['HB']

            LM_score_M = prob_mean(sent_m)
            LM_score_W = prob_mean(sent_w)

            score_M = Context_score(sent_m, context, pronoun_m, emd, emd_path)
            score_W = Context_score(sent_w, context, pronoun_w, emd, emd_path)

            if score_M > score_W:
                computed_gender = 'M'
            elif score_M < score_W:
                computed_gender = 'W'
            else:
                computed_gender = 'Equal'

            human_bias_match = 1 if computed_gender == human_bias else 0
            bias_towards = 'M' if computed_gender == 'M' else 'W'

            total_human_bias_matches += human_bias_match
            if computed_gender == 'M':
                count_m += 1
            elif computed_gender == 'W':
                count_f += 1
            total_sentences += 1

            output_data.append({
                'sent_m': sent_m,
                'sent_w': sent_w,
                'context': context,
                'prounoun_m': pronoun_m,
                'prounoun_w': pronoun_w,
                'human_bias': human_bias,
                'LM_score_M': LM_score_M,
                'LM_score_W': LM_score_W,
                'score_M': score_M,
                'score_W': score_W,
                'gender_score': computed_gender,
                'human_bias_match': human_bias_match,
                'bias_towards': bias_towards
            })

    human_bias_match_ratio = total_human_bias_matches / total_sentences if total_sentences > 0 else 0
    m_ratio = count_m / (count_m + count_f) if (count_m + count_f) > 0 else 0
    f_ratio = count_f / (count_m + count_f) if (count_m + count_f) > 0 else 0

    print(f'Human Bias Match Ratio: {human_bias_match_ratio:.2f}')
    print(f'M Ratio: {m_ratio:.2f}')
    print(f'F Ratio: {f_ratio:.2f}')

    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['sent_m', 'sent_w', 'context', 'prounoun_m', 'prounoun_w', 'human_bias',
                      'LM_score_M', 'LM_score_W', 'score_M', 'score_W',
                      'gender_score', 'human_bias_match', 'bias_towards']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in output_data:
            writer.writerow(data)

    with open(summary_file, 'w') as txtfile:
        txtfile.write(f'Human Bias Match Ratio: {human_bias_match_ratio:.2f}\n')
        txtfile.write(f'M Ratio: {m_ratio:.2f}\n')
        txtfile.write(f'F Ratio: {f_ratio:.2f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute gender bias with LLM + word similarity.')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file.')
    parser.add_argument('--output', type=str, default='result_xl_final.txt', help='Path to output CSV file.')
    parser.add_argument('--emd', type=str, default='glove', help="Embedding type: 'glove', 'fasttext', or 'word2vec'")
    parser.add_argument('--emd_path', type=str, default='', help="Path to the embedding model file.")
    parser.add_argument('--model_name', type=str, default='gpt2-xl', help="HuggingFace model name like 'gpt2', 'gpt2-xl', 'EleutherAI/gpt-neo-1.3B'")
    parser.add_argument('--summary', type=str, default='summary.txt', help="Path to summary file.")

    args = parser.parse_args()
    main(args.input, args.output, args.emd, args.emd_path, args.model_name, args.summary)

