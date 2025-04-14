import pandas as pd
import json
import re 
import os, sys, json
import textstat
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
import nltk 
import torch
from alignscore import AlignScore
from summac.model_summac import SummaCConv

nltk.download('punkt_tab')

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence]

def calc_rouge(preds, refs):
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                    use_stemmer=True, split_summaries=True)
  scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
  return np.mean([s['rouge1'].fmeasure for s in scores]), \
         np.mean([s['rouge2'].fmeasure for s in scores]), \
         np.mean([s['rougeLsum'].fmeasure for s in scores])

def calc_bertscore(preds, refs):
  P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cuda:0')
  return np.mean(F1.tolist())

def calc_readability(preds):
  fkgl_scores = []
  cli_scores = []
  dcrs_scores = []
  for pred in preds:
    fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
    cli_scores.append(textstat.coleman_liau_index(pred))
    dcrs_scores.append(textstat.dale_chall_readability_score(pred))
  return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)

def calc_alignscore(preds, docs):
  alignscorer = AlignScore(model='roberta-base', batch_size=16, device='cuda:0', \
                           ckpt_path='./models/AlignScore/AlignScore-base.ckpt', evaluation_mode='nli_sp')
  return np.mean(alignscorer.score(contexts=docs, claims=preds))

def cal_summac(preds, docs):
  model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
  return np.mean(model_conv.score(docs, preds)['scores'])

def eval(pred, gt):
    score_dict = {}
    rouge1_score, rouge2_score, rougel_score = calc_rouge(pred, gt)
    score_dict['ROUGE1'] = rouge1_score
    score_dict['ROUGE2'] = rouge2_score
    score_dict['ROUGEL'] = rougel_score
    fkgl_score, cli_score, dcrs_score = calc_readability(pred)
    score_dict['FKGL'] = fkgl_score
    score_dict['DCRS'] = dcrs_score
    score_dict['CLI'] = cli_score
    score_dict['BERTScore'] = calc_bertscore(pred, gt)
    # score_dict['AlignScore'] = calc_alignscore(pred, gt)
    # score_dict['SummaC'] = cal_summac(pred, gt)
    return score_dict


if __name__ == '__main__':
    plos_test = pd.read_json('../test/plos_test.json')
    plos_base = pd.read_csv('../result/plosBaseLLM.csv')
    plos_topk = pd.read_csv('../result/plosTopKLLM.csv')

    elife_test = pd.read_json('../test/elife_test.json')
    elife_base = pd.read_csv('../result/elifeBaseLLM.csv')
    elife_topk = pd.read_csv('../result/elifeTopKLLM.csv')

    out_dir = "./"

    # plos
    eval_table_base = pd.DataFrame()
    eval_table_topk = pd.DataFrame()
    for idx in range(len(plos_test)):
        tid = plos_test.iloc[idx]['id']
        gt = [' '.join(plos_test.iloc[idx]['summary'])]
        pred_base = [plos_base[plos_base['id'] == tid]['summary'].values[0]]
        pred_topk = [plos_topk[plos_topk['id'] == tid]['summary'].values[0]]

        score_dict_base = eval(pred_base, gt)
        score_dict_topk = eval(pred_topk, gt)
        score_dict_base['id'] = tid
        score_dict_topk['id'] = tid
        eval_table_base = pd.concat([eval_table_base, pd.DataFrame([score_dict_base])], ignore_index=True)
        eval_table_topk = pd.concat([eval_table_topk, pd.DataFrame([score_dict_topk])], ignore_index=True)
    eval_table_base.to_csv(os.path.join(out_dir, 'plosBaseLlmEval.csv'), index=False)
    eval_table_topk.to_csv(os.path.join(out_dir, 'plosTopkLlmEval.csv'), index=False)

    # elife
    eval_table_base = pd.DataFrame()
    eval_table_topk = pd.DataFrame()
    for idx in range(len(elife_test)):
        tid = elife_test.iloc[idx]['id']
        gt = [' '.join(elife_test.iloc[idx]['summary'])]
        pred_base = [elife_base[elife_base['id'] == tid]['summary'].values[0]]
        pred_topk = [elife_topk[elife_topk['id'] == tid]['summary'].values[0]]
        score_dict_base = eval(pred_base, gt)
        score_dict_topk = eval(pred_topk, gt)
        score_dict_base['id'] = tid
        score_dict_topk['id'] = tid
        eval_table_base = pd.concat([eval_table_base, pd.DataFrame([score_dict_base])], ignore_index=True)
        eval_table_topk = pd.concat([eval_table_topk, pd.DataFrame([score_dict_topk])], ignore_index=True)
    eval_table_base.to_csv(os.path.join(out_dir, 'elifeBaseLlmEval.csv'), index=False)
    eval_table_topk.to_csv(os.path.join(out_dir, 'elifeTopkLlmEval.csv'), index=False)


