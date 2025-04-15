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
# from alignscore import AlignScore
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
    score_dict['SummaC'] = cal_summac(pred, gt)
    return score_dict

def add_summary_row(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    mean_row = df[numeric_cols].mean()
    std_row = df[numeric_cols].std()
    summary_row = [f"{mean:.3f} ± {std:.3f}" for mean, std in zip(mean_row, std_row)]
    summary_df = pd.DataFrame([summary_row + ["mean ± std"]], columns=numeric_cols.tolist() + ["id"])
    final_df = pd.concat([df, summary_df], ignore_index=True)
    return final_df





if __name__ == '__main__':
    plos_test = pd.read_json('../test/plos_test.json')
    plos_base = pd.read_csv('../result/plosBaseLLM.csv')
    plos_top20 = pd.read_csv('../result/plosTextRankTop20LLM.csv')
    plos_top50 = pd.read_csv('../result/plosTextRankTop50LLM.csv')
    plos_super = pd.read_csv('../result/plosSupervisedTop50LLM.csv')

    elife_test = pd.read_json('../test/elife_test.json')
    elife_base = pd.read_csv('../result/elifeBaseLLM.csv')
    elife_top20 = pd.read_csv('../result/elifeTextRankTop20LLM.csv')
    elife_top50 = pd.read_csv('../result/elifeTextRankTop50LLM.csv')
    elife_super = pd.read_csv('../result/elifeSupervisedTop50LLM.csv')

    out_dir = "./"
    
    # evaluation 
    for dataset in ['plos', 'elife']:
        if dataset == 'plos':
            test_res = plos_test
            base_res = plos_base
            top20_res = plos_top20
            top50_res = plos_top50
            super_res = plos_super
        else:
            test_res = elife_test
            base_res = elife_base
            top20_res = elife_top20
            top50_res = elife_top50
            super_res = elife_super

        eval_table_base = pd.DataFrame()
        eval_table_top20 = pd.DataFrame()
        eval_table_top50 = pd.DataFrame()
        eval_table_super = pd.DataFrame()
        for idx in range(len(test_res)):
            print('===='+str(idx)+'/'+str(len(test_res))+'====') 
            tid = test_res.iloc[idx]['id']
            gt = [' '.join(test_res.iloc[idx]['summary'])]
            pred_base = [base_res[base_res['id'] == tid]['summary'].values[0]]
            pred_top20 = [top20_res[top20_res['id'] == tid]['summary'].values[0]]
            pred_top50 = [top50_res[top50_res['id'] == tid]['summary'].values[0]]
            pred_super = [super_res[super_res['id'] == tid]['summary'].values[0]]

            score_dict_base = eval(pred_base, gt)
            score_dict_top20 = eval(pred_top20, gt)
            score_dict_top50 = eval(pred_top50, gt)
            score_dict_super = eval(pred_super, gt)

            score_dict_base['id'] = tid
            score_dict_top20['id'] = tid
            score_dict_top50['id'] = tid
            score_dict_super['id'] = tid

            eval_table_base = pd.concat([eval_table_base, pd.DataFrame([score_dict_base])], ignore_index=True)
            eval_table_top20 = pd.concat([eval_table_top20, pd.DataFrame([score_dict_top20])], ignore_index=True)
            eval_table_top50 = pd.concat([eval_table_top50, pd.DataFrame([score_dict_top50])], ignore_index=True)
            eval_table_super = pd.concat([eval_table_super, pd.DataFrame([score_dict_super])], ignore_index=True)

        eval_table_base = add_summary_row(eval_table_base[[eval_table_base.columns[-1]] + list(eval_table_base.columns[:-1])])
        eval_table_top20 = add_summary_row(eval_table_top20[[eval_table_top20.columns[-1]] + list(eval_table_top20.columns[:-1])])
        eval_table_top50 = add_summary_row(eval_table_top50[[eval_table_top50.columns[-1]] + list(eval_table_top50.columns[:-1])])
        eval_table_super = add_summary_row(eval_table_super[[eval_table_super.columns[-1]] + list(eval_table_super.columns[:-1])])

        eval_table_base.to_excel(os.path.join(out_dir, dataset+'BaseLlmEval.xlsx'), index=False)
        eval_table_top20.to_excel(os.path.join(out_dir, dataset+'Top20LlmEval.xlsx'), index=False)
        eval_table_top50.to_excel(os.path.join(out_dir, dataset+'Top50LlmEval.xlsx'), index=False)
        eval_table_super.to_excel(os.path.join(out_dir, dataset+'Supervised50Eval.xlsx'), index=False)


