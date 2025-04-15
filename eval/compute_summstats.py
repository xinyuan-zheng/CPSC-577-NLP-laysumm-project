import pandas as pd
import numpy as np

csv_file = "/data22/user/xz583/project/nlp/NLP_project-main/eval/plosTopkLlmEval.csv"
df = pd.read_csv(csv_file)


numeric_cols = df.select_dtypes(include=[np.number]).columns
mean_row = df[numeric_cols].mean()
std_row = df[numeric_cols].std()
summary_row = [f"{mean:.3f} ± {std:.3f}" for mean, std in zip(mean_row, std_row)]
summary_df = pd.DataFrame([summary_row + ["mean ± std"]], columns=numeric_cols.tolist() + ["id"])
final_df = pd.concat([df, summary_df], ignore_index=True)
output_file = "/data22/user/xz583/project/nlp/NLP_project-main/eval/plosTopkLlmEval.xlsx"
final_df.to_excel(output_file, index=False)