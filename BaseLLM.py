import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import re
from vllm import LLM, SamplingParams
import torch
import pandas as pd
import json
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Using GPU: {torch.cuda.current_device() if torch.cuda.is_available() else 'None'}")

# module load cuda/11.8
# srun --gres=gpu:1 --mem=72G --cpus-per-task=4 --pty bash
# srun --mem=128G --cpus-per-task=8 --pty bash
# srun --nodelist=c20 --gres=gpu:1 --mem=128G --cpus-per-task=1 --pty bash
# nvidia-smi

if __name__ == '__main__':
    torch.cuda.empty_cache()
    sampling_params = SamplingParams(temperature=0.15, max_tokens=2048)
    # sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
    llm = LLM(model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
              gpu_memory_utilization=0.95,
              # dtype='half',
              max_model_len=8000)
    # llm = LLM(model="google/gemma-3-4b-it", gpu_memory_utilization=0.95)
    # llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-1M",
    #           gpu_memory_utilization=0.95)

    tokenizer = llm.get_tokenizer()
    MAX_PROMPT_TOKENS = 7500

    plos = pd.read_json('../elife/test_textrank50.json')
    print(plos.shape[0])

    batch_size = 1
    count = 0
    for batch_start in range(0, len(plos), batch_size):
        batch_end = min(batch_start + batch_size, len(plos))
        prompts = []
        for index, row in plos[batch_start:batch_end].iterrows():
            top_k = row['top_k']
            comment = ' '.join([sentence for sentence in top_k])
            # section_ls = row['sections']
            # comment = ' '.join([sentence for sublist in section_ls for sentence in sublist])
            comment_tokens = tokenizer.encode(comment, truncation=True, max_length=MAX_PROMPT_TOKENS)
            comment_truncated = tokenizer.decode(comment_tokens, skip_special_tokens=True)
            base = f"""
Below you will see a research paper between `[START]` and `[END]`.
 
Please summarize this research in plain language for a general audience. The summary should have 100-200 words.

Please use plain text without subsections and bullet points.

Focus on the main question, why it matters, what was done, what was found, and what it means—without using technical jargon.

**Text to Summarize:**  
[START] {comment_truncated} [END]
"""

            prompts.append(base)
        try:
            outputs = llm.generate(prompts, sampling_params)
            results = []
            for i, output in enumerate(outputs):
                row = plos.iloc[batch_start + i]
                answer = output.outputs[0].text.strip()
                print(answer)
                result_dict = {
                    "id": row['id'],
                    "summary": answer
                }
                results.append(result_dict)

            with open(f"../test/elife/top_50-{batch_start}.json", "w") as file:
                json.dump(results, file, indent=4)
        except Exception as e:
            print(f"Error processing: {e}")
        count += 1
        if count % 50 == 0:
            print(f"Processing {batch_end} messages")
            torch.cuda.empty_cache()