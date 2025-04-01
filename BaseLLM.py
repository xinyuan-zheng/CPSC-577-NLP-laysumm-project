import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
# srun --nodelist=c17 --gres=gpu:1 --mem=72G --cpus-per-task=1 --pty bash
# nvidia-smi

if __name__ == '__main__':
    torch.cuda.empty_cache()
    sampling_params = SamplingParams(temperature=0.15, max_tokens=2048)
    # llm = LLM(model="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    # llm = LLM(model="google/gemma-3-12b-it")
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-1M")

    plos = pd.read_json('../plos/test.json')
    print(plos.shape[0])

    batch_size = 1
    count = 0
    for batch_start in range(0, 5, batch_size):
        batch_end = min(batch_start + batch_size, len(plos))
        prompts = []
        for index, row in plos[batch_start:batch_end].iterrows():
            section_ls = row['sections']
            comment = ' '.join([sentence for sublist in section_ls for sentence in sublist])
            base = f"""
Below you will see a research paper between `[START]` and `[END]`.
 
Please summarize this research in plain language for a general audience. The summary should have about 200 words.

Focus on the main question, why it matters, what was done, what was found, and what it means—without using technical jargon.

**Text to Summarize:**  
[START] {comment} [END]
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
                    "id": int(row['id']),
                    "summary": answer
                }
                results.append(result_dict)

            with open(f"../test/plos/{batch_start}.json", "w") as file:
                json.dump(results, file, indent=4)
        except Exception as e:
            print(f"Error processing: {e}")
        count += 1
        if count % 50 == 0:
            print(f"Processing {batch_end} messages")
            torch.cuda.empty_cache()



