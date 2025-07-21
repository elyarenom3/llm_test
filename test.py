import json
import math
import random
import requests
import time
import base64
import os
from collections import Counter
from dotenv import load_dotenv
from itertools import combinations

load_dotenv()

API_PW = os.getenv('API_PW')
UID = os.getenv('UID')
MODEL_NAME = os.getenv('MODEL_NAME')
URL = 'https://gmlabllm.dev.echonet/v1/completions'
PROXIES = {'http': None, 'https': None}
CALLER_ENDPOINT = ''
API_USE_CASE = 'LAB'

def write_results_to_file(fewshots, accuracy, output_file="best_fewshots_results.txt"):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Best Fewshot Examples:\n\n")
            for fs in fewshots:
                f.write(fs + "\n---\n")
            f.write(f"\nBest Accuracy: {accuracy:.2%}\n")
        print(f"Results successfully written to {output_file}")
    except Exception as e:
        print(f"Failed to write results: {e}")


def request_llm(input_text, criteria, fewshots, temp=0.8):
    fewshots_str = "\n".join(fewshots)
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant that uses the following criteria and fewshots to judge which summary is better and answers with just the letter of the summary, no punctuation. Make sure your answer is the one that adheres closest to the criteria and fewshots: {criteria}.
Few-shot examples:\n{fewshots_str}
<|eot_id|><|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    params = {
        "model": MODEL_NAME,
        "prompt": [prompt],
        "logprobs": 5,
        "temperature": temp,
        "max_tokens": 5,
        "stream": False
    }

    headers = {
        'authorization': f"Basic {base64.b64encode(f'{UID}:{API_PW}'.encode('utf-8')).decode('utf-8')}",
        'caller-endpoint': CALLER_ENDPOINT,
        'api-use-case': API_USE_CASE,
        'uid': UID,
    }

    for attempt in range(3):
        try:
            response = requests.post(URL, json=params, verify=False, proxies=PROXIES, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}. Retrying in 10 seconds...")
            time.sleep(10)
    raise Exception("Failed to get a response from the LLM API after 3 attempts.")

def evaluate_accuracy(fewshots, eval_data, criteria):
    correct = 0
    for example in eval_data:
        input_text = example['question']
        correct_answer = example['answer']
        try:
            result = request_llm(input_text, criteria, fewshots)
            model_answer = result['choices'][0]['text'].strip()
            if model_answer == correct_answer:
                correct += 1
        except Exception as e:
            print(f"Error during evaluation: {e}")
    return correct / len(eval_data)

def greedy_fewshot_selector(data, criteria, k=5):
    selected = []
    remaining = data[:]
    best_score = 0

    for _ in range(k):
        best_example = None
        for candidate in remaining:
            candidate_text = candidate['question']
            candidate_answer = candidate['answer']
            fewshot_str = f"{candidate_text}\nAnswer: {candidate_answer}"
            temp_fewshots = selected + [fewshot_str]
            eval_set = [e for e in data if f"{e['question']}\nAnswer: {e['answer']}" not in temp_fewshots]
            acc = evaluate_accuracy(temp_fewshots, eval_set, criteria)
            if acc > best_score:
                best_score = acc
                best_example = candidate
        if best_example:
            selected.append(f"{best_example['question']}\nAnswer: {best_example['answer']}")
            remaining.remove(best_example)

    return selected, best_score

def main():
    try:
        json_file_path = 'data/example_data/single_stock/filtered_pairs/prompt_vs_original.json'
        prompt_path = 'results/op6.txt'

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(prompt_path, 'r', encoding='utf-8') as f:
            criteria = f.read()

        best_fewshots, best_accuracy = greedy_fewshot_selector(data, criteria, k=5)

        write_results_to_file(best_fewshots, best_accuracy, output_file="best_fewshots_results.txt")

    except Exception as e:
        print(f"Error in execution: {e}")

    except Exception as e:
        print(f"Error in execution: {e}")

main()
