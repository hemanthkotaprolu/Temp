import os
import csv
import json
import torch
import ast
import argparse

import datetime
import pandas as pd
from pytz import timezone

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline

now = datetime.datetime.now(timezone("Asia/Kolkata"))
timestamp = now.strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--cot", action="store_true", default=False)
args = parser.parse_args()

MODEL_NAME=args.model_name
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
FILE_NAME=MODEL_NAME.split("/")[-1]
cot_str="_cot" if args.cot else ""
EVAL_FILE_NAME = f"{FILE_NAME}{cot_str}_{timestamp}"
NUM_iter=3

EMOTION_LABELS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'negative', 'positive']

EVAL_COLUMNS = ["Scenario", "Subject", "Feeling"]

for e in EMOTION_LABELS:
    for i in range(1, 4):
        EVAL_COLUMNS.append(f"{e}_Response_{i}")

print(EVAL_COLUMNS)

def write_res(d):
    ver = "data"
    cot = ""
    file_dir = "./results/"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_name = f"{file_dir}/{EVAL_FILE_NAME}.csv"
    
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            # writer.writerow(columns["EU"])
            writer.writerow(EVAL_COLUMNS)
    
    with open(file_name, "a") as f:
        writer = csv.writer(f)
        writer.writerow(d)


if args.cot:
    print("Doing Chain-of-Thought Prompting")
    SYSTEM_TEMPLATE="""You are an expert in emotional analysis and natural language processing. Your task is to answer whether the subject might feel the particular emotion with yes or no, yes if the subject experience the emotion, no if the subject doesn't experience the emotion. Think step by step to arrive at the final answer. In your response, first provide a very brief reasoning, then provide your final answer between <answer></answer>. For example if your answer is no then write <answer>no</answer>, if it's an yes <answer>yes</answer>.
    
Consider the following when answering:
- You must read and understand the scenario very carefully in the persepective of the subject.
- Pay close attention to the actions, tone, and imagery mentioned in the scenario.
- Your output should contain a brief reasoning for your answer and the final answer between <answer></answer> tag."""
    USER_TEMPLATE="""**Scenario**:
{scenario}

**Subject**:
{subject}

**Questions**:
Does {subject} feel {emotion}?

Based on the above-provided information, answer the question and return only yes or no. Your output should only contain yes or no with no other content. Please note!!! Your output should contain a very brief reasoning to your answer and final answer wrapped <answer></answer> tags."""
else:
    SYSTEM_TEMPLATE="""You are an expert in emotional analysis and natural language processing. Your task is to answer whether the subject might feel the particular emotion with yes or no, yes if the subject experience the emotion, no if the subject doesn't experience the emotion. You just need to say yes or no, indicating whether subject experience the emotion or not.

Consider the following when answering:
    - You must read and understand the scenario very carefully in the persepective of the subject.
    - Pay close attention to the actions, tone, and imagery mentioned in the scenario.
    - Your output should just be yes or no.
    - **Do not** explain your answer."""
    
    USER_TEMPLATE="""**Scenario**:
{scenario}
    
**Subject**:
{subject}
    
**Question**:
Does {subject} feel {emotion}?
    
Based on the above-provided information, answer the question and return only yes or no. Your output should only contain yes or no with no other content."""

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir="./models", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model.to(DEVICE)
model.generation_config.pad_token_id = tokenizer.pad_token_id

generation_args = {
    "max_new_tokens": 200 if args.cot else 50,
    "return_full_text": False,
    "temperature": 0.5,
    "do_sample": True,
}

pipeline =pipeline(
   "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    pad_token_id=tokenizer.eos_token_id
)




data = pd.read_csv("./temp.csv")

for instance in tqdm(range(data.shape[0])):
# for _, row in tqdm(data.iterrows()):
    row = data.iloc[instance]
    scenario = row['text']
    subject = ast.literal_eval(row['subject'])[0]

    user_messages = []
    for emotion in EMOTION_LABELS:
        user_messages.append(USER_TEMPLATE.format(scenario=scenario, subject=subject, emotion=emotion))

    row = [scenario, subject, row["emotion"]]

    for user_message in user_messages:
        if "gemma" in MODEL_NAME:
            messages = [
                {"role": "user", "content": f"{SYSTEM_TEMPLATE}\n\n{user_message}"}
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": user_message},
            ]
        for itr in range(NUM_iter):
            response = pipeline(messages, **generation_args)[0]["generated_text"]
            row += [response]
    
    write_res(row)
