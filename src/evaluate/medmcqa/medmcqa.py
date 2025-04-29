
import json
import openai  # install openai package
from typing import Dict
import re
import csv
import os
import time

# !pip install openai==0.28

# Set your OpenAI API key
#openai.api_key = "YOUR_OPENAI_API_KEY"

def ask_gpt4_mini(context, question, options):
    # Format the options
    option_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

    # Prompt design
    prompt = f"""You are a medical assistant answering multiple-choice questions. Read the context and question carefully, and answer by selecting the **letter** of the best option (A, B, C, or D). Just return the letter, nothing else.

Context:
{context}

Question:
{question}

Options:
{option_str}

Answer:"""

    # Call GPT-4.1 nano
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        reply = response["choices"][0]["message"]["content"].strip()
        # Extract only the letter (A, B, C, D)
        letter = reply[0].upper()
        if letter in options:
            return letter
        else:
            # fallback: try to match based on content
            for k, v in options.items():
                if v.strip().lower() in reply.lower():
                    return k
            return "?"  # can't parse
    except Exception as e:
        print("Error from GPT:", e)
        return "?"

# Initialize counters and CSV
label_correct = 0
total = 0
file_path="medmcqa_testset.jsonl"
csv_output = "medmcqa_results.csv"
with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "question", "model_answer_letter",
        "correct_letter", "correct_text", "label_correct"
    ])
    writer.writeheader()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            context = item["exp"]
            generatedConclusion =item["generated_conclusion"]
            #comment the following line if you want to test without the generated conclusion
            context = context+" "+generatedConclusion

            question = item["question"]
            options = {"A": item['opa'].strip().lower(),
                "B": item['opb'].strip().lower(),
                "C": item['opc'].strip().lower(),
                "D": item['opd'].strip().lower()}
            options_list = list(options.items())
            #selected answer
            cop = item["cop"]
            correct_letter, correct_text = options_list[cop-1]
            model_answer_letter = ask_gpt4_mini(context, question, options)
            predicted_text = options.get(model_answer_letter, "").strip().lower()

            is_label_correct = model_answer_letter == correct_letter
            if is_label_correct:
                label_correct += 1
            total += 1
            print(f"Q: {question}")
            print(f"Model Chose: {model_answer_letter}: {predicted_text}")
            print(f"Correct Answer: {correct_letter}: {correct_text}")
            if is_label_correct:
                print("Correct Answer\n")
            else:
                print("Wrong answer\n")

            writer.writerow({
    "question": question,
    "model_answer_letter": model_answer_letter,
    "correct_letter": correct_letter,
    "correct_text": correct_text,
    "label_correct": int(is_label_correct)

})

# Summary
print(f"\nTotal Questions: {total}")
print(f"Label Accuracy: {label_correct}/{total} = {label_correct/total:.2%}")
print(f"Detailed log saved to: {csv_output}")