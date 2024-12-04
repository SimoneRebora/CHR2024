#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 25 18:06:00 2024
@author: Simone Rebora
Note: before you start, remember to setup OpenAI API key:
https://platform.openai.com/docs/quickstart?context=python
"""

my_prompt = 'procedural_binary'
my_model = 'gpt-4o-2024-05-13'

import os
import pandas as pd
from openai import OpenAI
client = OpenAI()

# read system prompt
with open('System_prompts/'+my_prompt+'.txt', 'r') as file:
  system_prompt = file.read()

# read full test dataset
full_df = pd.read_excel('Curation/GPT_test.xlsx')
all_ids = full_df['ID'].drop_duplicates().tolist()

# save reviews to single .csv files
for my_rev_id in all_ids:

    selected_review = my_rev_id

    # Boolean mask
    mask = full_df['ID'] == selected_review

    # Extract portion of dataframe
    extracted_df = full_df[mask]

    # Select the specified columns
    extracted_df = extracted_df[['book_title', 'sent_id', 'sentence']]

    # write review to separated csv
    extracted_df.to_csv("GPT_datasets/test_"+str(my_rev_id)+".csv", index=False)

# process each review with GPT4
for my_rev_id in all_ids:

    print("Processing review", my_rev_id, "...")

    # read review
    with open("GPT_datasets/test_"+str(my_rev_id)+".csv", "r") as file:
        my_review = file.read()

    # get LLM answer
    completion = client.chat.completions.create(
        model=my_model,
        temperature=0,
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": my_review}
        ]
      )

    # write result
    with open("GPT_datasets/test_prompt_"+my_prompt+"_"+str(my_rev_id)+".csv", "w") as file:
      my_review = file.write(completion.choices[0].message.content)

    print("Done")


# delete first and last line if "```plaintext" appears
def process_file(input_filename, output_filename):
    with open(input_filename, 'r') as infile:
        lines = infile.readlines()

    inside_csv_block = False
    csv_lines = []

    for line in lines:
        if line.strip() == '```csv':
            inside_csv_block = True
            continue
        elif line.strip() == '```' and inside_csv_block:
            inside_csv_block = False
            continue

        if inside_csv_block:
            csv_lines.append(line)

    # If no "```csv" found, write original lines, otherwise write csv_lines
    output_lines = csv_lines if csv_lines else lines

    with open(output_filename, 'w') as outfile:
        outfile.writelines(output_lines)


for my_rev_id in all_ids:

    my_file = "GPT_datasets/test_prompt_"+my_prompt+"_"+str(my_rev_id)+".csv"
    my_file2 = "GPT_datasets/test_prompt_" + my_prompt + "_" + str(my_rev_id) + "_CLEAN.csv"
    process_file(my_file, my_file2)


# join the results to a single Excel file
# Initialize an empty list to store the DataFrames
dfs = []

# Iterate over each CSV file and read it into a DataFrame, then append to the list
for my_rev_id in all_ids:

    my_file = "GPT_datasets/test_prompt_"+my_prompt+"_"+str(my_rev_id)+"_CLEAN.csv"
    df = pd.read_csv(my_file)
    if "label" not in df.columns:
        print("Missing label column in", my_file)
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# verify that there is no mismatch
if final_df['sent_id'].equals(full_df['sent_id']):
    print("No mismatch found.")
else:
    print("Warning: possible mismatch!")

# create GPT df
GPT_df = pd.DataFrame({
    'rev_id': full_df['ID'],
    'book_title': full_df['book_title'],
    'sent_id': full_df['sent_id'],
    'sentence': full_df['sentence'],
    'label': final_df['label'],
})

GPT_df.to_excel("GPT_results/GPT_test_GPT4_prompt_"+my_prompt+".xlsx")

