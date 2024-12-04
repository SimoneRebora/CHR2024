#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 18 2024
@author: Simone Rebora
Note: before you start, remember to setup OpenAI API key:
https://platform.openai.com/docs/quickstart?context=python
"""

my_prompt = 'complex_binary'
my_model = 'gpt-4o-2024-05-13'
train_reviews = 8

import pandas as pd
from openai import OpenAI
from collections import Counter
client = OpenAI()

def label_maker_binary(x):
  if x != 'no_val':
    return 'val'
  else:
    return x

# read system prompt
with open('System_prompts/'+my_prompt+'.txt', 'r') as file:
  system_prompt = file.read()

# read full dataset
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

# Prepare train reviews
train_df = pd.read_excel('Curation/GPT_train.xlsx')
train_ids = train_df['ID'].drop_duplicates().tolist()

# Select reviews with best distribution
general_stats = Counter(train_df['label'].map(label_maker_binary))
general_prop = general_stats['val']/general_stats['no_val']
reviews_prop = []

for my_rev_id in train_ids:

    selected_review = my_rev_id

    # Boolean mask
    mask = train_df['ID'] == selected_review

    # Extract portion of dataframe
    extracted_df = train_df[mask]
    if ("binary" in my_prompt):
        extracted_df['label'] = extracted_df['label'].map(label_maker_binary)

    review_stats = Counter(extracted_df['label'])
    reviews_prop.append(review_stats['val']/review_stats['no_val'])

train_ids_scores = [abs(number - general_prop) for number in reviews_prop]

train_ids = [id for id, score in sorted(zip(train_ids, train_ids_scores), key=lambda x: x[1])]

train_ids = train_ids[:train_reviews]

# save train reviews (and annotations) to single .csv files
for my_rev_id in train_ids:

    print(my_rev_id)

    selected_review = my_rev_id

    # Boolean mask
    mask = train_df['ID'] == selected_review

    # Extract portion of dataframe
    extracted_df = train_df[mask]

    # Select the specified columns
    extracted_df = extracted_df[['book_title', 'sent_id', 'sentence']]

    # write review to separated csv
    extracted_df.to_csv("GPT_datasets/GPT_train_"+str(my_rev_id)+".csv", index=False)

    # write ground truth to separated csv
    extracted_df = train_df[mask]
    extracted_df = extracted_df[['sent_id', 'label']]
    if("binary" in my_prompt):
        extracted_df['label'] = extracted_df['label'].map(label_maker_binary)
    extracted_df.to_csv("GPT_datasets/GPT_train_" + str(my_rev_id) + "_GT.csv", index=False)

    review_stats = Counter(extracted_df['label'])
    print(len(extracted_df['label']), "sentences")
    print("score:", (review_stats['val'] / review_stats['no_val'])-general_prop)

# prepare prompt
my_prompt_fewshot = [{"role": "system", "content": system_prompt}]

# increase prompt
for my_rev_id in train_ids:

    with open("GPT_datasets/GPT_train_"+str(my_rev_id)+".csv", "r") as file:
        my_review = file.read()
    my_prompt_fewshot.append({"role": "user", "content": my_review})

    with open("GPT_datasets/GPT_train_"+str(my_rev_id)+"_GT.csv", "r") as file:
        my_annotations = file.read()
    my_prompt_fewshot.append({"role": "assistant", "content": my_annotations})

# process each review with GPT4
for my_rev_id in all_ids:

    print("Processing review", my_rev_id, "...")

    # read review
    with open("GPT_datasets/test_"+str(my_rev_id)+".csv", "r") as file:
        my_review = file.read()

    # prepare prompt
    my_prompt_request = my_prompt_fewshot.copy()
    my_prompt_request.append({"role": "user", "content": my_review})

    # get LLM answer
    completion = client.chat.completions.create(
        model=my_model,
        temperature=0,
        messages=my_prompt_request
      )

    # write result
    with open("GPT_datasets/fewshot_"+str(train_reviews)+"_test_prompt_"+my_prompt+"_"+str(my_rev_id)+".csv", "w") as file:
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

    my_file = "GPT_datasets/fewshot_"+str(train_reviews)+"_test_prompt_"+my_prompt+"_"+str(my_rev_id)+".csv"
    my_file2 = "GPT_datasets/fewshot_"+str(train_reviews)+"_test_prompt_" + my_prompt + "_" + str(my_rev_id) + "_CLEAN.csv"
    process_file(my_file, my_file2)


# join the results to a single Excel file
# Initialize an empty list to store the DataFrames
dfs = []

# Iterate over each CSV file and read it into a DataFrame, then append to the list
for my_rev_id in all_ids:

    my_file = "GPT_datasets/fewshot_"+str(train_reviews)+"_test_prompt_"+my_prompt+"_"+str(my_rev_id)+"_CLEAN.csv"
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

GPT_df.to_excel("GPT_results/fewshot_"+str(train_reviews)+"_"+my_round+"_GPT4_prompt_"+my_prompt+".xlsx")

