from transformers import pipeline,set_seed
import torch
import pandas as pd
import argparse

MODEL_PATH="med-rcq/MedConclusion"

set_seed(42)
SYSTEM_PROMPT='''You are a helpful medical assistant. Write a conclusion for the following article:\n
Title: _TITLE_
_CONTEXT_
Conclusion:'''

pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    model_kwargs={"torch_dtype": torch.bfloat16},
    trust_remote_code=True,
    do_sample=True,
    temperature=0.01,
    device="cuda",  # replace with "mps" to run on a Mac device
)


def generate_ai_conclusion(prompt):
    """
    Generates medical conclusion based on a given prompt.
    Args:
        prompt (str): The input prompt for the model.
    Returns:
        str: The generated conclusion.
    """
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages, max_new_tokens=250)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    
    return assistant_response

# Function to process the CSV file
def process_csv(input_file, output_file):
    """
    Processes a CSV file by generating medical conclusions for each row. Each row represent a pubmed article. 
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the processed CSV file with the generated conclusion
    """
    
    # Read the input CSV file into a DataFrame
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return

    cols_order = ['ID', 'Question', 'Context_with_label','LONG_ANSWER','final_decision']
    df = df[cols_order]
 

    # Loop over each row in the DataFrame
    for index, row in df.iterrows():

        # Extract the relevant columns
        article_id = row['ID']
        # Pubmed article title is in a Question form
        title = row['Question']
        # context_string represent the Pubmed article without the conclusion or the title. It include the labels like "background", "Methods"...etc
        context_string=row['Context_with_label']
        # long answer represent Pubmed article conclusion section 
        long_answer = row['LONG_ANSWER']
        # The final decision is either yes or no or maybe, it depends on what both annotators agreed on
        final_decision = row['final_decision']
        print("\n########## INDEX:"+str(index)+" ## QID:"+str(article_id)+" ##########\n")
        #Prepare system prompt
        formatted_prompt=SYSTEM_PROMPT.replace("_TITLE_",title)
        formatted_prompt=formatted_prompt.replace("_CONTEXT_",context_string)
        # Call the LLM model to generate the conclusion using the constructed prompt
        generated_conclusion = generate_ai_conclusion(formatted_prompt)
        print(generated_conclusion)
        #Save the generated conclusion
        df.at[index, 'Medconc_Generated_conclusion'] = generated_conclusion



    # Save output in CSV file
    df.to_csv(output_file, index=False)
    



if __name__ == "__main__":
    
    # output_file 
    parser = argparse.ArgumentParser(description="Process a CSV file to generate conclusions.")
    # input_file = 'pubmedqa_testset.csv'
    parser.add_argument("input_file", help="Path to the input CSV file")
    # write the name of the output file
    parser.add_argument("output_file", help="Path to save the processed CSV file")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    # Process the file
    process_csv(input_file, output_file)
