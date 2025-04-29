from transformers import  pipeline, set_seed
import torch
import pandas as pd
import argparse



MODEL_PATH = "med-rcq/MedQA"
GREEN = '\033[32m'
RED = '\033[31m'
RESET = '\033[0m'


set_seed(42) 

pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    model_kwargs={"torch_dtype": torch.bfloat16},
    trust_remote_code=True,
    do_sample=True,
    temperature=0.01,
    top_k=3,
    device="cuda",  # replace with "mps" to run on a Mac device
)


#If the outcomes are partially significant and results are partially leaning towards agreeing with the question then answer with Maybe.
SYSTEM_PROMPT='''You are a helpful medical assistant. I will give you a context and a question of a study, based on the context you will answer the question by Yes or No or Maybe. In order to answer correctly you will analyze the study findings and results carefully.
If the outcomes are significant or evidences are strong and results are mostly leaning towards agreeing with the question then answer with Yes.
If the outcomes are insignificant or evidences are weak and results are mostly against the question by refuting it then answer with No.
If the outcomes are tie between Yes and No, then answer with Maybe
Think carefully.
Context: _CONTEXT_
Question: _QUESTION_
Final Answer:'''



def generate_ai_decision(prompt):
    """
    Generates a decision based on a given prompt.
    Args:
        prompt (str): The input prompt for the model.
    Returns:
        str: The generated decision.
    """
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages, max_new_tokens=5)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response


# Function to process the CSV file
def process_csv(input_file, output_file):
    """
    Processes a CSV file by generating medical decision for each row. Each row represent a pubmed article and a question that need to be answered either by Yes, No or Maybe
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the processed CSV file with the generated decision
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

    ##reorder columns
    cols_order = ['ID', 'Question', 'Context_with_label', 'LONG_ANSWER', 'final_decision','Medconc_Generated_conclusion']
    df = df[cols_order]
    #df.columns = cols_order

    correct_answer_count=0
    parseerror_count=0
    correct_answer=0


    length = len(df)
    print("Processing "+str(length)+" questions")

    # Loop over each row in the DataFrame
    for index, row in df.iterrows():

        # Extract the relevant columns
        article_id = row['ID']
        question = row['Question']
        final_decision = row['final_decision']
        parse_error=0
        correct_answer=0
        context_string=row['Context_with_label']
        long_answer = row['LONG_ANSWER']
        generated_conclusion=row['Medconc_Generated_conclusion']
        context_string = f"{context_string} Conclusion: {generated_conclusion}"
        formatted_prompt=SYSTEM_PROMPT.replace("_QUESTION_",question)
        formatted_prompt=formatted_prompt.replace("_CONTEXT_",context_string)
        
        df.at[index, 'prompt'] = formatted_prompt


        # Process the question and article using generated conclusion
        model_answer= generate_ai_decision(formatted_prompt)
        print("\n########## INDEX:"+str(index)+" ## QID:"+str(article_id)+" ##########\n")
        print("*****Model Output as is****** ::"+model_answer)
        

        # Do some cleaning for the output. 
        model_answer = model_answer.replace("<|end|>","")
        model_answer = model_answer.lower().strip()

        final_answer = model_answer


        print("Answer:",final_answer," Truth:",final_decision)
        #Check if the generated final decision exist in the model output, if not mark it as a parse error
        if "yes" not in final_answer and "no" not in final_answer and "maybe" not in final_answer and "may" not in final_answer:
          parseerror_count=parseerror_count+1
          parse_error=1
          print(RED+">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Parse error:",final_answer+ RESET)
          
        else:
          if "yes" in final_answer:
            final_answer = "yes"
          elif "no" in final_answer:
            final_answer = "no"
          elif "maybe" in final_answer or "may" in final_answer:
            final_answer = "maybe"
        #Check if final answer by the model match the ground truth (i.e the final_decision)
        if final_answer==final_decision:
          correct_answer_count=correct_answer_count+1
          correct_answer=1
        elif final_answer.startswith(final_decision):
          correct_answer_count=correct_answer_count+1
          correct_answer=1
        elif final_decision in final_answer:
          correct_answer_count=correct_answer_count+1
          correct_answer=1
        #If correct answer is not found, 
        if correct_answer==0:
          print(RED+"Question:"+question+RESET)
          print(RED+"Generated Conclusion:"+generated_conclusion+RESET)
          print(RED+"Actual Conclusion:"+long_answer+RESET)
          print(RED+"------------------")

        # Update the LONG_ANSWER with the processed output
        df.at[index, 'ai_answer'] = final_answer
        #df.at[index, 'final_decision'] = final_decision
        df.at[index, 'is_correct'] = correct_answer
        df.at[index, 'parse_error'] = parse_error

        acc= (100*correct_answer_count)/(index+1)
        print(GREEN+"Accuracy now is "+str(round(acc,2))+"%"+RESET)



    # Write the modified DataFrame to a new CSV file
    acc= (100*correct_answer_count)/length
    df.to_csv(output_file, index=False)
    #print Accuracy and parse error numbers
    print("*******")
    print("Number of Correct prediction is "+str(correct_answer_count)+" out of total "+str(length))
    print("\033[34mAccuracy is "+str(round(acc,2))+"%")
    print("Count of wrong parsing "+str(parseerror_count))
    confusion_matrix = pd.crosstab(df['final_decision'], df['ai_answer'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)



if __name__ == "__main__":
    #input_file ='testset_w_genecon.csv'
    
    parser = argparse.ArgumentParser(description="Process a CSV file to make decisions.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to save the processed CSV file")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    # Process the file
    process_csv(input_file, output_file)
  

