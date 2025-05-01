# Med-RCQ (Medical Reasoning by Concluding and Questioning)

Welcome to the GitHub page of Med-RCQ (Medical Reasoning by Concluding and Questioning), an LLM-based method for analyzing health and biomedical information with the objective of supporting informed decisions. It is composed of two [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) fine-tuned models:

- [MedConclusion](https://huggingface.co/med-rcq/MedConclusion): An LLM fine-tuned for reasoning by processing medical literature and generating conclusions.

- [MedQA](https://huggingface.co/med-rcq/MedQA): An LLM fine-tuned for decision-making by answering medical questions with either yes, no, or maybe.

This repository contains testing code to evaluate Med-RCQ using medical benchmarks, along with the testing dataset. The training dataset used can be found on Hugging Face under [`med-rcq/med-rcq-dataset`]([https://github.com/Teddy-XiongGZ/MedRAG](https://huggingface.co/datasets/med-rcq/med-rcq-dataset/tree/main)) 
### Prompt Templates
All prompts used during training and evaluation are documented in [`src/prompts/template.py`](src/prompts/template.py).

## Table of Contents

- [System requirements and Setup](#system-requirements-and-setup)
- [Evaluation Results](#evaluation-results)
- [PubMedQA Evaluation](#pubmedqa-evaluation)
- [BioASQ Evaluation](#bioasq-evaluation)
- [MedQA-US Evaluation](#medqa-us-evaluation)
- [MedMCQA Evaluation](#medmcqa-evaluation)
- [MMLU Evaluation](#mmlu-evaluation)

## System requirements and setup

- System used:  
    \- GPU: RTX A6000 or A40  
    \- OS: Ubuntu 22.04.3
- Install the following dependencies on your system:
```
curl -O <https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh>; /bin/bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p /opt/conda; source ~/.bashrc; export PATH=/opt/conda/bin:$PATH; source /opt/conda/bin/activate; conda create -n medrcq_env python=3.11.7 -y; conda activate medrcq_env; pip install torch==2.5.1 transformers==4.48.0 pandas==2.1.4;pip install flash-attn==2.7.3
```
## Evaluation Results

Below is a summary of results using the Med-RCQ models to generate a conclusion with MedConclusion, then reason over it using MedQA.
| Dataset      | Accuracy    | Size     | Settings |Questions Type | Content Type
|:----------|:---------:|:----------:|:----------:|:----------|:----------:|
| PubMedQA      | 81%    |    500   | Reason-required Settings|Multi Choice: Yes/No/Maybe|Medical Literature |
| BioASQ Task 12b | 90.2% |    100   | Reason over top 10 snippets|Multi Choice: Yes/No|Medical Literature |

Below is a summary of results obtained by utilizing GPT-4.1 Nano to reason over different medical use case contexts, with and without MedConclusion-generated conclusions.

| Dataset      | Accuracy    | Size     | Settings |Questions Type | Content Type
|:----------|:---------:|:----------:|:----------:|:----------|:----------:|
| MedQA-US | **72.11%**     |    1,273   | With MedConclusion|Multi Choice: A/B/C/D|Medical Use Case |
| MedQA-US | 67.64%     |    1,273   |Without MedConclusion|Multi Choice: A/B/C/D|Medical Use Case |
| MedMCQA | **86.46%**      |    1,041   | With MedConclusion|Multi Choice: A/B/C/D|Medical Information |
| MedMCQA | 85.30%      |    1,041   |Without MedConclusion|Multi Choice: A/B/C/D|Medical Information |
| MMLU-professional_medicine | **85.66%** | 272   | With MedConclusion|Multi Choice: A/B/C/D|Medical Use Case |
| MMLU-professional_medicine | 84.56% | 272   |Without MedConclusion|Multi Choice: A/B/C/D|Medical Use Case |
## PubMedQA Evaluation

The PubMedQA dataset is composed of 500 records for testing Yes/No/Maybe questions. The original dataset is available [here](https://github.com/pubmedqa/pubmedqa)
An example of the format for one record in the PubMedQA dataset is as follows:
```json
   "18239988": {
        "QUESTION": "Differentiation of nonalcoholic from alcoholic steatohepatitis: are routine laboratory markers useful?",
        "CONTEXTS": [
            "Specific markers for differentiation of nonalcoholic (NASH) from alcoholic steatohepatitis (ASH) are lacking. We investigated the role of routine laboratory parameters in distinguishing NASH from ASH.",
            "Liver biopsies performed at our hospital over a 10-year period were reviewed, 95 patients with steatohepatitis identified and their data prior to biopsy reevaluated. The diagnosis NASH or ASH was assigned (other liver diseases excluded) on the basis of the biopsy and history of alcohol consumption (<140 g/week). Logistic regression models were used for analysis.",
            "NASH was diagnosed in 58 patients (61%; 30 f) and ASH in 37 (39%; 9 f). High-grade fibrosis (59% vs. 19%, P<0.0001) and an AST/ALT ratio>1 (54.1% vs 20.7%, P = 0.0008) were more common in ASH. The MCV was elevated in 53% of ASH patients and normal in all NASH patients (P<0.0001). Multivariate analysis identified the MCV (P = 0.0013), the AST/ALT ratio (P = 0.011) and sex (P = 0.0029) as relevant regressors (aROC = 0.92). The AST/ALT ratio (P<0.0001) and age (P = 0.00049) were independent predictors of high-grade fibrosis. Differences in MCV were more marked in high-grade fibrosis."
        ],
        "LABELS": [
            "AIMS",
            "METHODS",
            "RESULTS"
        ],
        "MESHES": [
            "Adolescent",
            "Adult",
            "Aged",
            "Alanine Transaminase",
            "Aspartate Aminotransferases",
            "Biomarkers",
            "Biopsy",
            "Diagnosis, Differential",
            "Erythrocyte Indices",
            "Fatty Liver",
            "Fatty Liver, Alcoholic",
            "Female",
            "Humans",
            "Liver",
            "Liver Cirrhosis",
            "Liver Cirrhosis, Alcoholic",
            "Liver Function Tests",
            "Male",
            "Middle Aged",
            "Predictive Value of Tests",
            "Retrospective Studies"
        ],
        "YEAR": "2008",
        "reasoning_required_pred": "yes",
        "reasoning_free_pred": "no",
        "final_decision": "yes",
        "LONG_ANSWER": "Higher MCVs and AST/ALT ratios in ASH reflect the severity of underlying liver disease and do not differentiate NASH from ASH. Instead, these biomarkers might prove useful in guiding selection of patients for liver biopsy and in targeting therapy."
    },
```
To execute the evalution using Med-RCQ follow the following steps:
1- Navigate to src directory, each directory contain the benchmark and the correspond dataset to evaluate. For pubmed directory, it will cover PubMedQA and BioASQ.
2- Using the activated conda environment **medrcq_env** run the following command:
  ```
  python generate_conclusion.py pubmedqa_testset.csv g_conc_out.csv
  ```
3- After generating the conclusion use the generated file (i.e. g_conc_out.csv) to answer PubMedQA questions:
  ```
  python make_decision.py g_conc_out.csv g_qa_out.csv
  ```

## BioASQ Evaluation

The BioASQ Task12b dataset is composed of 100 records for testing Yes/No questions. The original dataset is available [here](http://participants-area.bioasq.org/datasets/).  

An example of the format for one record in the BioASQ dataset is as follows:
```json
{
"body": "Is there an approved vaccine against Helicobacter pylori?",
"ideal_answer": [
"No. There is currently no approved vaccine against Helicobacter pylori.",
"No, there is currently no approved vaccine for Helicobacter pylori.",
"No, there is currently no approved vaccine against Helicobacter pylori.",
"No, there is no approved vaccine against Helicobacter pylori",
"No, there is no approved vaccine against Helicobacter pylori.",
"No, there is not an approved vaccine against Helicobacter pylori."
],
"exact_answer": "no",
"id": "65f86a90c4010b4d78000057",
"snippets": [
"H. pylori, the development of an efficacious vaccine is a valid option to protect from disease or infection and ultimately prevent gastric cancer. However, despite more than 30 years of research, no vaccine has entered the market yet",
"advances of H. pylori vaccines from two aspects, candidates of antigens and adjuvants, to provide references for the development of vaccine against this bacterium",
"Despite an extensive list of attempts to develop a vaccine, no approved vaccine against H. pylori is available.",
"Unfortunately, no vaccine against H. pylori is currently licensed, and protective immunity mechanisms against H. pylori are only partially understood.",
"After the discovery of Helicobacter pylori (H. pylori), and the evidence of its relationship with gastric diseases, antibiotic-based therapies were developed, which efficacy was however limited by antibiotic resistance and lack of patient compliance. A vaccine would overcome these drawbacks, but currently there is not any H. pylori vaccine licensed.",
"An effective vaccine against H pylori is years away.",
"but currently there is not any H. pylori vaccine licensed.",
"A vaccine would overcome these drawbacks, but currently there is not any H. pylori vaccine licensed."
]
}
``` 

The dataset has been reformatted into CSV and is named bioasq_testset.csv. This test set includes the following fields:

1. **question**: Contains the value from the original _body_ field.
2. **context**: Represents the first 10 snippets.
3. **final_decision**: Represents the exact answer (i.e., _Yes_ or _No_).

To run the testing, similar to PubMedQA, the pipeline consists of two steps. First, invoke the MedConclusion model to reason over the article and generate conclusion, second process the output with MedQA to make the decision:
```python
python generate_conclusion.py bioasq_testset.csv bioasq_conc_out.csv

python make_decision.py bioasq_conc_out.csv bioasq_qa_out.csv  
```
## MedQA-US Evaluation

The MedQA-US dataset is composed of 1,273 records for testing and is available [here](https://github.com/jind11/MedQA).  
  
The dataset has been reformatted into JSONL and is named medqa_testset.jsonl. This test set includes the following fields:

1. **Title**: Represents a short title for the use case (i.e., question). The title is generated using GPT-4.1 nano.
2. **generated_conclusion**: A conclusion generated using the MedConclusion model.
3. **context** and **finalq**: The original question is split into two fields—context (the background information) and finalq (the final question)—to better structure the prompt using context + conclusion.
An example of the format for one record in the MedQA dataset is as follows:
```json

{
"question": "An otherwise healthy 50-year-old man comes to the physician because of a 6-month history of increasingly frequent episodes of upper abdominal pain, nausea, vomiting, and diarrhea. He has had a 3.2-kg (7-lb) weight loss during this time. Physical examination shows bilateral pitting pedal edema. An endoscopy shows prominent rugae in the gastric fundus. Biopsy shows parietal cell atrophy. Which of the following is the most likely underlying cause?", "answer": "Proliferation of gastric mucus-producing cells",
"options": {"A": "Serotonin-secreting gastric tumor", "B": "Proliferation of gastric mucus-producing cells", "C": "Excessive somatostatin secretion", "D": "Ectopic secretion of gastrin"},
"meta_info": "step1", "answer_idx": "B",
"metamap_phrases": ["healthy 50 year old man", "physician", "of", "month history", "frequent episodes", "upper abdominal pain", "nausea", "vomiting", "diarrhea", "3.2 kg", "weight loss", "time", "Physical examination shows bilateral pitting pedal edema", "endoscopy shows prominent rugae", "gastric", "Biopsy shows parietal cell atrophy", "following", "most likely underlying cause"],
"context": "An otherwise healthy 50-year-old man comes to the physician because of a 6-month history of increasingly frequent episodes of upper abdominal pain, nausea, vomiting, and diarrhea. He has had a 3.2-kg (7-lb) weight loss during this time. Physical examination shows bilateral pitting pedal edema. An endoscopy shows prominent rugae in the gastric fundus. Biopsy shows parietal cell atrophy.",
"finalq": "Which of the following is the most likely underlying cause?",
"title": "Routine Evaluation with Underlying Clue",
"generated_conclusion": "This patient has pernicious anemia, a condition in which the body is unable to absorb vitamin B12. The patient's symptoms are due to the vitamin B12 deficiency."
}
```

To run the test: 
```python
python medqa.py
``` 
## MedMCQA Evaluation

The MedMCQA dataset is composed of 4,183 records for testing and is available [here](https://github.com/MedMCQA/MedMCQA)

Since our focus is to reason over the context, we used the _context_ setting in the MedMCQA dataset, eliminating any explanations that are fewer than 60 words. The resulting dataset contains 1,041 records.  
The new dataset has been reformatted into JSONL and is named mmlu_testset.jsonl. This test set includes the following fields:

1. **Title**: Represents a short title for the use case (e.g., _exp_). The title is generated using GPT-4.1 nano.
2. **generated_conclusion**: A conclusion generated using the MedConclusion model.

To run the evaluation: 
```python
python medmcqa.py
```
An example of the format for one record in the MedMCQA dataset is as follows:
```json
{
"question": "Which vitamin is required for glycogen Phosphorylase?",
"exp": "Glycogen phosphorylase is the rate limiting enzyme of glycogenolysis. And it requires PLP. The active form of vitamin B6 is the coenzyme pyridoxal phosphate (PLP) PLP can be synthesized from the three compounds pyridoxine, pyridoxal and pyridoxamine. This PLP for this enzyme Glycogen phosphorylase is not required as co-enzyme, but it act as a phosphate donor. Enzyme glycogen phosphorylase will cut glycogen a (1-4) bond apa and the glucose released are transferred in Glucose-1-phosphate and that phosphate is taken from PLP.",
"cop": 1, "opa": "PLP", "opb": "TPP", "opc": "Riboflavin", "opd": "Lipoic acid",
"subject_name": "Biochemistry", "topic_name": "AIIMS 2017", "id": "3624dceb-9318-4aa7-add1-b4c2fbac3065",
"choice_type": "single",
"title": "Vitamin B6 as a Phosphate Donor in Glycogenolysis",
"generated_conclusion": "Glycogen phosphorylase is the rate limiting enzyme of glycogenolysis. And it requires PLP. The active form of vitamin B6 is the coenzyme pyridoxal phosphate (PLP) PLP can be synthesized from the three compounds pyridoxine, pyridoxal and pyridoxamine. This PLP for this enzyme Glycogen phosphorylase is not required as co-enzyme, but it act as a phosphate donor. Enzyme glycogen phosphorylase will cut glycogen a (1-4) bond apa and the glucose released are transferred in Glucose-1-phosphate and that phosphate is taken from PLP."
}
```

## MMLU Evaluation

The MMLU_professional_medicine dataset is composed of 272 records. The dataset is originally in CSV format and is a subset of the MMLU dataset available [here](https://github.com/hendrycks/test?tab=readme-ov-file).

The dataset has been reformatted into JSONL and is named mmlu_testset.jsonl. This test set includes new attributes as follows:

1. **Title**: Represents a short title for the use case (e.g., _full_context_). The title is generated using GPT-4.1 nano.
2. **generated_conclusion**: A conclusion generated using the MedConclusion model.
3. **context** and **question**: The original full_context is split into two fields—context (the background information) and question (the final question)—to better structure the prompt using context + conclusion.
An example of the format for one record in the MMLU professional medicine dataset is as follows:
```json

{
"full_context": "A 57-year-old woman comes to the physician because of an 8-week history of difficulty sleeping, fatigue, and muscle tension. During this period, she also has had memory lapses, difficulty concentrating, and has been reprimanded at work for arriving late. Over the past 2 weeks, she has had three episodes of palpitations and shortness of breath that have awakened her from sleep. Her pulse is 80/min, and blood pressure is 110/90 mm Hg. Physical examination shows no abnormalities. Mental status examination shows a depressed mood and constricted affect. She says that she is no longer interested in activities that she used to enjoy. She has suicidal ideation without a plan. Her hemoglobin concentration is 11 g/dL, and serum ferritin concentration is 140 ng/mL. Which of the following is the most appropriate initial step in treatment?",
"options": {"A": "Donepezil therapy", "B": "Ferrous sulfate therapy", "C": "Ginkgo biloba extract therapy", "D": "Paroxetine therapy"},
"answer_idx": "D",
"context": "A 57-year-old woman comes to the physician because of an 8-week history of difficulty sleeping, fatigue, and muscle tension. During this period, she also has had memory lapses, difficulty concentrating, and has been reprimanded at work for arriving late. Over the past 2 weeks, she has had three episodes of palpitations and shortness of breath that have awakened her from sleep. Her pulse is 80/min, and blood pressure is 110/90 mm Hg. Physical examination shows no abnormalities. Mental status examination shows a depressed mood and constricted affect. She says that she is no longer interested in activities that she used to enjoy. She has suicidal ideation without a plan. Her hemoglobin concentration is 11 g/dL, and serum ferritin concentration is 140 ng/mL.",
"question": "Which of the following is the most appropriate initial step in treatment?",
"title": "Initial Management of Suspected Anxiety or Depression",
"generated_conclusion": "This patient has symptoms of depression and anxiety. The initial management of patients with suspected anxiety or depression includes a thorough history and physical examination, laboratory testing, and a psychiatric evaluation."
}
``` 
To run the evaluation: 
```python
python mmlu.py
```

