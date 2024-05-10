import argparse
import evaluate
import numpy as np
import nltk
import time
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline

from utils.tokenization import Tokenization

# Constant definitions
BATCH_SIZE = 8
NOF_TRAIN_EPOCHS = 10

rouge = evaluate.load("rouge") # Define metric for evaluating model performance during
global_tokenizer = None 

def compute_metrics(eval_pred):
    """Compute evaluation metrics

    Args:
        eval_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    global rouge, global_tokenizer
    
    assert global_tokenizer is not None, "Global tokenizer should be set by now...."
    predictions, labels = eval_pred
    decoded_preds = global_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, global_tokenizer.pad_token_id)
    decoded_labels = global_tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != global_tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def get_argparser() -> argparse.ArgumentParser:
    # Instantiate parser
    parser = argparse.ArgumentParser()

    # Command line Options
    parser.add_argument("--outputDir", type=str, help="output path for trained model", 
                        required=True)

    parser.add_argument("--modelAndTokenizerName", type=str, 
                        default='facebook/bart-large-cnn')
                        
    parser.add_argument("--task", type=str, default='summarization')

    return parser

def postprocess_text(preds, labels) -> list:
    """Splits the generated summaries (the predictions) into sentences that are separated by newlines.\
        This is the format that the ROUGE metric expects

    Args:
        preds (_type_): _description_
        labels (_type_): _description_

    Returns:
        _type_: list
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def main() -> None:
    global global_tokenizer
    # Instantiate Parser object
    args = get_argparser().parse_args()
    
    # Download nltk punctuation rules
    nltk.download("punkt")
    
    # Instantiate pipeline, tokenizer, model, and data-collator
    model_name = args.modelAndTokenizerName
    summarizer = pipeline(task=args.task, model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name) # It is more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length
    global_tokenizer = tokenizer
    
    # Load dataset and tokenize it
    tokenization_inst = Tokenization(tokenizer)
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    start_time = time.time()
    tokenized_dataset = dataset.map(tokenization_inst.tokenize, batched=True)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    print(tokenized_dataset)
    
    # Define hyperparameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.outputDir,
        overwrite_output_dir=True,
        save_steps=5000,
        evaluation_strategy="steps",
        #learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        #weight_decay=0.01,
        #save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    # Instantiate trainer object
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    
    
if __name__ == '__main__':
    main()
