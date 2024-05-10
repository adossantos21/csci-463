"""
File Name: neural_net.py
Purpose: Deprecated - For instantiating a Bart LLM and using it for summarization
Author: Arteom Katkov
Documented: 05/10/2024
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

class NeuralNet:
    MODEL_NAME = "facebook/bart-large-cnn"
    RESOURCES_DIR = "resources"
    
    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(NeuralNet.RESOURCES_DIR + "/bart_finetuned")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(NeuralNet.RESOURCES_DIR + "/bart_finetuned")
        self._summarizer_pipeline = pipeline(task="summarization", model=self._model, tokenizer=self._tokenizer)
        
    def summarize_text(self, text: str) -> list:
        print(f"Summarizing {text} w/ len {len(text)}")
        return self._summarizer_pipeline(text)
    