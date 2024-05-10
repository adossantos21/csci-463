"""
File: tokenization.py
Description: Supporting module to tokenize input text
"""

from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding

class Tokenization:
    PREFIX = "summarize: "
    
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        
    def tokenize(self, data) -> BatchEncoding:
        """Tokenize some set of input data
            TODO: Make more generic

        Args:
            data (dict): Dictionary of input data

        Returns:
            BatchEncoding: Tokenized output data
        """
        inputs = [Tokenization.PREFIX + doc for doc in data["article"]]
        model_inputs = self._tokenizer(inputs, max_length=1024, truncation=True)

        labels = self._tokenizer(text_target=data["highlights"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def getTokenizer(self):
        return self._tokenizer
