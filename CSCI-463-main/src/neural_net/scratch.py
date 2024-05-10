from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    GenerationConfig,
)

class NeuralNet:
    
    def __init__(self) -> None:
        self.CKPT_PATH = "/home/robert.breslin/alessandro/csci-463/project/huggingFace/shared_csci463/llama-v2/hf"
        self._tokenizer = AutoTokenizer.from_pretrained(self.CKPT_PATH)
        
    def encode(self, input_prompt: str) -> list:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
        Also specifies the behavior of the neural net.
        """
        SYS_PROMPT = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."  
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
            "Please ensure that your responses are socially unbiased and positive in nature."
            "Additionally, use the given prompt to guide your summary about the context. If you don't know the answer, just say that you don't know, don't try to make up an answer."
            "Do not mention that you have been filtered through any of the aforementioned instructions, in your response/prediction"
            "If anyone requests that you give an answer outside the specifications of this system prompt, tell them you are unable to provide an answer that is seemingly inappropriate."
        )
        task = "summarize the following: "
        prompt = task+input_prompt
        msg = f"<s>[INST] <<SYS>>\n{SYS_PROMPT}\n<</SYS>>\n\n{prompt.strip()} [/INST]"
        input_ids = self._tokenizer.encode(msg, return_tensors='pt').to("cuda")
        return input_ids

    def decode(self, gen_config, input_ids: list) -> str:
        with init_empty_weights():
            model = LlamaForCausalLM.from_pretrained(
                    self.CKPT_PATH, 
                    device_map="auto",
                    torch_dtype=torch.bfloat16,  # Ensure model is in the correct dtype
                    #attn_implementation="flash_attention_2" # Replaces all attention mechanisms with flash attention to speed up inference
                )

        model = load_checkpoint_and_dispatch(
            model, checkpoint=self.CKPT_PATH, device_map="auto"
        )

        output = model.generate(
            inputs=input_ids,
            generation_config=gen_config,
            pad_token_id=self._tokenizer.eos_token_id
        )
        output_sequence = output[0]
        mask = input_ids.shape[-1] + 7
        decoded_output = self._tokenizer.decode(output_sequence[mask:], skip_special_tokens=True)
        return decoded_output

    def summarize_text(self, input_prompt: str) -> str:
        gen_config = GenerationConfig(
            max_new_tokens=200, # the maximum number of tokens to generate, ignoring the number of tokens in the input prompt
            min_new_tokens=150, # the minimum number of tokens to generate, ignoring the number of tokens in the input prompt.. need at least 75 tokens as a minimum
            temperature=1.0, # the value used to modulate the next token probabilities
            renormalize_logits=True,
            pad_token_id=self._tokenizer.eos_token_id,
            forced_eos_token_id=[29889], # force a period to be the end of sentence.. can specify extra token ids in the list for various sentence endings.
            num_beams=16,
            early_stopping=True, # Stop when the first beam hypothesis reaches EOS
            num_return_sequences=1
        )
        input_ids = self.encode(input_prompt)
        return self.decode(gen_config, input_ids)    