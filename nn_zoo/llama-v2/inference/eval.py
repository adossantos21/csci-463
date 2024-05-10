"""
File Name: eval.py
Purpose: Performing inference with the Llama-v2-7B model
Author: Alessandro Dos Santos
Documented: 05/10/2024
"""
import time
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
    BitsAndBytesConfig
)

def main():

    # Specify latency flag, model path, and tokenizer
    latency_flag = True
    ckpt = 'shared_csci463/llama-v2/hf'
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    # Configure input and model behavior
    SYS_PROMPT = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."  
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature."
        "Additionally, use the given prompt to guide your summary about the context. If you don't know the answer, just say that you don't know, don't try to make up an answer."
        "Do not mention that you have been filtered through any of the aforementioned instructions, in your response/prediction"
    )
    task = "summarize the following: "
    input_prompt = "The task of semantic edge detection (SED) is aimed at both detecting visually salient edges and recognizing their categories,or more concretely, locating fine edges utlizing lowlevel features and meanwhile identifying semantic categories with abstracted high-level features. An intuitive way for a deep CNN model to achieve both targets is to integrate highlevel semantic features with low-level category-agnostic edge features via a fusion model, which is conventionally designed following a fixed weight fusion strategy, independent of the input, as illustrated in the top row in Figure 1. In many existing deep SED models [Yu et al., 2017; Liu et al., 2018b; Yu et al., 2018], fixed weight fusion of multi-level features is implemented through 1 x 1 convolution, where the learned convolution kernel serves as the fusion weights. However, this fusion strategy cannot fully exploit multi-level information, especially the low-level features. This is because, first, it applies the same fusion weights to all the input images and ignores their variations in contents, illumination, etc. The distinct properties of a specific input need be treated adaptively for revealing the subtle edge details.  besides, for a same input image, different spatial locations on the corresponding feature map convey different information, but the fixed weight fusion manner applies the same weights to all these locations, regardless of their different semantic classes or object parts. This would unfavorably force the model to learn universal fusion weights for all the categories and locations. Consequently, a bias would be caused toward high-level features, and the power of multilevel response fusion is significantly weakened."
    prompt = task+input_prompt
    msg = f"<s>[INST] <<SYS>>\n{SYS_PROMPT}\n<</SYS>>\n\n{prompt.strip()} [/INST]"
    input_ids = tokenizer.encode(msg,return_tensors='pt').to("cuda")

    # Configure output generation
    gen_config = GenerationConfig(
        max_new_tokens=101, # the maximum number of tokens to generate, ignoring the number of tokens in the input prompt
        min_new_tokens=75, # the minimum number of tokens to generate, ignoring the number of tokens in the input prompt
        temperature=1.0, # the value used to modulate the next token probabilities
        renormalize_logits=True,
        pad_token_id=tokenizer.eos_token_id,
        #forced_eos_token_id=List[int]
        num_beams=16,
        early_stopping=True, # Stop when the first beam hypothesis reaches EOS
        num_return_sequences=1
    )

    if latency_flag == True:
        # The bottleneck in the forward pass comes from loading the model layer weights into the computation cores of your device,
        # not from performing the computations themselves.
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        # Initialize an empty skeleton of the model, which won't take up any RAM
        with init_empty_weights():
            model = LlamaForCausalLM.from_pretrained(
                ckpt, 
                #quantization_config=quantization_config, # Quantize model weights to INT8 (reduces layer weight size) to speed up inference
                device_map="auto",
                torch_dtype=torch.bfloat16,  # Ensure model is in the correct dtype
                attn_implementation="flash_attention_2" # Replaces all attention mechanisms with flash attention to speed up inference
            )
        
        #model.to_bettertransformer() # no longer need this for llama because it is automatically applied
        # Load the checkpoint inside the empty model and dispatch the weights for each layer across all available devices (GPUs)
        model = load_checkpoint_and_dispatch(
            model, checkpoint=ckpt, device_map="auto"
        )

        # Generate output
        start_time = time.time()
        output = model.generate(
            inputs=input_ids,
            generation_config=gen_config,
            pad_token_id=tokenizer.eos_token_id
        )
        output_sequence = output[0]
        mask = input_ids.shape[-1] + 7 # masks input prompt + [/INST] from output
        decoded_output = tokenizer.decode(output_sequence[mask:], skip_special_tokens=True)
        print(f"\nPrediction: \n{decoded_output}")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
 
    else:
        with init_empty_weights():
            model = LlamaForCausalLM.from_pretrained(
                ckpt,
                device_map="auto"
            )
        # Load the checkpoint inside the empty model and dispatch the weights for each layer across all available devices (GPUs)
        model = load_checkpoint_and_dispatch(
            model, checkpoint=ckpt, device_map="auto"
        )

        # Generate output
        start_time = time.time()
        output = model.generate(
            inputs=input_ids,
            generation_config=gen_config,
            #synced_gpus=True
        )
        output_sequence = output[0]
        mask = input_ids.shape[-1] + 7 # masks input prompt + [/INST] from output
        decoded_output = tokenizer.decode(output_sequence[mask:], skip_special_tokens=True)
        print(f"\nPrediction: \n{decoded_output}")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()
