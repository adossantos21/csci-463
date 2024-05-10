"""
File Name: evaluateSummarizer.py
Purpose: Performing inference using the LongT5 model
Author: Alessandro Dos Santos
Documented: 05/10/2024
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():

    prefix = "summarize: "
    input_prompt = "The task of semantic edge detection (SED) is aimed at both detecting visually salient edges and recognizing their categories,or more concretely, locating fine edges utlizing lowlevel features and meanwhile identifying semantic categorieswith abstracted high-level features. An intuitive way for a deep CNN model to achieve both targets is to integrate highlevel semantic features with low-level category-agnostic edge features via a fusion model, which is conventionally designed following a fixed weight fusion strategy, independent of the input, as illustrated in the top row in Figure 1. In many existing deep SED models [Yu et al., 2017; Liu et al., 2018b; Yu et al., 2018], fixed weight fusion of multi-level features is implemented through 1 x 1 convolution, where the learned convolution kernel serves as the fusion weights. However, this fusion strategy cannot fully exploit multi-level information, especially the low-level features. This is because, first, it applies the same fusion weights to all the input images and ignores their variations in contents, illumination, etc. The distinct properties of a specific input need be treated adaptively for revealing the subtle edge details.  besides, for a same input image, different spatial locations on the corresponding feature map convey different information, but the fixed weight fusion manner applies the same weights to all these locations, regardless of their different semantic classes or object parts. This would unfavorably force the model to learn universal fusion weights for all the categories and locations. Consequently, a bias would be caused toward high-level features, and the power of multilevel response fusion is significantly weakened."
    hub_model_id = (
        'shared_csci463/longT5_csci463/longt5-tglobal-large-16384'
    )
    tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
    input_ids = tokenizer.encode(prefix+input_prompt, return_tensors="pt")
    model = AutoModelForSeq2SeqLM.from_pretrained(hub_model_id)
    output = model.generate(
    input_ids,
    max_new_tokens=1024,  # Generate up to 1024 new tokens
    min_new_tokens=512,      # Ensure the new tokens length is at least 512 tokens
    )
    #print(f"Shape of input_ids.shape[-1]: {input_ids.shape}") # check mask shape
    #print(f"Shape of output[0]: {output[0].shape}") # check output shape
    output_sequence = output[0]
    mask = input_ids.shape[-1] # masks input prompt from output.. This does not work for this type of tokenizer, use 269 as mask instead
    decoded_output = tokenizer.decode(output_sequence[269:], skip_special_tokens=True)
    print(f"\nPrediction: \n{decoded_output}") # print masked output
    print(f"\nPrediction: \n{tokenizer.decode(output[0], skip_special_tokens=True)}") # print non-masked output (includes entire input prompt)


if __name__ == "__main__":
    main()
