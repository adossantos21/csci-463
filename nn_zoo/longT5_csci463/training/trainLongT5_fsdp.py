"""
File Name: trainLongT5_fsdp.py
Purpose: Training LongT5 or Llama-v2-7B model using FSDP. Replace AutoSeq2Seq objects with AutoCausalForLM to train Llama-v2-7B models
Author: Alessandro Dos Santos
Documented: 05/10/2024
"""
import argparse
import gc
import threading
import psutil
import numpy as np
import pandas
import evaluate
import sys
import os
import time
import logging
import math
import json
#import nvidia_smi
from pynvml import *
from tqdm.auto import tqdm

import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
    CPUOffload,
    StateDictType,
    FullOptimStateDictConfig, 
    FullStateDictConfig
)
from torch.optim import AdamW #, Adafactor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import is_npu_available, is_xpu_available, GradientAccumulationPlugin

import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import get_scheduler
from transformers import set_seed
from transformers.utils import send_example_telemetry
from huggingface_hub import Repository, create_repo, get_full_repo_name

import datasets
from datasets import load_dataset

import nltk
from nltk.tokenize import sent_tokenize

#from peft import LoraConfig, TaskType
#from peft import get_peft_model


def monitor_gpuTemp(handle0, handle1):
    temp0 = nvidia_smi.nvmlDeviceGetTemperature(handle0, nvidia_smi.NVML_TEMPERATURE_GPU)
    temp1 = nvidia_smi.nvmlDeviceGetTemperature(handle1, nvidia_smi.NVML_TEMPERATURE_GPU)
    if temp0 >= 75 or temp1 >= 75: 
        print(f"\nGPU 0 Temperature: {temp0}C")
        print(f"\nGPU 1 Temperature: {temp1}C")
        raise RuntimeError("GPU Temperature is too high!")
    
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def training_function(config, args):
    # For GPU temp monitoring
    #nvidia_smi.nvmlInit()
    #handle0 = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    #handle1 = nvidia_smi.nvmlDeviceGetHandleByIndex(1)

    dateAndTime = os.environ['CURRENT_DATE_TIME']
    nltk.download("punkt")

    # Output directory
    output_dir = args.outputDir

    # Pass the advanced FSDP settings not part of the accelerate config by creating fsdp_plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy(1),
        backward_prefetch=BackwardPrefetch(1),
        mixed_precision_policy=MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True),
        auto_wrap_policy=None,
        state_dict_type=StateDictType(3),
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False), # can set these to true to provide more GPU memory at the cost of computation time
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False), # can set these to true to provide more GPU memory at the cost of computation time
        use_orig_params=True,
        sync_module_states=True,
        activation_checkpointing=False
    )

    # Initialize accelerator and gradient accumulation plugin
    grad_accum_plugin = GradientAccumulationPlugin(num_steps=args.gradient_accumulation_steps, sync_with_dataloader=False)
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=False,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            project_dir=args.logging_dir,
            gradient_accumulation_plugin=grad_accum_plugin,
            fsdp_plugin=fsdp_plugin,
        )
    else:
        accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_plugin=grad_accum_plugin, fsdp_plugin=fsdp_plugin)
    accelerator.print(accelerator.distributed_type)

    @accelerator.on_main_process
    def training_log(epoch, num_epoch, i_iter, epoch_iters, optimizer, loss):
        msg = 'Epoch: [{}/{}] Iter:[{}/{}], lr: {}, Loss: {:.6f}'.format(
            epoch, num_epoch, i_iter, epoch_iters,
            [x['lr'] for x in optimizer.param_groups], loss)
        print(msg)

    @accelerator.on_main_process
    def print_rouge(epoch, result):
        print(f"\nEpoch {epoch}: {result}\n")

    @accelerator.on_main_process
    def print_something(variable):
        print(f"\nVariable: {variable}\n")

    @accelerator.on_main_process
    def print_blank_line():
        print(f"\n")


    # Handle repository push
    if accelerator.is_main_process:
        # Clones an existing branch from an existing remote repository to a local directory for finetuning a model
        if args.finetune == 'Yes' and args.existsBranch is not None: 
            exist_repo = args.repoName
            repo = Repository(local_dir=output_dir, clone_from=exist_repo, revision=args.existsBranch) # Saves training data to local directory specified in first argument.
            repo.git_checkout(args.existsBranch) # activates branch
            repo.git_pull(rebase=False) # updates the current local branch with changes from a remote repository
        
        # Creates a new branch for an existing remote repository and clones it to your local directory for finetuning
        if args.finetune == 'Yes' and args.newBranch is not None:
            exist_repo = args.repoName
            create_branch(exist_repo, repo_type="model", branch=args.newBranch)
            repo = Repository(local_dir=output_dir, clone_from=exist_repo, revision=args.newBranch)
            repo.git_checkout(args.newBranch)

        # Creates a new repository for training from scratch
        if args.finetune != 'Yes': 
            new_repo = create_repo(args.repoName)
            repo = Repository(local_dir=output_dir, clone_from=new_repo)

    accelerator.wait_for_everyone()
    
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(args.train_epochs)
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("fsdp_pubMed_no_trainer", experiment_config)

    tokenizer = AutoTokenizer.from_pretrained(args.modelAndTokenizerName)
    raw_train_dataset = load_dataset("ccdv/pubmed-summarization", "document", split="train")
    raw_val_dataset = load_dataset("ccdv/pubmed-summarization", "document", split="validation")
    column_names = raw_train_dataset.column_names
    metric = evaluate.load("rouge")

    # Define tokenizer pre-processing function for the dataset
    max_input_length = args.max_input_length # this defines the maximum number of tokens the model can take as input for any given task.
    max_target_length = args.max_target_length
    padding = "max_length"
    truncation = "longest_first"
    def tokenize_function(examples):

        model_inputs = tokenizer(examples["article"], max_length=max_input_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=examples["abstract"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        train_dataset = raw_train_dataset.map(
            tokenize_function, 
            batched=True,
            num_proc=6,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on raw train split"
            )
        val_dataset = raw_val_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=6,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on raw val split"
            )
        train_dataset.set_format("torch")
        val_dataset.set_format("torch")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = args.gradient_accumulation_steps
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size

    set_seed(seed)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    autoConfig = AutoConfig.from_pretrained(args.modelAndTokenizerName)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.modelAndTokenizerName, config=autoConfig) #, device_map="auto")
    model.gradient_checkpointing_enable() # reduces memory usage during training

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True, # for speeding up training set = true, enables faster transfers between CPU and GPU memory
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
        num_workers=4 # for speeding up training, spawns several workers to preload the data faster. If GPU utilization is far from 100%, increase number of workers.
    )

    val_dataloader = DataLoader(
        val_dataset,
        pin_memory=True,
        collate_fn=data_collator,
        batch_size=val_batch_size,
        num_workers=4
    )


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.003,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    #optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    #optimizer = Adafactor(model.parameters(), lr=args.learning_rate)

    # Training loop updates
    num_train_epochs = args.train_epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_training_steps = num_train_epochs * num_update_steps_per_epoch

    # Instantiate scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_training_steps,
    )

    # Validation loop updates
    num_val_epochs = args.train_epochs
    num_update_steps_per_epoch_val = math.ceil(len(val_dataloader) / args.gradient_accumulation_steps)
    max_validation_steps = num_val_epochs * num_update_steps_per_epoch_val

    # Prepare accelerator
    model = accelerator.prepare(model)
    optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Recalculate training loop updates because it changes after prepare method sometimes
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_training_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_training_steps / num_update_steps_per_epoch)
    progress_bar = tqdm(range(max_training_steps), disable=not accelerator.is_local_main_process)

    # Recalculate validation loop updates because it changes after prepare method
    num_update_steps_per_epoch_val = math.ceil(len(val_dataloader) / args.gradient_accumulation_steps)
    max_validation_steps = num_val_epochs * num_update_steps_per_epoch_val
    num_val_epochs = math.ceil(max_validation_steps / num_update_steps_per_epoch_val)
    val_progress_bar = tqdm(range(max_validation_steps), disable=not accelerator.is_local_main_process)

    if args.checkpointing_steps == "step":
        checkpointing_steps = int(math.floor(num_update_steps_per_epoch * 0.5))
    else:
        checkpointing_steps = None
    completed_steps = 0
    val_completed_steps = 0
    progress_bar.update(completed_steps)
    val_progress_bar.update(val_completed_steps)

    print_something(checkpointing_steps)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            num_epochs -= int(training_difference.replace("epoch_", ""))
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            num_epochs -= resume_step // len(train_dataloader)
            # If resuming by step, we also need to know exactly how far into the DataLoader we went
            resume_step = (num_epochs * len(train_dataloader)) - resume_step

    # Now we train the model
    for epoch in range(num_train_epochs):
        #monitor_gpuTemp(handle0, handle1)
        #completed_steps=0
        #progress_bar.update(completed_steps)
        # Training
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            #monitor_gpuTemp(handle0, handle1)
            with accelerator.accumulate(model):
                batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
                accelerator.wait_for_everyone()
                print_blank_line()
                accelerator.wait_for_everyone()
                progress_bar.update(1)
                accelerator.wait_for_everyone()
                print_blank_line()
                accelerator.wait_for_everyone()
                completed_steps += 1 

            # Save checkpoint on specific step
            if args.checkpointing_steps == "step":
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0 and completed_steps > 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            print(f"Saving checkpoint for epoch {epoch}, step {completed_steps}")
                        checkpoint_step_dir = f"{output_dir}/{dateAndTime}/checkpoints/epoch-{epoch+1}/step-{completed_steps}"
                        os.makedirs(checkpoint_step_dir, exist_ok=True)
                        accelerator.wait_for_everyone() 
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(checkpoint_step_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(checkpoint_step_dir)

            accelerator.wait_for_everyone()
            training_log(epoch, num_train_epochs, completed_steps, max_training_steps, optimizer, loss)
            accelerator.wait_for_everyone()

            if completed_steps >= max_training_steps: # change to 4 if you want to begin validation and checkpoint saving after 5 iterations.
                break
                

        #val_completed_steps=0
        #val_progress_bar.update(val_completed_steps)
        model.eval()
        for step, batch in enumerate(val_dataloader):
            #monitor_gpuTemp(handle0, handle1)
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                predictions = outputs.logits.argmax(dim=-1)
                accelerator.wait_for_everyone()
                predictions, targets = accelerator.gather_for_metrics((predictions, batch["labels"]))

                # Send to cpu for conversion to numpy
                predictions = predictions.cpu().numpy()
                targets = targets.cpu().numpy()
                # Replace -100 in the references (targets) since we can't decode them
                targets = np.where(targets != -100, targets, tokenizer.pad_token_id)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                decoded_preds = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True
                )
                decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)

                decoded_preds, decoded_targets = postprocess_text(
                    decoded_preds, decoded_targets
                )

                metric.add_batch(predictions=decoded_preds, references=decoded_targets)
                if accelerator.sync_gradients:
                    val_progress_bar.update(1)
                    val_completed_steps += 1 
                training_log(epoch, num_epochs, val_completed_steps, max_validation_steps, optimizer, loss)
                # Specify how many iterations contribute towards validation rouge scores metric
                if val_completed_steps >= max_validation_steps: # change to 4 if you want to begin validation and checkpoint saving after 5 iterations.
                    break

        # Compute metrics
        result = metric.compute(use_stemmer=True)

        # Extract the median ROUGE scores
        result = {k: round(v * 100, 4) for k, v in result.items()}
        accelerator.wait_for_everyone()
        print_rouge(epoch, result)
        accelerator.wait_for_everyone()

        if args.with_tracking:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)
                
        # Save and upload
        if args.checkpointing_steps == "epoch":
            if epoch < num_train_epochs:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    print(f"Saving checkpoint for epoch {epoch+1}")
                checkpoint_dir = f"{output_dir}/{dateAndTime}/checkpoints/epoch-{epoch+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                accelerator.wait_for_everyone() 
                #if accelerator.state.fsdp_plugin is not None:
                #    accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
                #accelerator.wait_for_everyone() # This blocks all processes that have finished first from continuing until all remaining processes have reached the same point (this has no effect if you’re running on a single GPU or CPU).
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
                # Also save to default directory
                unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(output_dir)
                    # Function that adds, commits
                    #repo.push_to_hub(
                    #commit_message=f"Model {args.modelAndTokenizerName} checkpoint from epoch {epoch}", blocking=True
                    #)


    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(checkpoint_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    # Also save to default directory
    unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(output_dir)
        # Function that adds, commits
        repo.push_to_hub(
        commit_message=f"Model {args.modelAndTokenizerName}, all checkpoints per step and epoch", blocking=True
        )

        all_results = {f"eval_{k}": v for k, v in result.items()}
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)

    #nvidia_smi.nvmlShutdown()
    

    if args.with_tracking:
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        required=False,
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        required=True,
    )
    parser.add_argument(
        "--val_batch_size", 
        type=int, 
        required=True
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        required=True
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        required=True
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        required=True
    )
    parser.add_argument(
        "--outputDir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--repoName", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--finetune", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--existsBranch",
        type=str, 
        required=False
    )
    parser.add_argument(
        "--newBranch", 
        type=str, 
        required=False
    )
    parser.add_argument(
        "--modelAndTokenizerName",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        help="Maximum input sequence length for inference",
        required=True,
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        help="Maximum output target sequence length for prediction",
        required=True,
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        help="Gradient clipping value. For normalized gradient clipping, the range should be [0.5, 10.0]. For min-max clipping, the range is [-1.0, 10.0]",
        required=True,
    )

    args = parser.parse_args()
    config = {"lr": args.learning_rate, "num_epochs": args.train_epochs, "seed": 42, "batch_size": args.train_batch_size}
    training_function(config, args)

if __name__ == "__main__":
    main()