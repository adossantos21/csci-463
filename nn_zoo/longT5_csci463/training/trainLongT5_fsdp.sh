# File Name: trainLongT5_fsdp.sh
# Purpose: Executing training of an LLM across multiple GPUs correctly
# Author: Alessandro Dos Santos
# Documented: 05/10/2024

export CUDA_VISIBLE_DEVICES=0,1
GPUS=2
CURRENT_DATE_TIME=$(date '+%Y-%m-%d___%H-%M-%S')
export CURRENT_DATE_TIME
TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG

# Define Environment Variables - use these to alter training scripts below.
FILE_NAME='trainLongT5_fsdp.py'
USER_PROFILE='adossantos21'
OUTPUT_DIR='csci463_summarizer_longT5'
#MODEL_AND_TOKENIZER_NAME='google/long-t5-tglobal-large'
MODEL_AND_TOKENIZER_NAME='Stancld/longt5-tglobal-large-16384-pubmed-3k_steps'
REPO_NAME=$USER_PROFILE/$OUTPUT_DIR
FINETUNE='Yes'
BRANCH_NAME='main'
PRECISION='bf16' # choices ["no", "fp16", "bf16", "fp8"]
WITH_TRACKING='yes'
ACCUMULATION_STEPS=16
LR=0.001
WEIGHT_DECAY=0.0
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=4
MAX_INPUT_LENGTH=16384
MAX_TARGET_LENGTH=1024
EPOCHS=4
CLIP_VALUE=1.0
LOG_DIR='logs'
LOGS=$LOG_DIR/$OUTPUT_DIR/$CURRENT_DATE_TIME
CONFIG='/home/robert.breslin/alessandro/csci-463/project/huggingFace/default_fsdp_config.yaml'
CHECKPOINT_ITERATIONS='step'
mkdir -p $LOGS
export CONFIG


#: <<'END_COMMENT'
#torchrun --nnodes=1 --nproc-per-node=$GPUS \
nohup \
accelerate launch --config_file $CONFIG \
$FILE_NAME \
--outputDir "$OUTPUT_DIR" \
--modelAndTokenizerName "$MODEL_AND_TOKENIZER_NAME" \
--repoName "$REPO_NAME" \
--finetune "$FINETUNE" \
--existsBranch "$BRANCH_NAME" \
--mixed_precision "$PRECISION" \
--checkpointing_steps "$CHECKPOINT_ITERATIONS" \
--logging_dir "$LOGS" \
--train_batch_size $TRAIN_BATCH_SIZE \
--val_batch_size $VAL_BATCH_SIZE \
--max_input_length $MAX_INPUT_LENGTH \
--max_target_length $MAX_TARGET_LENGTH \
--gradient_accumulation_steps $ACCUMULATION_STEPS \
--learning_rate $LR \
--clip_grad $CLIP_VALUE \
--weight_decay $WEIGHT_DECAY \
--train_epochs $EPOCHS \
--with_tracking \
>> $LOGS/output.log 2>&1 &
#END_COMMENT


: <<'END_COMMENT'
#torchrun --nnodes=1 --nproc-per-node=4 \
accelerate launch --config_file $CONFIG \
$FILE_NAME \
--outputDir "$OUTPUT_DIR" \
--modelAndTokenizerName "$MODEL_AND_TOKENIZER_NAME" \
--repoName "$REPO_NAME" \
--finetune "$FINETUNE" \
--existsBranch "$BRANCH_NAME" \
--mixed_precision "$PRECISION" \
--checkpointing_steps "$CHECKPOINT_ITERATIONS" \
--logging_dir "$LOGS" \
--train_batch_size $TRAIN_BATCH_SIZE \
--val_batch_size $VAL_BATCH_SIZE \
--gradient_accumulation_steps $ACCUMULATION_STEPS \
--learning_rate $LR \
--clip_grad $CLIP_VALUE \
--weight_decay $WEIGHT_DECAY \
--train_epochs $EPOCHS \
--with_tracking \
2>&1 | tee $LOGS/output.log
END_COMMENT
