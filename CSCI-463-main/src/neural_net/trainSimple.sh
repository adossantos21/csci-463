if [ -z "$1" ]
then
  echo "Error: You did not specify a model name for the training file's output directory. Please specify your model name. \nFor example: ./trainSimple.sh bart-large-cnn"
  exit 1
fi

current_date_time=$(date '+%Y\%m\%d___%H-%M-%S')
mkdir -p output/$1/$current_date_time
nohup python3 huggingFace/finetuneTransformer_main.py --outputDir output/$1 --modelAndTokenizerName facebook/bart-large-cnn --task summarization >> output/$1/$current_date_time/output.log 2>&1 &