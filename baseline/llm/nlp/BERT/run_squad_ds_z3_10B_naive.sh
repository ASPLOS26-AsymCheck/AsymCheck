

echo "Container nvidia build = " $NVIDIA_BUILD_ID

# export DIR_Model="/home/mzq/mingzq/workspaces/project/grace/examples/torch/nlp/bert/pre-model/bert-large-uncased/uncased_L-24_H-1024_A-16"
# export DIR_Model = "/data/dataset/nlp/bert/pre-model/bert-base-uncased/uncased_L-12_H-768_A-12"
# export DIR_DataSet="/data/dataset/nlp/bert"
# export DIR_Model="/home/data/mzq/nlp/pre-model/bert-base-uncased/uncased_L-12_H-768_A-12"
export DIR_Model="/data/dataset/nlp/bert/pre-model/bert-large-uncased/uncased_L-24_H-1024_A-16"

export DIR_DataSet="/data/dataset/nlp/bert"
# export dir_path_checkpoint="/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/elastic/pytorch/nlp/bert/scripts/squad_base/squad_elastic_topk_001_checkpoint/pytorch_model.bin"

# init_checkpoint = ${1:-"/home/mzq/mingzq/workspaces/project/grace/examples/torch/nlp/bert/pre-model/bert-base-uncased/uncased_L-12_H-768_A-12/bert_model.ckpt"}
# init_checkpoint=${1:-"$DIR_Model/bert_model.ckpt"}
# init_checkpoint=${1:-"/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/elastic/pytorch/nlp/bert/scripts/squad_base/squad_elastic_topk_001_checkpoint/pytorch_model.bin"}
# init_checkpoint=${1:-"$DIR_Model/bert_base_wiki.pt"}
# init_checkpoint=${1:-"$DIR_Model/bert_large_pretrained_amp.pt"}

# init_checkpoint=${1:-"$dir_path_checkpoint"}

epochs=${2:-"3.0"}
batch_size=${3:-"4"}
learning_rate=${4:-"3e-5"}
warmup_proportion=${5:-"0.1"}
precision=${6:-"fp16"}
num_gpu=${7:-"8"}
seed=${8:-"1"}
squad_dir=${9:-"$DIR_DataSet/squad"}
vocab_file=${10:-"$DIR_Model/vocab.txt"}

OUT_DIR=${11:-"/home/mzq/workspace/project/DeepSpeedExamples/training/BERT/out"}


# train + eval
mode=${12:-"train eval"}
# mode=${12:-"train"}
CONFIG_FILE=${13:-"./bert_configs/10B.json"}
max_steps=${14:-"-1"}


# setup
# density="${density:-0.1}"
density="${density:-1}"
threshold="${threshold:-8192}"
# compressor="${compressor:-sidcoexp}"
compressor="${compressor:-topkef}"
# max_epochs="${max_epochs:-200}"
memory="${memory:-residual}"


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
fi


# Force deepspeed to run with only local node
NUM_NODES=1
NGPU_PER_NODE=4
# NGPU_PER_NODE=1


MASTER_PORT="${1:-10086}"

HOSTFILE=/dev/null

NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=24
MAX_GPU_BATCH_SIZE=3
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
JOB_NAME="deepspeed_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size"


# config_json=./deepspeed_bsz24_fp16_z3_config.json
config_json=./deepspeed_bsz24_z3_config_0310.json
# config_json=deepspeed_bsz24_config.json


batch_size=${3:-"1"}

CMD+="deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NGPU_PER_NODE} \
       --master_port=${MASTER_PORT} \
       --hostfile ${HOSTFILE} \
       run_squad_ds_z3_10B_gemini.py \
       --deepspeed \
       --deepspeed_config ${config_json} "


# CMD+="--init_checkpoint=$init_checkpoint "
# CMD+="--density=$density "
# CMD+="--compressor=$compressor  "
# CMD+="--threshold  $threshold "


if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
fi


CMD+=" --do_lower_case "
CMD+=" --bert_model=bert-large-uncased "
# CMD+=" --bert_model=bert-base-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
# CMD+=" $use_fp16"

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE





