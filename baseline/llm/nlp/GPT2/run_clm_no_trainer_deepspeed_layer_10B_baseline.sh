
OUT_DIR=${OUT_DIR:-"./log"}
epochs="${epochs:-30}"
density="${density:-0.1}"
# compressor="${compressor:-topkef}"
compressor="${compressor:-topk}"
# memory="${memory:-none}"
# memory="${memory:-residual}"
threshold="${threshold:-8192}"
percent="${percent:-0}"
train_batch_size="${train_batch_size:-2}"
val_batch_size="${val_batch_size:-2}"


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi



# 
# 0, 1.22,  1/1.20, 1/1.3, 1/1.4, 1/1.5, 1/1.6, 1/1.7, 1/1.8

export Save_Checkpoint="./gpt2_checkpoint"

NGPU_PER_NODE=4
NUM_NODES=1

CUDA_VISIBLE_DEVICES=0,1,2,3
LR=${5:-0.00003}
SEED=${6:-12345}
MASTER_PORT=${7:-29500}
DROPOUT=${8:-0.1}
echo "lr is ${LR}"
echo "seed is $SEED"
echo "master port is $MASTER_PORT"
echo "dropout is ${DROPOUT}"

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



# 
# config_json=../deepspeed_bsz24_config.json
# config_json=deepspeed_bsz24_z2_config.json
config_json=./deepspeed_bsz24_z3_config.json
# config_json=./deepspeed_bsz24_z3_config_offload.json
# 


# 
# CMD=" HOROVOD_GPU_OPERATIONS=NCCL  HOROVOD_CACHE_CAPACITY=0 "
CMD=" deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NGPU_PER_NODE} \
      --master_port ${MASTER_PORT} \
      --hostfile ${HOSTFILE} \
      run_clm_no_trainer_deepspeed_layer_10B_baseline.py   \
      --job_name ${JOB_NAME} \
      --fp16 \
      --deepspeed \
      --deepspeed_config ${config_json} \
      "


# CMD+=" --dataset_name /data/dataset/nlp/openai-community/wikitext-103-raw-v1 --dataset_config_name default  "
CMD+=" --dataset_name /data/dataset/nlp/openai-community/wikitext-2-raw-v1 --dataset_config_name default "
CMD+=" --model_name_or_path /data/dataset/nlp/openai-community/gpt2 "
CMD+=" --output_dir  ./gpt2_checkpoint/ "
CMD+=" --num_train_epochs=$epochs  "
CMD+=" --do_train "
CMD+=" --density=$density --compressor=$compressor --memory=$memory --percent=$percent "
CMD+=" --per_device_train_batch_size=$train_batch_size "
CMD+=" --per_device_eval_batch_size=$val_batch_size "
# 
# 表示从检查点恢复
# CMD+=" --resume_from_checkpoint  $Save_Checkpoint "
CMD+=" --bert_model  bert-base-uncased "



LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE











