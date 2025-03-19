# DelayCheck

**Delaying Checkpointing**, called **DelayCheck**, is a new checkpointing mechanism that delays the checkpointing operation until the forward pass ends and the backward pass begins, so as to reduce the training stalls for efficient modern distributed training. Further, we also propose an optimized on-disk checkpointing scheme and a fast failure recovery scheme to enhance the performance of DelayCheck.

# Implementation

## The system architecture of DelayCheck
**DelayCheck** employs a decoupled and hierarchical storage design for checkpointing and consists of three modules:   

1. an in-memory checkpoint creation module
2. an on-disk checkpoint creation module 
3. a failure recovery module

The system architecture of **DelayCheck** is as follows: 

<center class ='img'>
<img src="checkpoint_workflow_1.jpg" width="600px" />
</center>





# Installation

## **Prerequisites**
- CUDA-12.6
- DeepSpeed-0.14.5 
- NCCL-2.20.5 
- Hadoop-3.3.6
- Huggingface-0.24.6

## **Get the code**
```shell
git clone https://github.com/EuroSys25-DelayCheck
cd EuroSys25-DelayCheck
pip install -r requirements.txt
python setup.py
```

## **Quick start**

Codes of four types of checkpointing methods are provided. They are baseline, CheckFreq, tsnapshot and DelayCheck. For each methods, there are codes for four models, which are BERT, GPT2, ResNet and VIT.

For example, to run BERT with DelayCheck:

```shell
cd DelayCheck/bert
bash nvidia_run_squad_ds_lib_DelayCheck.sh
```

## **Referred Datasets**

- ImageNet: [https://www.image-net.org/](https://www.image-net.org/)
- Wikitex-103: [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)
- SQuAD: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
