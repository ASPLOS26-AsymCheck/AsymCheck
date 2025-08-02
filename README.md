# AsymCheck: Delaying Checkpointing for Efficient Modern Distributed Training

**AsymCheck** is an asymmetric partitioned checkpointing mechanism that adjusts partition sizes—employing smaller partitions during forward passes and larger ones during backward passes, in contrast to existing fixed-size approaches.
Further, **AsymCheck** also proposes a fine-grained compression scheme to enhance checkpoint efficiency, and a batched flushing mechanism to reduce persistence latency.

# Implementation

## The system architecture of AsymCheck
**AsymCheck** employs a decoupled and hierarchical storage design for checkpointing and consists of four modules:

1. an asymmetric partitioning and snapshot module
2. a fine-grained checkpoint compression module
3. an optimal batch flushing and persistence module
4. a failure recovery module


The system architecture of **AsymCheck** is as follows: 

<center class ='img'>
<img src="checkpoint_workflow_.jpg" width="600px" />
</center>


# Installation

## **Prerequisites**
- Python >= 3.12
- PyTorch-1.3.+
- CUDA-12.6
- DeepSpeed-0.14.5 
- NCCL-2.20.5 
- Hadoop-3.3.6
- Huggingface-0.24.6


## **Get the code**
```shell
git clone https://github.com/FAST26-AsymCheck/AsymCheck
cd AsymCheck
pip install -r requirements.txt
python setup.py
```

## **Quick start**

We provide codes for seven types of checkpointing solutions. They are ExCP, DataStates-LLM, FastPersist, Gemini, DeepFreeze, CheckFreq, TorchSnapshot and AsymCheck. For each methods, there are codes for six models, which are GPT2, BERT, RoBERT, BLOOM, ResNet and ViT.

For example, to run gpt2-10B with AsymCheck:


```shell
cd AsymCheck/example/llm/gpt2
bash run_clm_no_trainer_ds_gpt2_layer_10B_asym.sh
```


## **Referred Datasets**


- Wikitex-103/2: [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)
- SQuAD: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
- ImageNet: [https://www.image-net.org/](https://www.image-net.org/)



