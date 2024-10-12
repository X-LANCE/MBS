# Multilingual Brain Surgeon: Large Language Models Can be Compressed Leaving No Language Behind
Implementation of **MBS** (Multilingual Brain Surgeon), as presented in our paper:

[**Multilingual Brain Surgeon: Large Language Models Can be Compressed Leaving No Language Behind**](https://arxiv.org/abs/2404.04748) </br>
*Hongchuan Zeng<sup>1</sup>, Hongshen Xu<sup>1</sup>, Lu Chen<sup>1,2</sup>, Kai Yu<sup>1,2</sup>* <br>
<sup>1</sup>X-LANCE Lab, Department of Computer Science and Engineering <br>
MoE Key Lab of Artificial Intelligence, SJTU AI Institute  <br>
Shanghai Jiao Tong University, Shanghai, China <br>
<sup>2</sup>Suzhou Laboratory, Suzhou, China

<p align="center">
<img src="https://github.com/HongchuanZeng/MBS/blob/main/mbs.png" width=100% height=100% 
class="center">
</p>


MBS overcomes the English-centric limitations of existing methods by sampling calibration data from various languages proportionally to the language distribution of the model training datasets, uncovering the dynamics of language interaction during compression, revealing that the larger the proportion of a language in the training set and the more similar the language is to the calibration language, the better performance the language retains after compression.

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage

### Wanda and SparseGPT
The bloom-7b1_MBS.sh contains the bash commands to apply MBS on Wanda and SparseGPT.

Below is an example command for pruning bloom-7b1 with MBS empowered wanda, to achieve unstructured 50% sparsity.
```sh
python main.py \
    --model bigscience/bloom-7b1 \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save results/wanda/bloom_7b1/ \
    --language en,zh-Hans,fr,es,pt,ar,vi,hi,id,bn,ta,te,ur,ne,mr,gu,zh-Hant,sw,yo,ig \
    --nsamples 87,47,37,31,14,13,7,4,3,3,1,1,1,1,1,1,1,1,1,1 \
    --language_eval english,chinese_simplified,chinese_traditional,french,spanish,portuguese,arabic,vietnamese,hindi,indonesian,bengali,tamil,telugu,urdu,nepali,marathi,gujarati,swahili,yoruba,igbo,wikitext2 \
    --task boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,xstory_cloze_en,pawsx_en,xnli_en,xwinograd_en,copa,xstory_cloze_zh,pawsx_zh,xnli_zh,xcopa_zh,xwinograd_zh,pawsx_fr,xnli_fr,xwinograd_fr,xstory_cloze_es,pawsx_es,xnli_es,xwinograd_pt,xstory_cloze_ar,xnli_ar,xnli_vi,xcopa_vi,xstory_cloze_hi,xnli_hi,xstory_cloze_id,xcopa_id,xcopa_ta,xstory_cloze_te,xnli_ur,xstory_cloze_sw,xcopa_sw,xnli_sw
```

We provide a quick overview of the arguments:  
- `--model`: The identifier for the BLOOM model on the Hugging Face model hub.
- `--prune_method`: We have implemented two pruning methods with MBS, namely [ `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--save`: Specifies the directory where the result will be stored.
- `--language`: Specifies the languages that we sample for the calibration set(using the abbreviations of the CC100 dataset).
- `--nsamples`: Specifies the number of segments of each language that we sample for the calibration set respectively.
- `--language_eval`: Specifies the languages that we want to evaluate their perplexity(using the abbreviations of the xlsum dataset).
- `--task`: Specifies the zero-shot tasks that we want to evaluate the results(using the abbreviations of the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework). 

### GPTQ

The gptq/bloom-7b1_MBS.sh contains the bash commands to apply MBS on GPTQ.
Please note that the implementation here is fastidious as you may need to change the calibration languages and segment number under the function get_cc100 in gptq/datautils.py.

We encourage you to test directly using the [implementation of GPTQ on Hugging Face](https://huggingface.co/blog/gptq-integration).


## Acknowledgement
This repository is build upon the [Wanda](https://github.com/locuslab/wanda) and the [GPTQ](https://github.com/IST-DASLab/gptq) repository.
