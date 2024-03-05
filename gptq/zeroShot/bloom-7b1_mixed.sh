# Set common variables
model="bigscience/bloom-7b1"

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command

python main.py --model $model c4 --wbits 3 --groupsize 1024 --tasks "boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,xstory_cloze_en,pawsx_en,xnli_en,xwinograd_en,copa,xstory_cloze_zh,pawsx_zh,xnli_zh,xcopa_zh,xwinograd_zh,pawsx_fr,xnli_fr,xwinograd_fr,xstory_cloze_es,pawsx_es,xnli_es,xwinograd_pt,xstory_cloze_ar,xnli_ar,xnli_vi,xcopa_vi,xstory_cloze_hi,xnli_hi,xstory_cloze_id,xcopa_id,xcopa_ta,xstory_cloze_te,xnli_ur,xstory_cloze_sw,xcopa_sw,xnli_sw"





