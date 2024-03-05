# Set common variables
model="bigscience/bloom-7b1"
sparsity_ratio=0.5

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    --language $4 \
    --nsamples $5 \
    --language_eval $6 \
    --task $7
}


#English Monolingual Pruning

## Wanda
run_python_command "wanda" "unstructured" "results/wanda/bloom_7b1/" "en" "256" "english,chinese_simplified,chinese_traditional,french,spanish,portuguese,arabic,vietnamese,hindi,indonesian,bengali,tamil,telugu,urdu,nepali,marathi,gujarati,swahili,yoruba,igbo,wikitext2" "boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,xstory_cloze_en,pawsx_en,xnli_en,xwinograd_en,copa,xstory_cloze_zh,pawsx_zh,xnli_zh,xcopa_zh,xwinograd_zh,pawsx_fr,xnli_fr,xwinograd_fr,xstory_cloze_es,pawsx_es,xnli_es,xwinograd_pt,xstory_cloze_ar,xnli_ar,xnli_vi,xcopa_vi,xstory_cloze_hi,xnli_hi,xstory_cloze_id,xcopa_id,xcopa_ta,xstory_cloze_te,xnli_ur,xstory_cloze_sw,xcopa_sw,xnli_sw"

## SparseGPT
run_python_command "sparsegpt" "unstructured" "results/sparsegpt/bloom_7b1/" "en" "256" "english,chinese_simplified,chinese_traditional,french,spanish,portuguese,arabic,vietnamese,hindi,indonesian,bengali,tamil,telugu,urdu,nepali,marathi,gujarati,swahili,yoruba,igbo,wikitext2" "boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,xstory_cloze_en,pawsx_en,xnli_en,xwinograd_en,copa,xstory_cloze_zh,pawsx_zh,xnli_zh,xcopa_zh,xwinograd_zh,pawsx_fr,xnli_fr,xwinograd_fr,xstory_cloze_es,pawsx_es,xnli_es,xwinograd_pt,xstory_cloze_ar,xnli_ar,xnli_vi,xcopa_vi,xstory_cloze_hi,xnli_hi,xstory_cloze_id,xcopa_id,xcopa_ta,xstory_cloze_te,xnli_ur,xstory_cloze_sw,xcopa_sw,xnli_sw"


# MBS Pruning

## Wanda + MBS
run_python_command "wanda" "unstructured" "results/wanda/bloom_7b1/" "en,zh-Hans,fr,es,pt,ar,vi,hi,id,bn,ta,te,ur,ne,mr,gu,zh-Hant,sw,yo,ig" "87,47,37,31,14,13,7,4,3,3,1,1,1,1,1,1,1,1,1,1" "english,chinese_simplified,chinese_traditional,french,spanish,portuguese,arabic,vietnamese,hindi,indonesian,bengali,tamil,telugu,urdu,nepali,marathi,gujarati,swahili,yoruba,igbo,wikitext2" "boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,xstory_cloze_en,pawsx_en,xnli_en,xwinograd_en,copa,xstory_cloze_zh,pawsx_zh,xnli_zh,xcopa_zh,xwinograd_zh,pawsx_fr,xnli_fr,xwinograd_fr,xstory_cloze_es,pawsx_es,xnli_es,xwinograd_pt,xstory_cloze_ar,xnli_ar,xnli_vi,xcopa_vi,xstory_cloze_hi,xnli_hi,xstory_cloze_id,xcopa_id,xcopa_ta,xstory_cloze_te,xnli_ur,xstory_cloze_sw,xcopa_sw,xnli_sw"

## SparseGPT + MBS
run_python_command "sparsegpt" "unstructured" "results/sparsegpt/bloom_7b1/" "en,zh-Hans,fr,es,pt,ar,vi,hi,id,bn,ta,te,ur,ne,mr,gu,zh-Hant,sw,yo,ig" "87,47,37,31,14,13,7,4,3,3,1,1,1,1,1,1,1,1,1,1" "english,chinese_simplified,chinese_traditional,french,spanish,portuguese,arabic,vietnamese,hindi,indonesian,bengali,tamil,telugu,urdu,nepali,marathi,gujarati,swahili,yoruba,igbo,wikitext2" "boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,xstory_cloze_en,pawsx_en,xnli_en,xwinograd_en,copa,xstory_cloze_zh,pawsx_zh,xnli_zh,xcopa_zh,xwinograd_zh,pawsx_fr,xnli_fr,xwinograd_fr,xstory_cloze_es,pawsx_es,xnli_es,xwinograd_pt,xstory_cloze_ar,xnli_ar,xnli_vi,xcopa_vi,xstory_cloze_hi,xnli_hi,xstory_cloze_id,xcopa_id,xcopa_ta,xstory_cloze_te,xnli_ur,xstory_cloze_sw,xcopa_sw,xnli_sw"






