import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_sparsegpt_multilingual, prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers, prune_wanda_multilingual
from lib.eval import eval_ppl, eval_ppl_multilingual

import json
import logging
from lm_eval import tasks, evaluator, utils
import ast



print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LL model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--prune_method", type=str)
    parser.add_argument('--nsamples', type=str, default="128", help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--language', type=str, default=None, help='Language to prune.')
    parser.add_argument('--language_eval', type=str, default=None, help='Language to evaluate perplexity.')


###### Metrics arguments #####
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=1)  ###originally was default = None
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
#################################

    args = parser.parse_args()

    language = args.language
    language_eval = args.language_eval

    language_eval=[str(item) for item in language_eval.split(',')]

    if "coordinated" in args.prune_method:

        args.nsamples = [int(item) for item in args.nsamples.split(',')]
        print(args.nsamples)
        language=[str(item) for item in language.split(',')]
        print(language)

        assert type(args.nsamples)== list and type(language)== list
    else:
        args.nsamples = int(args.nsamples)
        assert type(args.nsamples)== int and type(language)== str



    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                               use_fast=False
                                               )


    device = torch.device("cuda:0")

    print("use device ", device)
    print(str(device))

    if args.sparsity_ratio != 0 and args.language != "None":
        print("pruning starts")
        if args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda":
            prune_wanda_multilingual(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, language=language)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt_multilingual(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, language=language)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"{language}_{len(language_eval)}_log.txt")
    

    if language_eval[0]!="None":

        with open(save_filepath, "a") as f:
            print("actual_sparsity\tpruning_language\teval_language\tppl", file=f, flush=True)
        
        for lang_eval in language_eval:

            if lang_eval!="wikitext2":
                ppl = eval_ppl_multilingual(model, tokenizer, device, lang_eval)
                print(f"ppl on xlsum of {lang_eval} of {model_name} pruned on {language}: {ppl}")
                
            else:
                ppl = eval_ppl(model, tokenizer, device)
                print(f"ppl of {model_name} pruned on {language} on wikitext2 {ppl}")
            
            with open(save_filepath, "a") as f:
                print(f"{sparsity_ratio:.4f}\t{language}\t{lang_eval}\t{ppl:.4f}", file=f, flush=True)

    ################# Zero-shot evaluation #################
    if args.tasks!="None":
        if args.limit:
            print(
                "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
            )

        if args.tasks is None:
            task_names = tasks.ALL_TASKS
        else:
            task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

        print(f"Selected Tasks: {task_names}")

        description_dict = {}
        if args.description_dict_path:
            with open(args.description_dict_path, "r") as f:
                description_dict = json.load(f)

        results = evaluator.simple_evaluate(
            model=model,
            model_args="",
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=device,
            no_cache=args.no_cache,
            limit=args.limit,
            description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            output_base_path=args.output_base_path,
        )

        dumped = json.dumps(results, indent=2)
        print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
        print(
            f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
            f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(evaluator.make_table(results))

        output_path = os.path.join(args.save, f"{language}_result.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a") as f:
            f.write(dumped)

    ####################################################

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()