import json
import logging
import os
import time
from datautils import get_loaders
import torch

import evaluator
import tasks
from utils import parse_args, pattern_match
import models
from lm_eval import utils


def main():
    args = parse_args()

    # if args.tasks is None:
    #     raise ValueError("Please specify a task to run")
    # else:
    #     task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    # print(f"Selected Tasks: {task_names}")

    # results = evaluator.simple_evaluate(
    #     args=args,
    #     tasks_list=task_names,
    # )

    # dumped = json.dumps(results, indent=2)
    # print(dumped)

    # if args.output_path:
    #     with open(args.output_path, "w") as f:
    #         f.write(dumped)

    # print(
    #     f"{args.model}"
    #     f"num_fewshot: {args.num_fewshot},"
    #     f" batch_size: {args.batch_size}"
    # )
    # if args.table_results:
    #     print(evaluator.make_table(results))
    # else:
    #     from pprint import pprint
    #     pprint(results)


    lm = models.get_model(args.model).create_from_arg_string({"args": args})

    if args.wbits < 16 and not args.nearest:

        tick = time.time()
        dataloader, _ = get_loaders(
            "c4", seed=0, model=args.model, seqlen=lm.seqlen
        )
        quantizers = lm.bloom_sequential(dataloader)
        print(time.time() - tick)


        ################# Zero-shot evaluation #################
    if args.tasks!="None":
        # if args.limit:
        #     print(
        #         "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        #     )

        if args.tasks is None:
            task_names = tasks.ALL_TASKS
        else:
            task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

        print(f"Selected Tasks: {task_names}")

        # description_dict = {}
        # if args.description_dict_path:
        #     with open(args.description_dict_path, "r") as f:
        #         description_dict = json.load(f)

        results = evaluator.simple_evaluate(
            model=lm,
            model_args="",
            tasks=task_names,
            # num_fewshot=args.num_fewshot,
            # batch_size=args.batch_size,
            # max_batch_size=args.max_batch_size,
            # device=device,
            # no_cache=args.no_cache,
            # limit=args.limit,
            # description_dict=description_dict,
            # decontamination_ngrams_path=args.decontamination_ngrams_path,
            # check_integrity=args.check_integrity,
            # write_out=args.write_out,
            # output_base_path=args.output_base_path,
        )

        dumped = json.dumps(results, indent=2)
        print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
        print(
            f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
            f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(evaluator.make_table(results))

        output_path = os.path.join(args.save, f"result.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a") as f:
            f.write(dumped)

    ####################################################



if __name__ == "__main__":
    main()
