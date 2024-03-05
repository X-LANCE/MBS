from datautils import get_cc100, get_loaders
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
import torch
import torch.nn as nn
import time

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        print(inputs.get_device())

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


pretrained_model_dir = "bigscience/bloom-7b1"

seed=0
device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)


examples_brut = get_cc100(256, 0, 2048, "model")
examples = []

for example in examples_brut:
    examples.append({"input_ids": example[0][0], "attention_mask": example[1][0]})



quantize_config = BaseQuantizeConfig(
    bits=3,  # quantize model to 4-bit
    group_size=1024,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)
# model.to(device)




del model
del examples_brut
del examples


datasets = ['wikitext2'] 
languages = [
    "english",
    # "chinese_simplified","chinese_traditional","french","spanish","portuguese","arabic","vietnamese","hindi","indonesian","bengali","tamil","telugu","urdu","nepali","marathi","gujarati","swahili","yoruba",
                "igbo"]
save_dir = f"ppl_results_{pretrained_model_dir[-4:]}.txt"

with open(save_dir, "a") as f:
    print(f"{time.time()}", file=f, flush=True)
    print("eval_language\tppl", file=f, flush=True)
    
for dataset in datasets: 
    _, testloader = get_loaders(
        dataset, seed=seed, model=model, seqlen=2048
    )
    print(dataset)
    ppl = eval_ppl(model, testloader, 1, device=device)

    with open(save_dir, "a") as f:
        print(f"{dataset}\t{ppl}", file=f, flush=True)

    del testloader


for dataset in languages: 
    _, testloader = get_loaders(
        "xlsum", seed=seed, model=model, seqlen=2048, language=dataset
    )
    print(dataset)
    ppl = eval_ppl(model, testloader, 1, device)

    with open(save_dir, "a") as f:
        print(f"{dataset}\t{ppl}", file=f, flush=True)

    del testloader


# ################# Zero-shot evaluation #################
# if args.tasks!="None":
#     # if args.limit:
#     #     print(
#     #         "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
#     #     )

#     if args.tasks is None:
#         task_names = tasks.ALL_TASKS
#     else:
#         task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

#     print(f"Selected Tasks: {task_names}")

#     # description_dict = {}
#     # if args.description_dict_path:
#     #     with open(args.description_dict_path, "r") as f:
#     #         description_dict = json.load(f)

#     results = evaluator.simple_evaluate(
#         model=model,
#         model_args="",
#         tasks=task_names,
#         # num_fewshot=args.num_fewshot,
#         batch_size=1,
#         # max_batch_size=args.max_batch_size,
#         device=device,
#         # no_cache=args.no_cache,
#         # limit=args.limit,
#         # description_dict=description_dict,
#         # decontamination_ngrams_path=args.decontamination_ngrams_path,
#         # check_integrity=args.check_integrity,
#         # write_out=args.write_out,
#         # output_base_path=args.output_base_path,
#     )

#     dumped = json.dumps(results, indent=2)
#     print(dumped)

#     batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
#     print(
#         f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
#         f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
#     )
#     print(evaluator.make_table(results))

#     output_path = f"./result.json"
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "a") as f:
#         f.write(dumped)





