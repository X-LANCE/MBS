import json
import math
import time
from lm_eval import tasks, evaluator, utils

import torch
import torch.nn as nn
import transformers
from transformers import BloomForCausalLM, AutoModelForCausalLM

from gptq import * 
from modelutils import *
from quant import *



def get_bloom(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    # model = AutoModelForCausalLM.from_pretrained(
    #     model, 
    #     torch_dtype=torch.float16, 
    #     low_cpu_mem_usage=True, 
    #     device_map="auto"
    # )
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def bloom_sequential(model, dataloader, dev, means=None, stds=None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize)
            quantizers['transformer.h.%d.%s' % (i, name)] = gptq[name].quantizer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def bloom_eval(model, testenc, bs=100, dev=None):
    print('Evaluation...')

    testenc = testenc.input_ids
    nsamples_total = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False


    dtype = next(iter(model.parameters())).dtype


    testenc = testenc.to(dev)
    nlls = []


    for idx in range(0,nsamples_total,bs):
        layers = model.transformer.h
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
        layers[0] = layers[0].to(dev)

        if idx % 50 == 0:
            print(f"sample {idx}")

        nsamples = min(bs, nsamples_total-idx)

        inps = torch.zeros(
            (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {'i': 0, 'attention_mask': None, 'alibi': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['alibi'] = kwargs['alibi']
                raise ValueError
        layers[0] = Catcher(layers[0])
        for i in range(nsamples):
            batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        alibi = cache['alibi']

        for i in range(len(layers)):
            # print(i)
            layer = layers[i].to(dev)

            if args.nearest:
                subset = find_layers(layer)
                for name in subset:
                    quantizer = Quantizer()
                    quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(
                        W, quantizer.scale, quantizer.zero, quantizer.maxq
                    ).to(next(iter(layer.parameters())).dtype)

            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            layers[i] = layer.cpu() 
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        model.transformer.ln_f = model.transformer.ln_f.to(dev)
        model.lm_head = model.lm_head.to(dev)

        
        for i in range(nsamples):
            hidden_states = inps[i].unsqueeze(0)
            hidden_states = model.transformer.ln_f(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = testenc[
                :, (i * model.seqlen):((i + 1) * model.seqlen)
            ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
        
        torch.cuda.empty_cache()
        
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples_total * model.seqlen))

    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl


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


def bloom_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'cc100'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))


    args = parser.parse_args()
    device = torch.device("cuda:0")

    model = get_bloom(args.model)
    print("model loaded")
    model.eval()
    print("loading calibration data")
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = bloom_sequential(model, dataloader, device)
        print(time.time() - tick)

    model.to(device)

    datasets = ['wikitext2'] 
    languages = [
        "english",
        "chinese_simplified","chinese_traditional","french","spanish","portuguese","arabic","vietnamese","hindi","indonesian","bengali","tamil","telugu","urdu","nepali","marathi","gujarati","swahili","yoruba",
                 "igbo"]

    result_save_dir = f"ppl_results_{args.model[-4:]}_1_multilingual_equal.txt"

    with open(result_save_dir, "a") as f:
        print(f"{time.time()}", file=f, flush=True)
        print("eval_language\tppl", file=f, flush=True)
        
    for dataset in datasets: 
        _, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        ppl = bloom_eval(model, testloader,  100, device)

        with open(result_save_dir, "a") as f:
            print(f"{dataset}\t{ppl}", file=f, flush=True)

        del testloader

    
    for dataset in languages: 
        _, testloader = get_loaders(
            "xlsum", seed=args.seed, model=args.model, seqlen=model.seqlen, language=dataset
        )
        print(dataset)
        ppl = bloom_eval(model, testloader, 100, device)

        with open(result_save_dir, "a") as f:
            print(f"{dataset}\t{ppl}", file=f, flush=True)

        del testloader


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
            model=model,
            model_args="",
            tasks=task_names,
            # num_fewshot=args.num_fewshot,
            batch_size=1,
            # max_batch_size=args.max_batch_size,
            device=device,
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

        output_path = f"./result.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a") as f:
            f.write(dumped)
    

    if args.save:
        bloom_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
