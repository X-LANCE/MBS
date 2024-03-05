import argparse
import os 
from lib.data import get_bibles, get_more_bibles, get_xlsum, mix_cc100
from main import get_llm
import torch
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from lib.prune import prepare_calibration_input
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--language', type=str, help='language')
args = parser.parse_args()


print("Loading Model")
model_path = "bigscience/bloom-7b1"
model = get_llm(model_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
print("Loading Model Complete")

nsamples = 256
seeds = [1
         ]
seqlen = 2048

languages = [
            args.language
             ]
device = torch.device("cpu")

if languages[0]!="Measure":

    for seed in seeds:
        for language in languages:
            print(f"Loading {language}")
            if not os.path.exists(f"./language_distribution/{language}_dist_{seed}_{model_path[-3:]}.pt"):
                dataloader, _ = mix_cc100([nsamples], seed, seqlen, tokenizer, [language])

                inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

                scaler_row = torch.zeros((model.config.hidden_size), device=device)
                
                n_samples = 0

                for inp in inps:

                    if len(inp.shape) == 2:
                        inp = inp.unsqueeze(0)
                    tmp = inp.shape[0]
                    
                    if len(inp.shape) == 3:
                        inp = inp.reshape((-1, inp.shape[-1]))
                    inp = inp.t()

                    scaler_row *= n_samples / (n_samples + tmp)
                    n_samples += tmp

                    inp = inp.type(torch.float32)
                    scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / n_samples

                if not os.path.exists("./language_distribution"):
                    os.makedirs("./language_distribution")
                torch.save(scaler_row, f"./language_distribution/{language}_dist_{seed}_{model_path[-3:]}.pt")
            else:
                scaler_row=torch.load(f"./language_distribution/{language}_dist_{seed}_{model_path[-3:]}.pt")

            print(languages.index(language) + len(languages) * seed)
            print(f"{language} {seed}_{model_path[-3:]} complete")
            torch.cuda.empty_cache()

            del model
            del tokenizer
            del dataloader
            del inps
            del outs
            del attention_mask
            del position_ids
            del scaler_row

            torch.cuda.empty_cache()

else:

    languages = [
            "en","zh-Hans","fr","es","pt","ar","vi","hi","id","bn","ta","te","ur","ne","mr","gu","zh-Hant","sw",
             "yo",
             "ig"
    ]
    scaler_rows = torch.zeros((len(languages)*len(seeds), model.config.hidden_size))
    for seed in seeds:
        for language in languages:
            scaler_rows[languages.index(language) + len(languages) * seed*0] = torch.load(f"./language_distribution/{language}_dist_{seed}_{model_path[-3:]}.pt")

    print("calculating cos sim dist")

    cos_sim = cosine_similarity(scaler_rows.detach().numpy())
    print("cos_sim\n" + str(cos_sim) + "\n\n")

    euc_dist = euclidean_distances(scaler_rows.detach().numpy())
    print("euc_dist\n" + str(euc_dist))


    cos_sim_degree = (np.arccos(cos_sim)/3.1415926*180).astype(int)

    print("cos_sim_degree\n" + str(cos_sim_degree) + "\n\n")

    with open(f"./language_distribution/dist_{seed}_{model_path[-3:]}.txt", "w") as f:
        f.write("cos_sim\n" + str(cos_sim) + "\n\n")
        f.write("cos_sim_degree\n" + str(cos_sim_degree) + "\n\n")

    np.save(f"./language_distribution/cos_sim_{seed}_{model_path[-3:]}.npy", cos_sim)
    np.save(f"./language_distribution/cos_sim_degree_{seed}_{model_path[-3:]}.npy", cos_sim_degree)