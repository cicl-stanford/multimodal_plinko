import numpy as np
import uniform_sampler as uni
import sequential_sampler as seq
from sys import argv

model_type = argv[1]
evidence = argv[2]

dn = float(argv[-3])
cm = float(argv[-2])
csd = float(argv[-1])

param_list = [evidence]

if model_type == "uniform":
    n = int(argv[3])
    param_list.append(n)
    par_ind = 4

elif model_type == "sequential":
    hole_select = argv[3]
    param_list.append(hole_select)

    if hole_select == "softmax":
        beta = float(argv[4])
        param_list.append(beta)
        par_ind = 5
    else:
        par_ind = 4

    n = int(argv[par_ind])
    param_list.append(n)
    par_ind += 1

else:
    raise Exception(f"Model type {model_type} not implemented.")

if "vision" in evidence:
    vbw = int(argv[par_ind])
    param_list.append(vbw)
    par_ind += 1

if "sound" in evidence:
    sbw = float(argv[par_ind])
    param_list.append(sbw)
    par_ind += 1

if "timing" in evidence:
    tb = int(argv[par_ind])
    par_ind += 1
    tbw = float(argv[par_ind])
    par_ind += 1
    param_list.append(tb)
    param_list.append(tbw)

if model_type == "sequential":
    vision_prior = argv[par_ind]
    param_list.append(vision_prior)
    par_ind += 1

    audio_prior = argv[par_ind]
    param_list.append(audio_prior)
    par_ind += 1

params = tuple(param_list + [dn, cm, csd])

print("Parameters:")
print(params)
np.random.seed(1)
    
if model_type == "uniform":
    _ = uni.run_uniform_all_trials(params,
                                 precompute_sims=True)
    
elif model_type == "sequential":
    _ = seq.run_seq_all_trials(params,
                               precompute_sims=True)
                               
else:
    raise Exception(f"Model type {model_type} not implemented.")