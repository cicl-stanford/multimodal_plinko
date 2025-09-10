import regression_analysis as r
import evaluate_heatmaps as e
from sys import argv
import pandas as pd
import time

start_time = time.time()

model_type = argv[1]
evidence = argv[2]

if argv[-1] in ["vision", "audio", "occluded"]:
    cut = None
    condition = argv[-1]
    dn = float(argv[-4])
    cm = float(argv[-3])
    csd = float(argv[-2])
else:
    cut = int(argv[-1])
    condition = argv[-2]
    dn = float(argv[-5])
    cm = float(argv[-4])
    csd = float(argv[-3])

if model_type == "uniform":
    n = int(argv[3])
    par_ind = 4

elif model_type == "sequential":
    hole_select = argv[3]

    if hole_select == "softmax":
        beta = float(argv[4])
        par_ind = 5
    else:
        par_ind = 4

    n = int(argv[par_ind])
    par_ind += 1

else:
    raise Exception(f"Model type {model_type} not implemented.")

if "vision" in evidence:
    vbw = int(argv[par_ind])
    par_ind += 1

if "sound" in evidence:
    sbw = float(argv[par_ind])
    par_ind += 1

if "timing" in evidence:
    tb = int(argv[par_ind])
    par_ind += 1
    tbw = float(argv[par_ind])
    par_ind += 1


if model_type == "uniform":
    filename = f"uniform_{evidence}_num_samples_{n}"

    if "vision" in evidence:
        filename += f"_bwv_{vbw}"

    if "sound" in evidence:
        filename += f"_bws_{sbw}"

    if "timing" in evidence:
        filename += f"_timing_bins_{tb}_bwt_{tbw}"

    filename += f"_phys_params_{dn}_{cm}_{csd}.pkl"

elif model_type == "sequential":
    filename = f"{model_type}_{evidence}_hole_select_{hole_select}"

    if hole_select == "softmax":
        filename += f"_beta_{beta}"

    filename += f"_num_samples_{n}"

    if "vision" in evidence:
        filename += f"_bwv_{vbw}"

    if "sound" in evidence:
        filename += f"_bws_{sbw}"

    vision_prior = argv[par_ind]
    par_ind += 1
    audio_prior = argv[par_ind]
    par_ind += 1
    filename += f"_prior_{vision_prior}_{audio_prior}"

    filename += f"_phys_params_{dn}_{cm}_{csd}.pkl"

else:
    raise Exception(f"Model type {model_type} not implemented.")

path = "model_performance/"

print("Loading model performance...")
print(filename)
print()
model_events = r.load_model_perf(path + "collisions/" + filename)
print("Computing regression...")
_, model_pred, _, labels = r.compute_regression(model_events, model_type=model_type, condition=condition, cut=cut)
print("Computing EMD...")
model_emd = e.compute_emd_all_trials(model_pred, condition, labels=labels, cut=cut)

df = pd.DataFrame({"trial": e.world_nums, "distance": model_emd})

if cut is None:
    add_str = ""
else:
    add_str = f"_cut_{cut}"

df.to_csv(path + "emd/" + filename[:-4] + f"{add_str}_condition_{condition}.csv")

print()
print("Runtime: {}".format(time.time() - start_time))
