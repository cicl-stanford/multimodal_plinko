import pandas as pd
import evaluate_heatmaps as e
import regression_analysis as r
import pickle
from sys import argv

split_num = int(argv[1])-1

emd_dict = {
    "split": [],
    "condition": [],
    "trial": [],
    "emd": [],
    "use": []
}

with open("model_performance/cross_val/visual_features/train.pkl", "rb") as f:
    train = pickle.load(f)

with open("model_performance/cross_val/visual_features/test.pkl", "rb") as f:
    test = pickle.load(f)

spl_train = train[split_num]
spl_test = test[split_num]

for condition in ["vision", "audio", "occluded"]:
    print(condition)
    print()

    spl_reg, train_pred, _, _ = r.compute_regression(None, "visual_features", condition, cut=300, world_nums=spl_train)
    test_features, _ = r.setup_model_regression("visual_features", condition, cut=300, world_nums=spl_test)
    test_pred = spl_reg.predict(test_features)

    print("Train")
    train_emd = e.compute_emd_all_trials(train_pred, condition, world_nums=spl_train, cut=300)
    print()
    print("Test")
    test_emd = e.compute_emd_all_trials(test_pred, condition, world_nums=spl_test, cut=300)
    print()

    emd_dict["split"].extend([split_num]*(len(train_emd)+len(test_emd)))
    emd_dict["condition"].extend([condition]*(len(train_emd)+len(test_emd)))
    emd_dict["trial"].extend(spl_train)
    emd_dict["trial"].extend(spl_test)
    emd_dict["emd"].extend(train_emd)
    emd_dict["emd"].extend(test_emd)
    emd_dict["use"].extend(["train"]*len(train_emd))
    emd_dict["use"].extend(["test"]*len(test_emd))

df_emd = pd.DataFrame(emd_dict)
df_emd.to_csv(f"model_performance/cross_val/visual_features/emd_{split_num+1}.csv", index=False)
