import pandas as pd

set1 = pd.read_csv(r"C:\dev\biai\report\training\data\metrics1.csv")
set2 = pd.read_csv(r"C:\dev\biai\report\training\data\metrics2.csv")

# assume set1's last epoch was unfinished and drop it
set1_last_epoch = set1["epoch"].max()
set1 = set1[set1.epoch != set1_last_epoch]

# start set2 from set1's end poch
set2["epoch"] = set2["epoch"] + set1_last_epoch

df = pd.concat([set1, set2])  # full dataframe
df = df.drop(["step", "validation_iou"], axis=1)  # we don't need them
df = df.groupby("epoch").mean().dropna()  # group by epoch and drop NaNs
df = df[["training_loss", "validation_loss", "validation_accuracy"]]  # reorder columns

df.to_csv("./training.csv")
