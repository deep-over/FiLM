import glob
import pandas as pd
from sklearn import metrics

root_path = "/home/jaeyoung/downstream/fomc-hawkish-dovish-main/"
result_directory = root_path + "film_grid_search_results/"
model_name = "film"
xlsx_list = glob.glob(result_directory + f"*_{model_name}.xlsx")

# columns=["Seed", "Learning Rate", "Batch Size", "Val Cross Entropy", "Val Accuracy", "Val F1 Score", "Test Cross Entropy", "Test Accuracy", "Test F1 Score"]
seeds = [5768, 78516, 944601]
# To extract the weighted F1 scores of each 3 seed models from each Excel file
for xlsx in xlsx_list:
    data_category = xlsx.split("/")[-1].split("_")[1]
    model_name = xlsx.split("/")[-1].split("_")[2].split(".")[0]
    df = pd.read_excel(xlsx)
    # df = df[["Seed", "Val F1 Score", "Test F1 Score"]]
    # 각 seed 별 val f1 score, test f1 score를 추출
    seed5768_f1 = df[df.Seed == 5768].sort_values(by="Val F1 Score", ascending=False).iloc[0]["Test F1 Score"]
    seed78516_f1 = df[df.Seed == 78516].sort_values(by="Val F1 Score", ascending=False).iloc[0]["Test F1 Score"]
    seed944601_f1 = df[df.Seed == 944601].sort_values(by="Val F1 Score", ascending=False).iloc[0]["Test F1 Score"]
    # weighted f1 score 계산
    weighted_f1_score = (seed5768_f1 + seed78516_f1 + seed944601_f1) / 3
    print(f"{data_category} {model_name} weighted f1 score: {weighted_f1_score}")
    
