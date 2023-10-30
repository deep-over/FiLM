import pandas as pd
import os

root_path = "/home/jaeyoung/downstream/fomc-hawkish-dovish-main/"
result_name = "finbert"
file_name = 'fin_bert_models'
dir_path = "film_grid_search_results/" + result_name + "/"
files = os.listdir(root_path + dir_path)
files_xls = [f for f in files if f'_{file_name}.xlsx' in f]
print(files_xls)
result_df = pd.DataFrame()
for file in files_xls:
    df = pd.read_excel(root_path + dir_path + file)
    df_temp = df.groupby(['Learning Rate', 'Batch Size'], as_index=False).agg(
    {
        "Val F1 Score": ["mean"],
        "Test F1 Score": ["mean", "std"]
    }
    )
    df_temp.columns = ['Learning Rate', 'Batch Size', 'mean Val F1 Score', 'mean Test F1 Score', 'std Test F1 Score']
    # print(df_temp)
    max_element = df_temp.iloc[df_temp['mean Val F1 Score'].idxmax()]
    max_element['category'] = file
    print(file)
    # print(max_element)
    print(max_element['mean Test F1 Score'])
    print(format(max_element['std Test F1 Score'], '.4f'), "\n")

    result_df = result_df.append(max_element)
result_df = result_df.set_index('category')
result_df.to_csv(root_path + dir_path + f"all_result_{result_name}.csv", header=True, index=True)
