import pandas as pd


df = pd.read_csv('object_detection_results.csv')
df_s = df.sort_values(by=['Frame'])
a = df.loc[df['Class'] == 'car']
print(a)