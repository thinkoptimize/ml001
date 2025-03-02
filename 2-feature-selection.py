import pandas as pd
import numpy as np
from scipy.stats import spearmanr
df = pd.read_csv("datasets/all_data.csv")
df = df.drop(columns=['Unnamed: 0'])

# Özellik seçme. 65 tane özellik içinden Spearman Korelasyonu ile özellik seçimi yapıldı
target_column = 'Label'
X = df.drop(columns=[target_column])
y = df[target_column]
spearman_corr = {}

for column in X.columns:
    corr, _ = spearmanr(X[column], y)
    spearman_corr[column] = corr

corr_df = pd.DataFrame(spearman_corr.items(), columns=['Feature', 'Spearman_Correlation'])
corr_df['Absolute_Correlation'] = corr_df['Spearman_Correlation'].abs()
threshold = 0.20
selected_features = corr_df[corr_df['Absolute_Correlation'] > threshold]['Feature'].tolist()
# Seçilen özellikleri göster. 0.2 threshold ile 38 adet özellik seçildi.
print(f"Seçilen Özellikler ({len(selected_features)}):")

def create_selected_df(df, selected_features, label_column='Label'):
    columns_to_keep = selected_features + [label_column]
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    return df[columns_to_keep]

df_selected = create_selected_df(df, selected_features)
df_selected.info()
# Ön işleme ve özellik seçimi işlemlerinden sonra oluşan ve model kurma aşamasında kullanacğaımız veri setini kaydettim.
df_selected.to_csv("datasets/selected.csv")