import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# 3 Farklı CSV Dosyasını Tek DataFrame'e Dönüştürme#
df1 = pd.read_csv('datasets/metasploitable-2.csv')
df2 = pd.read_csv('datasets/Normal_data.csv')
df3 = pd.read_csv('datasets/OVS.csv')


if list(df1.columns) == list(df2.columns) == list(df3.columns):
    print("Tüm DataFrame'lerin kolon isimleri ve sıraları aynı.")
else:
    print("Kolon isimleri veya sıralar farklı.")

all_df = pd.concat([df1, df2, df3], ignore_index=True)

# Veri Ön İşleme Adımları##########################
# eksik değer kontrolü
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(all_df)

#tekrar eden 2 satır ve tüm değerleri 0 olan 11 kolon veri setinden kaldırıldı.
#Boş değer yok. Veri seti 343889 satırdan oluşuyor.

all_df["Label"].value_counts()
all_df = all_df.drop_duplicates()
sifir_kolonlar = [kolon for kolon in all_df.columns if (all_df[kolon] == 0).all()]
all_df = all_df.drop(columns=sifir_kolonlar)


# Dinamik değişkenlerin kaldırılması Flow ID, Src IP, Dst IP, Timestamp, Src Port, Dst Port, Protocol”
#Modelin yanlış eğitilmesine sebep olacakları için veri setinden kaldırıldı
#65 adet özniteliğimiz kaldı. hedef değişken hariç diğerleri numeric değerler.
dynamic_columns = ["Flow ID","Src IP","Dst IP","Timestamp","Src Port","Dst Port","Protocol"]
all_df = all_df.drop(columns=dynamic_columns)

#normal olanlar 0 , saldırı olanlar 1 olacak şekilde label değerlerini etiketleyelim.
#Normal : 68423 , Saldırı : 275465 . Veri seti dengesiz. Son halini ayrı bir .csv olarak kaydedelim
# veriyi standartlaştıralım
all_df['Label'] = all_df['Label'].apply(lambda x: 0 if x == 'Normal' else 1)
X = all_df.drop('Label', axis=1)
y = all_df['Label']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
df_scaled = pd.concat([X_scaled, y], axis=1)
df_scaled = df_scaled.dropna()
df_scaled["Label"].value_counts()
df_scaled.to_csv("datasets/all_data.csv")


