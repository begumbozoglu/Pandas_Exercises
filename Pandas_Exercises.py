
##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
import numpy as np
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].nunique()

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
col_names=["pclass","parch"]
df.loc[:,col_names].nunique()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

type(df[["embarked"]])
df["embarked"] = df["embarked"].astype(category)

#########################################
# Görev 7: embarked değeri C olanların tüm bilgilerini gösteriniz.
#########################################

df[df["embarked"] == "C"]

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgilerini gösteriniz.
#########################################

df[df["embarked"] != "S"]

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["age"] < 30) & (df["sex"] == "female")]

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df["age"] > 70) | (df["fare"] > 500)]

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop(columns="who", axis=1)

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################
#1.yol
mode_deck = df['deck'].mode()[0]
print(mode_deck)
df["deck"].isnull().sum()
df["deck"].fillna(mode_deck,inplace=True)  # inplace = True argümanı düzenlemeyi orjinal dataframe üzerinde yapılmasını sağlar.
df["deck"].isnull().sum()
#2.yol
df["deck"] = df["deck"].replace(to_replace=np.nan, value=df["deck"].mode()[0])
#3.yol
df['deck'] = np.where((df["deck"].isnull()), "C", df.deck)
#########################################
# Görev 14: age değişkenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df['age'].fillna(df['age'].median(), inplace=True)

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass","sex"]).agg({"survived": ["sum", "count", "mean"]})

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

#apply-lambda
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

#fonksiyonlu
def thirty_flag(a):
    if a < 30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(thirty_flag)

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df2 = sns.load_dataset("tips")
df2.info()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df2.groupby(["time"]).agg({"total_bill": ["sum", "min","max", "mean"]})

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df2.groupby(["time","day"]).agg({"total_bill": ["sum", "min","max", "mean"]})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

df2.loc[(df2["sex"] == "Female") & (df2["time"] == "Lunch")].groupby(["day"]).agg({"total_bill": ["sum", "min","max", "mean"],
                                                                                    "tip": ["sum", "min","max", "mean"]})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df2.loc[(df2["size"] < 3) & (df2["total_bill"] > 10)].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df2["total_bill_tip_sum"] = df2["total_bill"] + df2["tip"]
df2.head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

df2_new = df2.sort_values("total_bill_tip_sum", ascending=0).head(30).reset_index()

