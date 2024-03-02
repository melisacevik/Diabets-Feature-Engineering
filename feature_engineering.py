#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

dff = load_application_train()
dff.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()



#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler
###################

sns.boxplot(x=df["Age"])
plt.show()

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df["Age"].quantile(0.25) # yaş değişkenini küçükten büyüğe sıraladığımızda 25. çeyrek değeri bu
q3 = df["Age"].quantile(0.75) # 75. çeyrek değer bu

iqr = q3 - q1

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr # görmezden gelicez

df[(df["Age"] < low) | (df["Age"] > up)] #üst sınıra göre aykırı değerleri yakaladık. bunların indexine nasıl ulaşıcam?

df[(df["Age"] < low) | (df["Age"] > up)].index

###################
# Aykırı Değer Var mı Yok mu?
###################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None) # burada herhangi bir değer var mı? satır sütun hepsine bak => axis=None
df[~((df["Age"] < low) | (df["Age"] > up))].any(axis=None) # aykırı değerlerin dışında değer var mı

df[(df["Age"] < low)].any(axis=None)


# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################

# bu fonksiyon ile alt ve üst limitler hesaplanacak
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age") # yaş değişkeni için alt ve üst değer belirlendi
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare") #sadece up ı almak için böyle de yapabiliriz. (low, up returnden geliyor)

df[(df["Fare"] < low) | (df["Fare"] > up)].head()


df[(df["Fare"] < low) | (df["Fare"] > up)].index


# Aykırı değer var mı yok mu? fonksiyonu
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# içindeki fonksiyondaki argümanları genel fonksiyonda tanımlaman lazım.
check_outlier(df, "Age") # aykırı değer var
check_outlier(df, "Fare") # aykırı değer var

###################
# grab_col_names
###################

dff = load_application_train()
dff.head()

# belirli sayısal değişkenleri seçmem gerekiyor. hepsini tek tek mi yazacağım?

# bu fonksiyon ile :
# kategorik değişkenler
# numerik gibi gözüküp kategorik olanları tespit edeceğiz.
# kategorik gibi gözüken ama kategorik olmayan kardinaller
# bunların hepsini kategorik listesinde tutacağız. artık filtrelendi
# numeric değişkenler,


# cat_th = 10 , 10dan az sınıfa sahipse bu kategoriktir ( bu örnekte ) nunique() ile bakılır
# car_th = 20 , 20den fazla sınıfa sahip ise kardinaldir
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"] # kategorik
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"] #numerik ama kategorikse
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"] # kategorik gibi gözüken ama kardinalliği yüksek değişkenleri bulma
    cat_cols = cat_cols + num_but_cat # kategorik + numerik görünümlü kategorikler
    cat_cols = [col for col in cat_cols if col not in cat_but_car] #kardinalitesi yüksek olan kategorikleri dışarıda bıraktık

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] #tipi objectten farklı olanlar
    num_cols = [col for col in num_cols if col not in num_but_cat] # num_cols'dan numerik görünümlü kategorikleri dısarıda bıraktık

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}') # bu zaten cat_cols'un içinde raporlamak için yazdık
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df) # thlere dokunmadım

num_cols = [col for col in num_cols if col not in "PassengerId"] # bu numerik değişken olmamalı Id var

for col in num_cols:
    print(col, check_outlier(df, col))

# numerik kolonlarda gez, aykırı değer var mı bak


cat_cols, num_cols, cat_but_car = grab_col_names(dff) # aplication_trane.csv

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col)) #chechoutlier => outlier th yi çağırıyor

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

# aykırı değerlerin ne olduğunu incelemek için kullanıyoruz.
# shape[0] => gözlem  | shape[1] => değişken
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10: #aykırı değerlerin gözlem sayısı 10 dan büyükse 5 tanesini getir
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]) # gözlem sayısı 10dan küçükse hepsini getir

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age") # yaş değişkeninde 10 dan fazla aykırı gözlem olduğu için 5 gözlem geldi

grab_outliers(df, "Age", True) # indexlerini istersem True yaz , hem print hem index

age_index = grab_outliers(df, "Age", True)

# 3 şey yaptık
outlier_thresholds(df, "Age") # outlier threshold belirledik
check_outlier(df, "Age") # değişkende outlier var mı
grab_outliers(df, "Age", True) # aykırı değerleri gözlemledik

#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape # aykırı olmayanları getir. ( gözlemle )

# bunu fonksiyonlaştır
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col) # aykırı gözlemleri çıkaracağız ama yeni df ' e atıyoruz

df.shape[0] - new_df.shape[0] # kaç değişiklik oldu bilgisi

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]  #fare değişkenindeki aykırı değerler

df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"] # locla da yapılırdı

df.loc[(df["Fare"] > up), "Fare"] = up # üst sınıra göre aykırı olan değerler # Series[] boş gelir sol taraf cünkü artık up ' a eşitledin

df.loc[(df["Fare"] < low), "Fare"] = low # ihtiyacımız olur diye yazdık

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#sıfırdan yapalım. önce df tanımla
df = load()
#değişkenlerinin doğru sınıflandırıldığı fonksi. getir.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col)) # aykırı değerleri gözlemleme

for col in num_cols:
    replace_with_thresholds(df, col) # aykırı değerleri baskılama

for col in num_cols:
    print(col, check_outlier(df, col)) #tekrar soralım aykırı değer var mı diye artık False


###################
# Recap
###################

df = load() # veri setini okuduk
outlier_thresholds(df, "Age") # threshold belirle
check_outlier(df, "Age") # bu thresholdlara göre outlier var mı kontrol et
grab_outliers(df, "Age", index=True) # bu outlier ları getir

remove_outlier(df, "Age").shape # aykırı değerleri silme fonksiyonu
replace_with_thresholds(df, "Age") # aykırı değerleri baskılama işlemi ( silmek istemezsek )
check_outlier(df, "Age") # aykırı değerlerden kurtulduk




#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################

# 17, 3

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64']) #sadece sayısal değerleri
df = df.dropna() # eksik değerleri drop ederek getiriyoruz
df.head()
df.shape
for col in df.columns:
    print(col, check_outlier(df, col)) # aykırı değerim var mı bak


low, up = outlier_thresholds(df, "carat")
df.shape
df[((df["carat"] < low) | (df["carat"] > up))].shape # 1889 adet aykırı değer var

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape #2545 tane aykırı değer var.

# tek başına baktığımızda çok yüksek sayıda aykırılıklar geldi. çok değişkenli olarak bakalım.

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df) # local outlier factor skorları için hesaplamaları yaptık

df_scores = clf.negative_outlier_factor_ #local outlier factor skorlarını tuttu

# bunu negatif olarak kullanacağız çünkü eşik değere karar vermek için kullanıcı olarak bir bakış gerçekleştirmek istediğimizde
# oluşturacak olduğumuz elbow yöntemi ( dirsek yöntemi ) grafiği tekniğinde daha rahat okunabilirlik açısından eksi olarak bırakıcaz.
# negatif olduğu için artık -1 ' e yakın olması inlier olduğunu gösterir. -1'den uzaklaştıkça aykırı olma eğilimi artar

#df_scores = tüm gözlem birimleri için skorlar verdi(1000 değişken olsa da belirledik) ve bu skorlarla kendimizi th belirlicez

df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:5] # en kötü 5 gözlem

# eşik değer belirleme noktasına ihtiyacım var.
# temel bileşen analizinde PCA'de kullanılan bir dirsek yöntemi var - elbow yöntemi ile bu noktayı belirleyebiliriz.

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# buradaki her nokta eşik değerleri temsil ediyor
th = np.sort(df_scores)[3] # 3.noktayı aldım
# outlier hesaplama işleminde ana problem = eşik değeri belirleme problemiydi.

df[df_scores < th] # negatif olduğu icin daha - leri daha küçükleri aykırı değer olarak belirlicem

df[df_scores < th].shape # çok değişkenli etkiye baktığımızda 3 tane kaldı

#tek başına baktığımızda binlerce gelen aykırılıklar birlikte bakıldığında
#bunlar neden aykırı?
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index #bu 3ünün indexi

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index) # bunları siliyorum

# baskılama yapabilir miydik ?
# kimle baskılayacağız?  rastgele gözlem seçilebilir, en ortalarda olan gözlem seçilebilir ve bu gözlemle bu aykırılığı değiştirebiliriz. az olduğu için.

# ama çok olsaydı : aynı değeri veri setine baskılamış olacağız, değiştirmiş olacağız.
# bir gözlem birimine göre baskılama yapmamız gerektiğinden aykırılığı barındıran gözlemi komple kaldırıp yerine başka bir gözlem koymamız lazım.
# duplicate kayıt, çoklama kayıt üretmiş olacağız. eğer gözlem sayısı bir miktar fazlaysa buraya değiştirmeyle ve benzeri noktalarla dokunmak ciddi problem olabilir.
# ağaç yöntemi ile çalışılıyorsa bunlara hiç dokunmıcaz

# özetle , gözlem sayısı çoksa baskılamak mantıksız
# gözlem sayısı azsa o aykırılık çıkarılmalı



#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

df = load()
df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# isnull metodu sayesinde df bütün hücreleri gezer ve eksiklik var mı kontrol eder. true ya da false döner.
# bu true ya da false yanıtlarını values() ile tutarız.
# any() ile herhangi birinden true varsa getirir.
# bütün veriye en genelinden eksiklik var mı sorusu sormuş olduk!

# degiskenlerdeki eksik deger sayisi
df.isnull().sum() #df.isnull() => eksiklik var mı

# degiskenlerdeki tam deger sayisi
df.notnull().sum() # dolu mu # isnullin tersi

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum() # 866 gelmesinin sebebi bir satırda en az 1 eksiklik varsa o hücreyi sayıyor
# kendisinde en az 1 tane eksik hücre olan satır sayısı

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)] #sütunlara göre herhangi biri eksikse getir

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False) # eksikliğin veri setindeki oranı

# eksik değer sayıları / toplam gözlem sayıları

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0] # eksik değerlerin ismini yakala


# eksik veriye ilk bakış
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] #eksik değerleri barındıran kolonlar

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) # eksik değer sayısı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False) # eksik değer oranı
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio']) # değişken isimlerini ve oranlarını birleştir
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df) # eksikliklerin frekansı ne ? hangi değişkenlerde? bu eksikliklerin oranı ne?

missing_values_table(df, True) # isimlerini almak için


#############################################
# Eksik Değer Problemini Çözme
#############################################
#ağaca dayalı yöntemler kullanılıyorsa eksik değerler aykırı değerler gibi göz ardı edilebilir.

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
df.dropna().shape # gözlem sayısı azalacak bu durumda. çünkü en az 1 tane bile eksik değer varsa silinir.

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################

df["Age"].fillna(df["Age"].mean()).isnull().sum() #df["Age"].fillna(df["Age"].mean()) => yaş değişkenlerini ortalama ile doldurma işlemi . 0 geldi yani eksik yok artık
df["Age"].fillna(df["Age"].median()).isnull().sum() # medyan ile doldurma
df["Age"].fillna(0).isnull().sum() # sabit bir değerle de doldurabiliriz

# yaş değişkenini doldurduk fakat birçok değişken varsa?

# df.apply(lambda x: x.fillna(x.mean()), axis=0) #satırlara bakmamız lazım , sütunun ortalamasını bulmak için satırların ortalamasını almam lazım, ERROR . Veri tipi object olmamalı

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

#Kategorik değişkenler için en mantıklı görülebilecek doldurma yöntemlerinden birisi mod almak.

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum() # mode'un değerini getirdi df["Embarked"].mode()[0]

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

# koşul sağlanırsa modu ile doldur , sağlanmazsa olduğu gibi kalsın
# koşul tipi object ve eşsiz değer sayısı 10 dan küçük eşitse , 10 dan küçük kategorik değişken ise ( kardinalite problemi için kontrol)


# değişkeni doldurmak için axis=0 yazdım! çünkü satırlardaki değerleri gezmem lazım! o yüzden apply metoduna axis=0 verdim
# apply metodu satır ya da sütuna göre geziyor. o değişken yakalandığında lambda fonk. sayesinde dolduruluyor
# birden çok değişkene uygulamak istediğimiz için apply ve lambda kullandık
# tek değişkene uygulamak isteseydik direkt bunu yazardık df["Age"].fillna(df["Age"].mean()).isnull().sum()





###################
# Kategorik Değişken Kırılımında Değer Atama
###################
# kategorik değişkenlerin her birine kendi ortalamalarını atamamız daha doğru çözüm olur!


df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum() #kadınların ortalamasına göre ayrı, erkeklerin ortalamasına göre ayrı
#.transform("mean") ortalamaları ile değiştir

df.groupby("Sex")["Age"].mean()["female"]

#yukarıdaki kodun daha uzun hali 👇🏻

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################

# eksikliğe sahip olan değişkeni bağımlı değişken
# diğer değişkenleri bağımsız değişken olarak kabul edeceğiz.
# modelleme işlemine göre eksik değerlere sahip olan noktaları tahmin etmek için 2 işlem var
# kategorik değişkenleri one hot encodera sokup değişkenleri standartlaştırma
# KNN uzaklık temelli bir algoritma olduğundan değişkenler standartlaştırma

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True) # 1'den fazla kategorik değişkenleri numerik bir şekilde ifade etmek
#label encoding ve one hot encoding işlemini aynı anda yapmak için one hot encoder ı uygulayacağımız get_dummies() metodu,
# drop_first=True argümanını bu şekilde ayarlarsak iki sınıfa sahip kategorik değişkenlerin ilk sınıfını atıp 2. tutacak
# get_dummies() metodu sadece kategorik değişkenlere bir dönüşüm uygular. o yüzden değişkenleri bir araya getirmeyi tercih ediyoruz. kardinaller bilgi içermiyor dışarıda bırak
# kardinalleri dışarda bıraktık

# numerik olanlara dokunmadık ileride standartlaştırıcaz onları da

dff.head()


# değişkenlerin standartlaştırılması
scaler = MinMaxScaler() #değerleri 0 ile 1 arasına dönüştür => MinMaxScaler(), bilgiyi taşıyan scaler
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns) # scalerı veri setimize uyguluyoruz. gelen format iyi format olmad. için dataframe e çevirdik, ve isimlerini dff.columnsdan alıyoruz
dff.head()


# knn'in uygulanması. : uzaklık temelli olarak komşulara bakar. komşulardan, ilgili eksikliğe sahip gözlem birimine en yakın örn. 5 tanesini bulur. bunların dolu olan gözlemlerin ortalamasını eksik olan yere atar
from sklearn.impute import KNNImputer # makine öğrenmesi yöntemiyle tahmine dayalı şekilde eksik değerleri doldurma imkanı sağlar
imputer = KNNImputer(n_neighbors=5) # eksik değerli gözlemin en yakın 5 komşusunu bulur , dolu olan gözlemlerin ortalamasını buraya atar
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns) #fit_transform() ile ilgili df 'e 5 komşuluk özellikli bu imputer nesnesi çalışmış olacak
dff.head()

# değerleri doldurdum ama göremiyorum. bunlar standartlaştırılmış değerler. o yüzden kıyaslama yapabilmek için standartlaştırma işlemini geri almak için inverse yaptım.

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns) #değerler yerine geldi

df["age_imputed_knn"] = dff[["Age"]] # yaş değişkeninde eksik olan değerler dolduruldu ama eski haliyle

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]] # nereye ne atanmış - eksik olan değerlerin yerine tahmin edilmiş değerlerini atamış olduk.
df.loc[df["Age"].isnull()]


###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direkt median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma




#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df) # bar => veri setindeki tam olan gözlemlerin sayısını vermektedir.
plt.show() # cabin değişkeninde 204 tane tam değer varmış

msno.matrix(df)
plt.show() #hata alıyorum
# değişkenlerdeki eksikliklerin birlikte çıkıp çıkmadığı ile ilgili bir bilgi verir.

msno.heatmap(df) # eksikliklerin üzerine kurulu bir ısı haritası oluşturduk.
plt.show()

# bu ısı haritasının bize sunacak olduğu şey nullity correlation değerleridir .
# eksik değerlerin belirli bir korelasyonla ortaya çıkıp çıkmadığı ile ilgileniyorduk( eksik değerlerin rassallığı )
# burada 2 durum var eksikliklerin birlikte çıkması ve eksikliklerin belirli bir değişkene bağımlı olarak çıkması senaryosu
# +1 e yakın olması pozitif yönlü kuvvetli ilişki, -1e yakın olması negatif yönlü kuvvetli ilişki : pozitif yönlü kuvvetli ilişki olması durumunda değişkenlerdeki eksikliklerin birlikte ortaya çıktığı düşünülür.
# yani birinde eksiklik varsa diğerinde de vardır.




###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0) #na 'ye sahip değişkenlere yeni isimlendirme yapıldı. bu değişkende na gördüğü yere 1 görmediği yere 0 yaz.

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns # bu değişkenlerin ismini al

    for col in na_flags: #bu NA barındıran kolonlarda gez, gezerken kolonlara göre grpby alıp targetın ortalamasını ve sayısını al
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)
# eksik 1 dolu 0 # yaş değişkeninde eksiklik olma senaryosunun hayatta kalma oranı 0.29 | yaş değişkeninde dolu olan senaryonun hayatta kalma oranı 0.40
# survived ' a göre hayatta kalma durumunu ne etkiliyor

# cabin değişkeninin %75'inden fazlası eksikti. cabin değişkeni eksik olanların hayatta kalma oranı 0.30
# kabin numrası olmayanların hayatta kalma oranı daha düşük
# NA ifadelerinin birçoğu gemi çalışanlarına aitmiş

###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)





#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

#############################################
# Label Encoding & Binary Encoding
#############################################

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder() # nesneyi getir.
le.fit_transform(df["Sex"])[0:5] #değişkenime fit_transform ile uyguluyorum.
le.inverse_transform([0, 1]) #hangisine 0 hangisine 1 verdiğimizi unuttuk

# neye göre 0 1 veriyor, alfabetik olarak ilk gördüğü değere 0 => female => 0

# fonksiyonlaştırılmış hali
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

#elimde yüzlerce değişken olduğunda ne yapıcam?
# iki sınıflı kategorik değişkenleri seçmenin yolunu bulsak ve bu iki kat. değişkenleri label encoderdan geçirirsem problem çözülür.

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

# int , float ve 2 sınıfı varsa zaten binary encode edilmiş
# eşsiz sınıf sayısı 2 olan
# neden nunique() ? neden len(df[col].unique()) değil ? unique => değişkenin içerisindeki eksik değerleri de sınıf olarak görür. yani değişkende eksik değer varsa len() = 3 çıkar
# nunique() => eksik değeri bir sınıf olarak görmez.

# binary col çok fazla ise

for col in binary_cols:
    label_encoder(df, col) # bu fonksiyonun içinde yeniden atama var zaten

df.head()

# özet olarak titanic veri setindeki cinsiyet değişkenlerini 0-1 olarak encode ettik.

df = load_application_train()
df.shape

# 122 değişkenin kaç tanesi 2 sınıflı kategorik değişken?

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

# bu veri setlerini veri setinden seçmek istersem
df[binary_cols].head()


for col in binary_cols:
    label_encoder(df, col)

# eksik değere 2 atandıı!!! dikkat !

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

#############################################
# One-Hot Encoding
#############################################
# binary encode işleminde 1-0 gelmesi gerekirken True False gelmesi neden olur? dtype=int 'i argüman olarak girdiğinde çözüldü!

df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"], dtype=int).head() # dönüştürülecek olan değişkeni ver

pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int).head() #ilk sınıfı uçurur C gitti #değişkenler birbirlerinin üzerinden oluşturulmasın diye

pd.get_dummies(df, columns=["Embarked"], dummy_na=True, dtype=int).head() # eksik olanlar da değer olarak gelsin

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=int).head() # label encoding ve one hot aynı anda yapılabilir mi
# eğer bir değişkenin sınıf sayısı 2 ise drop first ü True yaptığımda o değişken binary encode edilmiş oluyor.


# elimde kategorik değişkenler var ve bunların hepsini one hottan geçirebilirim, drop_first=true  durumunda iki sınıflı olan değişkenleri label encoder işlemine sokulur.
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

# elimde kategorik değişkenler var ve bunların hepsini one hottan geçirebilirim, drop_first=true  durumunda iki sınıflı olan değişkenleri label encoder işlemine sokulur.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
#one hot encoding kolonları
# cinsiyet ve survived 'dan kurtuldum çübkü zaten bunlar 2 sınıflı ve survived bağımlı değişken

one_hot_encoder(df, ohe_cols).head()

df.head()

#############################################
# Rare Encoding
#############################################
# bonus içerik

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df) # bütün kat,num, değişkenlerimi getircem

# kategorik değişkenlerin sınıflarını getirsin
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col) # burada kat değişkeni seçtik yukarıda hepsi için

# onlarca kategorik değişken var. one hot encoder dan geçiricem ama gereksiz olanlar var o yüzden yaptık

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    # sınıf frekansları / oranları

    for var in rare_columns: #rare ' e sahip kolonlar gezilecek
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01) # bu oranın altında kalan sınıfları biraraya getirecek.

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()


#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

# gradient , doğrusal modelleri kullanıyorsak problem olmaz. taşıdığı bilgi duruyor. model için önemli.



###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()


###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

# robust scaler standart scaler ' a göre aykırı değerlere dayanıklı olduğundan dolayı daha tercih edilebilir.
# aykırı değerlerden etkilenmiyor !
# standart scaler ' a göre daha kullanışlı -önerilen -

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head() #yaş değişkeni standartlaştırılmış dönüştürülmüş formlarıyla gelmiş oldu.

age_cols = [col for col in df.columns if "Age" in col]


# bir sayısal değişkenin çeyreklik değerlerini göstermek ve histogram grafiğini göstermek.
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)

# qcut metodu => bir değişkenin değerlerini küçükten büyüğe doğru sıralar ve çeyrek değerlere göre x parçaya ayırır.

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################
# yeni bir şeyler türetmeye çalışıyoruz, var olanı değiştirmiyoruz !! ( encoding kısmı değiştirme idi.)



df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int') # not null true false geleceğinden dolayı 1-0'a çevirmek için astype("int")

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"}) #bağımlı değişken ( survived ) ' a göre bir ort. al

# bağımlı değişken ile bu feature'un arasında istatistiki olarak ilişkisi için oran testi
# 1- başarı sayısı 2- gözlem sayısı

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(), # cabin numarası olup hayatta kalan kaç kişi var
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# proportion z testi için ho : p1 ve p2 oranlarının arasında fark yoktur.
# h0 hipotezi p value değeri 0.05ten küçük old. için reddedilir!!! fark var !!


df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df.head()

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

###################
# Regex ile Değişken Türetmek
###################

# useful

df.head()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year # dt => import edilen datetime dan geliyor . dt.year => dt modülünü kullanarak yılı getir

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################

# değişkenlerin birbirleriyle etkileşime girmesi demek. iki değişkenin çarpılması, toplanması

df = load()
df.head()

# yaşı büyük olanların ya da küçük olanların yolculuk sınıflarına göre refah düzeyini ölçme

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['SEX'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['SEX'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['SEX'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['SEX'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()


#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)


df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))


df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#############################################
# 4. Label Encoding
#############################################

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#############################################
# 5. Rare Encoding
#############################################

rare_analyser(df, "SURVIVED", cat_cols)


df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


#############################################
# 8. Model
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


