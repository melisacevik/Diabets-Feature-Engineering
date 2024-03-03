# Veri Seti Hikayesi
#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies :Hamilelik sayısı
# Glucose :Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure :Kan Basıncı (Küçük tansiyon) (mm Hg)
# SkinThickness :Cilt Kalınlığı
# Insulin :2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction : Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI :Vücut kitle endeksi
# Age :Yaş (yıl)
# Outcome : Hastalığa sahip (1) ya da değil (0)

# Import

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

df = pd.read_csv('datasets/diabetes.csv')

# Adım 1: Genel resmi inceleyiniz.

def check_df(dataframe, head=5):
    print("######################## Shape ############################")
    print(dataframe.shape)
    print("######################## Types ############################")
    print(dataframe.dtypes)
    print("######################## Head #############################")
    print(dataframe.head(head))
    print("######################## Tail #############################")
    print(dataframe.tail(head))
    print("######################## NA ###############################")
    print(dataframe.isnull().sum())
    print("####################### Quantiles #########################")
    # Sayısal değişkenlerin dağılım bilgisi
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("####################### Value Count #########################")
    print(dataframe.value_counts())
    print("####################### Unique #########################")
    print(dataframe.nunique())

check_df(df)

df.groupby("Outcome").mean()
df["Outcome"].value_counts() * 100 / len(df)

plt.figure(figsize=(8,7))
plt.xlabel("Age", fontsize=10)
plt.ylabel("Count", fontsize=10)
df["Age"].hist(edgecolor="black")
plt.show()


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

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
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def density_graph(df):
    fig, ax = plt.subplots(4, 2, figsize=(20, 20)) #Bu satır, 4 satır ve 2 sütuna sahip bir figür oluşturur.
    variables = df.columns.tolist() #Bu satır, veri çerçevesindeki sütun isimlerini bir listeye dönüştürür. Böylece her bir sütuna erişmek için indeksleme yapabiliriz.
    for i in range(4): #4 satır
            for j in range(2): #sütun
                sns.distplot(df[variables[i*2 + j]], bins=20, ax=ax[i, j], color="red")

density_graph(df)
plt.show()

# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

#tek kategorim olduğu için hedef değişkene göre numerik değişkenin ortalamasını alıyorum.



df.head()
df.columns
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "Outcome", "Age")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# incelediğimiz sonuca göre diyabet hastası olan kişilerin bütün değerleri, olmayanlara göre yüksek.

# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    if df[(df[col_name] > up_limit)].any(axis=None):
        print(col_name, ": Yes")
    else:
        print(col_name, ": No")

    return up_limit, low_limit

outlier_thresholds(df, "Age")

for col in df.columns:
    outlier_thresholds(df,col)

plt.figure(figsize=(8,7))
sns.boxplot(x= df["Insulin"], color="red")
plt.show()

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age") # aykırı değer var

for col in df.columns:
    check_outlier(df,col)

# Adım 6: Eksik gözlem analizi yapınız.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] #eksik değerleri barındıran kolonlar

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) # eksik değer sayısı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False) # eksik değer oranı
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio']) # değişken isimlerini ve oranlarını birleştir
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

msno.bar(df, color="orange")
plt.show()

df.isnull().sum()
p = sns.pairplot(df, hue="Outcome")
plt.show()

# Adım 7: Korelasyon analizi yapınız.

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# Görev 2 : Feature Engineering

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde
# 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır.
# Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp
# sonrasında eksik değerlere işlemleri uygulayabilirsiniz.
