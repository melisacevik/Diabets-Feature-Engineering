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

df.columns= [col.upper() for col in df.columns]
df.columns
# Adım 1: Genel resmi inceleyiniz.

def check_df(dataframe, head=5):
    print("-" * 25 + "Shape" + "-" * 25)
    print(dataframe.shape)
    print("-" * 25 + "Types" + "-" * 25)
    print(dataframe.dtypes)
    print("-" * 25 + "The First data" + "-" * 25)
    print(dataframe.head(head))
    print("-" * 25 + "The Last data" + "-" * 25)
    print(dataframe.tail(head))
    print("-" * 25 + "Missing values" + "-" * 25)
    print(dataframe.isnull().sum())
    print("-" * 25 + "Describe the data" + "-" * 25)
    # Sayısal değişkenlerin dağılım bilgisi
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("-" * 25 + "Distinct Values" + "-" * 25)
    print(dataframe.nunique())

check_df(df)

## yaş için histogram grafiği

plt.figure(figsize=(8,7))
plt.xlabel("AGE", fontsize=10)
plt.ylabel("Count", fontsize=10)
df["AGE"].hist(edgecolor="black")
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
                   dataframe[col].dtypes != "O"] 
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"] 
    cat_cols = cat_cols + num_but_cat 
    cat_cols = [col for col in cat_cols if col not in cat_but_car] 

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] 
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def density_graph(df):
    fig, ax = plt.subplots(4, 2, figsize=(20, 20))
    variables = df.columns.tolist()
    for i in range(4): #4 satır
            for j in range(2): #2 sütun
                sns.distplot(df[variables[i*2 + j]], bins=20, ax=ax[i, j], color="red")

density_graph(df)
plt.show()

#sns.pairplot(df)
#plt.show()


# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
    print("-" * 30)

df["OUTCOME"].value_counts()

for col in num_cols:
    target_summary_with_num(df, "OUTCOME", col)


# incelediğimiz sonuca göre diyabet hastası olan kişilerin bütün değerleri, olmayanlara göre yüksek.

# Adım 5: Aykırı gözlem analizi yapınız.


def outlier_graph(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    plt.figure(figsize=(15, 20))
    for i, column in enumerate(num_cols):
        plt.subplot(len(num_cols) // 2 + 1, 2, i+1)
        sns.boxplot(x=df[column], color="red")
        plt.title(f"Boxplot of {column}", pad=20)

    plt.tight_layout()

outlier_graph(df)
plt.show()

# Adım 6: Eksik gözlem analizi yapınız.

msno.bar(df)
plt.show()


#####

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
p = sns.pairplot(df, hue="OUTCOME")
plt.show()

# Adım 7: Korelasyon analizi yapınız.

corr = df.corr()
corr_df= corr.unstack().sort_values(ascending=False)
corr_df= pd.DataFrame(corr_df)
corr_df.reset_index(inplace=True)
corr_df.columns=["var1","var2","corr"]
corr_df[(corr_df["var1"]=="OUTCOME") & (corr_df["corr"].apply(lambda x: x!=1))].head()


sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()




# Base Model

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


# Görev 2 : Feature Engineering

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde
# 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır.
# Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp
# sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

df[num_cols].describe().T

selected_list =["GLUCOSE","SKINTHICKNESS","INSULIN","BMI","BLOODPRESSURE"]

for col in selected_list:
    df[col] = df[col].apply(lambda x: np.nan if x == 0 else x)

df.isnull().sum()
# 0 olan değerler NaN ile dolduruldu.

# eksik değerlerin yerine ortalama değeri atama.

for col in selected_list:
    df[col] = df[col].fillna(df.groupby("OUTCOME")[col].transform("mean"))

df.isnull().sum()

# aykırı değer tespiti


def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

#bütün col için
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col ,check_outlier(df,col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col)) # aykırı değerleri gözlemleme

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 2: Yeni değişkenler oluşturunuz.

df["AGE_NEW"]= pd.cut(df["AGE"],bins= [20,45,max(df["AGE"])], labels=["mature", "senior"])
df["GLUCOSE_NEW"]= pd.cut(df["GLUCOSE"], bins=[0, 100, 140 , max(df["GLUCOSE"])], labels=["low", "normal", "high"])
df["BMI_NEW"]=pd.cut(df["BMI"], bins=[18,25,32,max(df["BMI"])], labels=["Normal Weight","Overweight","Obese"])
df.loc[df["INSULIN"]<=130,"INSULIN_NEW"]="normal"
df.loc[df["INSULIN"]>130, "INSULIN_NEW"]="anormal"

df["GLUCOSE_INSULIN"]=df["GLUCOSE"]*df["INSULIN"]
df["INSULIN_BMI"]=df["INSULIN"]*df["BMI"]
df["GLUCOSE_BLOODPRESSURE"]= df["GLUCOSE"]* df["BLOODPRESSURE"]
df["INSULIN_BLOODPRESSURE"]= df["INSULIN"]*df["BLOODPRESSURE"]

# AGE - GLUCOSE

df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="low"),"AGE_GLUCOSE"]="mature-low"
df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="normal"),"AGE_GLUCOSE"]="mature-normal"
df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="high"),"AGE_GLUCOSE"]="mature-high"

df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="low"),"AGE_GLUCOSE"]="senior-low"
df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="normal"),"AGE_GLUCOSE"]="senior-normal"
df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="high"),"AGE_GLUCOSE"]="senior-high"

df.columns

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols=[ col for col in cat_cols if col != "OUTCOME"]

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in cat_cols if df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df,col)

# one hot encoder

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if col not in binary_cols]

df = one_hot_encoder(df, cat_cols, drop_first=True)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df[num_cols].head()

# Adım 5: Model oluşturunuz.

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)

# 1 : Random Forest Classifier
rf_model = RandomForestClassifier(random_state=46)
rf_model.fit(X_train, y_train)
y_pred_1 = rf_model.predict(X_test)
rf_accuracy= accuracy_score(y_pred_1, y_test)
print(rf_accuracy)

# 2 : Light Gradient Boosting Machine Classifier

from lightgbm import LGBMClassifier

lgbm_model= LGBMClassifier(random_state=42, verbosity=-1)
lgbm_model.fit(X_train, y_train)
y_pred_2 =lgbm_model.predict(X_test)
lgbm_accuracy= accuracy_score(y_pred_2, y_test)
print(lgbm_accuracy)

# 3 : K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

knn_model= KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_6 =knn_model.predict(X_test)
knn= accuracy_score(y_pred_6, y_test)
print(knn)


# Görselleştir

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_model, X_train)
