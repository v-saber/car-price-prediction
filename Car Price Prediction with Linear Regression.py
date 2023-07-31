#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# In[2]:


# pd.set_option("display.max_rows", None)


# In[3]:


data = pd.read_csv("D:\programming\Machine Learning/cardata.csv")
# data


# In[4]:


df_cars = pd.DataFrame(data)
df_cars


# In[5]:


df_cars.describe()


# In[6]:


df_cars.info()


# In[7]:


df_cars.isna().any()


# In[8]:


df_cars.columns.to_list()


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# ستون سال ساخت خودرو را به ستون عمر خودرو تغییر می دهیم

# In[9]:


Age = []
for year in df_cars["Year"]:
    year = df_cars["Year"].max() + 1 - year
    Age.append(year)

df_cars.insert(0, "Age", Age)


# In[10]:


df_cars.drop(columns="Year", inplace=True)


# In[11]:


df_cars


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# برای پیدا کردن نویزها و میسینگ ولیوها نمودارها را رسم می نماییم

# In[12]:


# به جهت صرفه جویی در زمان، تابعی تعریف می کنیم که نمودارها را با تنظیمات دلخواه ما رسم کند

def one_scatter(df_name, x_ax_name, y_ax_name):
    
    scatter_name = f"{y_ax_name}-{x_ax_name}"
    
    fig_output_name = scatter_name
    
#     plt.figure(figsize=(15,15), dpi=80)
    
    plt.title(f"{x_ax_name} - {y_ax_name}\n", fontsize=30 )
    
    scatter_name = plt.scatter(df_name[x_ax_name], df_name[y_ax_name])
    
    scatter_name.axes.tick_params(gridOn=True, size=12, labelsize=10)
    
    plt.xlabel(f"\n{x_ax_name}", fontsize=20)
    plt.ylabel(f"{y_ax_name}\n", fontsize=20)
    
    plt.xticks(rotation=90)


# In[13]:


one_scatter(df_cars, df_cars.columns[0], df_cars.columns[2])


# In[14]:


# یک داده پرت داریم
# Age = 9 در
# برای پیدا کردن آن دیتافریم را دوباره می چینیم
# بعداً مدلی می سازیم که این داده را نداشته باشد و این دو حالت را مقایسه می کنیم


# In[15]:


df_cars.sort_values(by="Selling_Price", ascending=False)


# In[16]:


one_scatter(df_cars, df_cars.columns[1], df_cars.columns[2])


# In[17]:


one_scatter(df_cars, df_cars.columns[3], df_cars.columns[2])


# In[18]:


# یک داده پرت داریم
# Present_Price = 92.60
# برای پیدا کردن آن دیتافریم را دوباره می چینیم
# بعداً مدلی می سازیم که این داده را نداشته باشد و این دو حالت را مقایسه می کنیم


# In[19]:


df_cars.sort_values(by="Present_Price", ascending=False)


# In[20]:


one_scatter(df_cars, df_cars.columns[4], df_cars.columns[2])


# In[21]:


# یک داده پرت داریم
# Kms_Driven = 500,000
# برای پیدا کردن آن دیتافریم را دوباره می چینیم
# بعداً مدلی می سازیم که این داده را نداشته باشد و این دو حالت را مقایسه می کنیم


# In[22]:


df_cars.sort_values(by="Kms_Driven", ascending=False)


# In[23]:


one_scatter(df_cars, df_cars.columns[5], df_cars.columns[2])


# In[24]:


# تنها ۲ نمونه داده برای سوخت سی ان جی داریم

# می توانیم در آینده با توجه به تعداد کم آن ها در مقایسه با سایر سوخت ها، از آن ها صرف نظر کنیم و 
# مدلی بسازیم که فقط برای سوخت های بنزین و دیزل کاربرد داشته باشد

# ولی در حال حاضر و در تحلیل پیش رو آن را حذف نمی کنیم


# In[25]:


df_cars.sort_values(by="Fuel_Type")


# In[26]:


one_scatter(df_cars, df_cars.columns[6], df_cars.columns[2])


# In[27]:


one_scatter(df_cars, df_cars.columns[7], df_cars.columns[2])


# In[28]:


one_scatter(df_cars, df_cars.columns[8], df_cars.columns[2])


# In[29]:


# به نظر می رسد که یک داده پرت داریم ولی با توجه به تعداد کم نمونه های فیچر (تنها سه نمونه داریم) آن را حذف نمی کنیم


# In[30]:


df_cars.sort_values(by="Owner", ascending=False)


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# گفته شد که در تحلیل به ستون نام خودرو نیازی نداریم
# 

# In[31]:


df = df_cars.drop(columns=["Car_Name"])
df


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# در ابتدا آموزش مدل را با وجود داده های پرت انجام می دهیم
# و سپس آن را با حالت بدون داده های پرت مقایسه می کنیم

# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# حل مسئله با روش get_dummies تا نتیجه آن را با one hot encoding مقایسه نماییم

# In[32]:


df_dum = pd.get_dummies(df, columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=True)
df_dum


# In[33]:


# مشخص کردن فیچرها و تارگت


# In[34]:


X = df_dum.drop("Selling_Price", axis=1)
y = pd.DataFrame(df_dum, columns=["Selling_Price"])
y = y.values.reshape(-1, 1)


# In[35]:


# تعیین داده های آموزش و تست


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[37]:


# ایجاد مدل


# In[38]:


regressor_dum = LinearRegression()


# In[39]:


# فیت کردن مدل بر روی داده های آموزش


# In[40]:


regressor_dum.fit(X_train, y_train)


# In[41]:


y_pred = regressor_dum.predict(X_test)


#  

#  

# In[42]:


##################################################################################


# In[43]:


# را در دیتافریم فیچرها وارد می کنیم تا بتوانیم (y_pred) و تارگت پیش بینی شده (y_test) داده های تارگت تست
# نمودار رسم کنیم و ببینیم از روی نمودار می توان به نتایجی دست یافت یا نه؟


# In[44]:


# X_test
X_test.shape


# In[45]:


X_test.insert(8, "y_test", y_test)
X_test.insert(9, "y_pred", y_pred)
X_test


# In[46]:


df_2 = X_test.sort_values(by="Age", ascending=True)
# df2


# In[47]:


a = df_2.Age
b = df_2.y_test
c = df_2.Age
d = df_2.y_pred
plt.xlabel("Age")
plt.ylabel("Price")
plt.scatter(a, b)
plt.scatter(c, d)
plt.plot(c, d, color="red", marker="o", ms=5)
plt.show()


# In[48]:


# بر حسب یکدیگر y رسم نمودارهای

plt.scatter(y_test, y_pred)
plt.grid()
plt.xlabel("Test")
plt.ylabel("Prediction")


# In[49]:


# دستور زیر را برای این می نویسیم که تارگت های حقیقی و پیش بینی شده را کنار هم ببینیم برای مقایسه ابتدایی


# In[50]:


Compare = pd.DataFrame({"Actual": y_test.flatten(), "Predict": y_pred.flatten()})
Compare


# In[51]:


# در پیش بینی قیمت عدد منفی داریم!


# In[52]:


##################################################################################


#  

#  

# In[53]:


# عرض از مبدأ و ضرایب معادله رگرشن


# In[54]:


print(regressor_dum.intercept_)
print(regressor_dum.coef_)


# In[55]:


# چون داده ها نرمال سازی نشده اند امکان تحلیل ضرایب معادله رگرشن را نداریم
# در انتهای برنامه داده ها را نرمال سازی می کنیم و ضرایب معادله رگرشن را تحلیل می کنیم


# In[56]:


# محاسبه متریک ها در این دیتاست


# In[57]:


print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))


# In[58]:


# داده ای که نیاز به پیش بینی دارد


# In[59]:


df_sample = pd.DataFrame({"Age": 10,
                        "Kms_Driven": 42000,
                        "Fuel_Type_Diesel": 0,
                        "Transmission_Manual": 1,
                        "Seller_Type_Individual": 0,
                        "Owner": 1,
                        "Present_Price": [11.23],
                        "Fuel_Type_Petrol": 1,
                        "Selling_Price": 99})
df_sample


# In[60]:


# با انجام دستورات سلول زیر
# مطابق دیتافریم اصلی می شود df_sample اولاً ترتیب چینش ستون های
# های آموزش و پردیکت را تعریف کنیم y ها و X ثانیاً به راحتی در سلول بعد از آن می توانیم


# In[61]:


# df4 = pd.concat([df_dum, df_sample], axis=0)
df4 = pd.concat([df_dum, df_sample])
df_sample = df4.iloc[df.shape[0]:]
df_sample


# In[62]:


# حال داده های آموزش را از کل دیتافریم انتخاب می کنیم و داده تست را هم از داده های جدیدی که صورت سؤال داده است


# In[63]:


X_train = df_dum.drop("Selling_Price", axis=1)
y_train = pd.DataFrame(df_dum, columns=["Selling_Price"]).values.reshape(-1, 1)
X_test = df_sample.drop("Selling_Price", axis=1)


# In[64]:


# مدل را با دیتاست اصلی آموزش می دهیم


# In[65]:


regressor_dum.fit(X_train, y_train)
y_pred_dummies_outliers = regressor_dum.predict(X_test)


# In[66]:


# نتیجه پیش بینی مدل بر روی داده جدید


# In[67]:


print("قیمت پیش بینی شده:")
print(float(y_pred_dummies_outliers))


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# حل مسئله با روش one hot encoding با وجود داده های پرت یا همون outlier

# In[68]:


df


# In[69]:


onehot = OneHotEncoder(sparse=False, drop="first")


# In[70]:


columns_to_encode = ["Fuel_Type", "Seller_Type", "Transmission"]


# In[71]:


X_onehot = onehot.fit_transform(df[columns_to_encode])
X_onehot


# In[72]:


onehot.categories_


# In[73]:


# دو راه داریم برای به دست آوردن نام جدید ستون های انکد شده
# while راه ۱: با نوشتن حلقه
# راه ۲: با دستوری در کتابخانه سایکیت لرن
# هر دو راه در زیر آمده است


# In[74]:


# راه ۱


# In[75]:


columns_encoded = []
category_list = []

j = 0
while j < len(onehot.categories_):
    k = 0
    while k < (len(onehot.categories_[j]))-1:
        category_list.append(onehot.categories_[j][k])
        column_name = f"{columns_to_encode[j]}_{onehot.categories_[j][k]}"
        columns_encoded.append(column_name)
        k += 1
    j += 1
    
    
print("category_list:", category_list)
print()
print("columns:", columns_encoded)


# In[76]:


# راه ۲


# In[77]:


columns_encoded = onehot.get_feature_names_out()
# columns_encoded = onehot.get_feature_names_out(onehot.feature_names_in_)
columns_encoded


# In[78]:


# حل را از راه ۲ ادامه می‌دهیم


# In[79]:


# در ادامه فیچرهای انکد شده را به دیتافریم اصلی اضافه می کنیم


# In[80]:


df_X = pd.DataFrame(X_onehot, columns=columns_encoded)
df_onehot = pd.concat([df, df_X], axis=1)
df_onehot = df_onehot.drop(columns_to_encode, axis=1) 
df_onehot


# In[81]:


# فیچرها و تارگت را مشخص می نماییم


# In[82]:


X = df_onehot.drop("Selling_Price", axis=1)
y = pd.DataFrame(df_onehot, columns=["Selling_Price"])
y = y.values.reshape(-1, 1)


# In[83]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[84]:


regressor_onehot_1 = LinearRegression()


# In[85]:


regressor_onehot_1.fit(X_train, y_train)


# In[86]:


y_pred = regressor_onehot_1.predict(X_test)


# In[87]:


print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

score_onehot_outliers = metrics.r2_score(y_test, y_pred)


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
#     در این جا cross validation نیز انجام می‌دهیم تا ببینیم دقت آن تا چه حد به دقت مدل نزدیک است.

# In[88]:


model = LinearRegression()

kfold_val = KFold(5, shuffle=True, random_state=0)

results = cross_val_score(model, X, y, cv=kfold_val)

print(results)
print(np.mean(results))

A = np.mean(results)
A


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
#     داده‌ها را نرمال سازی می‌کنیم تا بتوانیم ضرایب معادله رگرشن را تفسیر نماییم.

# In[89]:


# Standard Scaling


# In[90]:


Scaler = StandardScaler()
X_train_scaling = Scaler.fit_transform(X_train)
X_test_scaling = Scaler.transform(X_test)


# In[91]:


regressor_onehot_1.fit(X_train_scaling, y_train)
y_pred_onehot_outliers = regressor_onehot_1.predict(X_test_scaling)


# In[92]:


print(y_pred_onehot_outliers)


# In[93]:


print(regressor_onehot_1.intercept_)
print(regressor_onehot_1.coef_)


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# ضرایبی که بزرگ‌تر از ۱ هستند تأثیر بیش‌تری در دقت پیش‌بینی مدل دارند.

# In[94]:


# داده ای که نیاز به پیش بینی دارد


# In[95]:


df_sample = pd.DataFrame({"Age": 10,
                    "Kms_Driven": 42000,
                    "Fuel_Type_Diesel": 0,
                    "Transmission_Manual": 1,
                    "Seller_Type_Individual": 0,
                    "Owner": 1,
                    "Present_Price": [11.23],
                    "Selling_Price": 99,
                    "Fuel_Type_Petrol": 1,
                   })
df_sample


# In[96]:


df4 = pd.concat([df_onehot, df_sample])
df_sample = df4.iloc[df4.shape[0]-1:]
df_sample


# In[97]:


X_train = df_onehot.drop("Selling_Price", axis=1)
y_train = pd.DataFrame(df_onehot, columns=["Selling_Price"]).values.reshape(-1, 1)
X_test = df_sample.drop("Selling_Price", axis=1)


# In[98]:


regressor_onehot_1.fit(X_train, y_train)
y_pred_onehot_outliers = regressor_onehot_1.predict(X_test)


# In[99]:


print("قیمت پیش بینی شده:")
print(float(y_pred_onehot_outliers))


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# حل مسئله با روش ordinal encoding با وجود داده های پرت یا همون outlier

# In[100]:


df


# In[101]:


orde = OrdinalEncoder()


# In[102]:


columns_to_encode = ["Fuel_Type", "Seller_Type", "Transmission"]


# In[103]:


X_ordinal = orde.fit_transform(df[columns_to_encode])
X_ordinal


# In[104]:


X_ordinal.shape


# In[105]:


orde.categories_


# In[106]:


# با دستور زیر متوجه می شویم که به هر کتگوری چه عددی نسبت داده شده است

orde.inverse_transform([[0, 0, 0], [1, 1, 1], [2, 1, 1]])


# In[107]:


df_ord = df.copy()
df_X = pd.DataFrame(X_ordinal, columns=columns_to_encode)

df_ord[columns_to_encode] = df_X[columns_to_encode]
# df_ord[columns_to_encode] = df_X[columns_to_encode].values

df_ord


# In[108]:


X = df_ord.drop("Selling_Price", axis=1)
y = pd.DataFrame(df_ord, columns=["Selling_Price"])
y = y.values.reshape(-1, 1)


# In[109]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[110]:


regressor_ord_1 = LinearRegression()


# In[111]:


regressor_ord_1.fit(X_train, y_train)


# In[112]:


y_pred = regressor_ord_1.predict(X_test)


# In[113]:


print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

score_ordinal_outliers = metrics.r2_score(y_test, y_pred)


# In[114]:


model = LinearRegression()

kfold_val = KFold(5, shuffle=True, random_state=0)

results = cross_val_score(model, X, y, cv=kfold_val)

print(results)
print(np.mean(results))
B = np.mean(results)
B


# In[115]:


df_sample = pd.DataFrame({"Age": 10,
                    "Kms_Driven": 42000,
                    "Transmission": 1,
                    "Owner": 1,
                    "Present_Price": [11.23],
                    "Fuel_Type": 2,
                    "Seller_Type": 0,
                    "Selling_Price": 99})
df_sample


# In[116]:


df4 = pd.concat([df_ord, df_sample])
df_sample = df4.iloc[df4.shape[0]-1:]
df_sample


# In[117]:


X_train = df_ord.drop("Selling_Price", axis=1)
y_train = pd.DataFrame(df_ord, columns=["Selling_Price"]).values.reshape(-1, 1)
X_test = df_sample.drop("Selling_Price", axis=1)


# In[118]:


regressor_ord_1.fit(X_train, y_train)
y_pred_ordinal_outliers = regressor_ord_1.predict(X_test)


# In[119]:


print("قیمت پیش بینی شده:")
print(float(y_pred_ordinal_outliers))


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# در مرحله بعد داده های پرت را حذف می کنیم و دوباره مدل سازی می کنیم و دقت را می سنجیم

# In[120]:


df_cars_2 = df_cars[df_cars["Present_Price"]<80]
df_cars_2 = df_cars_2[df_cars_2["Kms_Driven"]<400000]
# df_cars_2 = df_cars_2[df_cars_2["Owner"]<3.0]

df_cars_2.reset_index(drop=True, inplace=True)


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# نمودارهای دارای داده پرت را قبل و بعد از حذف داده های پرت رسم می کنیم که بفهمیم چه تغییری کرده اند

# In[121]:


def plt_subplot(column_number_1, column_number_2):
    
    plt.subplot(1, 2, 1)
    one_scatter(df_cars, df_cars.columns[column_number_1], df_cars.columns[column_number_2])
    plt.title(f"{df_cars.columns[column_number_1]} - {df_cars.columns[column_number_2]} \n\nbefore\n", fontsize=20)

    plt.subplot(1, 2, 2)
    one_scatter(df_cars_2, df_cars_2.columns[column_number_1], df_cars_2.columns[column_number_2])
    plt.title(f"{df_cars.columns[column_number_1]} - {df_cars.columns[column_number_2]} \n\nafter\n", fontsize=20)

    plt.tight_layout()

    plt.subplots_adjust(right=1.4)


# In[122]:


plt_subplot(0, 2)


# In[123]:


plt_subplot(3, 2)


# In[124]:


plt_subplot(4, 2)


# In[125]:


# حذف ستون نام خودرو

df2 = df_cars_2.drop(columns=["Car_Name"])
df2.shape


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# حالت اول
# 
# OneHotEncoding

# In[126]:


onehot = OneHotEncoder(sparse=False, drop="first")


# In[127]:


columns_to_encode = ["Fuel_Type", "Seller_Type", "Transmission"]


# In[128]:


X_onehot = onehot.fit_transform(df2[columns_to_encode])
X_onehot


# In[129]:


onehot.feature_names_in_


# In[130]:


onehot.n_features_in_


# In[131]:


onehot.categories_


# In[132]:


columns_encoded = onehot.get_feature_names_out()
# columns_encoded = onehot.get_feature_names_out(onehot.feature_names_in_)

columns_encoded


# In[133]:


df_X = pd.DataFrame(X_onehot, columns=columns_encoded)
df_onehot = pd.concat([df2, df_X], axis=1)
df_onehot = df_onehot.drop(columns_to_encode, axis=1) 
df_onehot


# In[134]:


X = df_onehot.drop("Selling_Price", axis=1)
y = pd.DataFrame(df_onehot, columns=["Selling_Price"])
y = y.values.reshape(-1, 1)


# In[135]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[136]:


regressor_onehot_2 = LinearRegression()


# In[137]:


regressor_onehot_2.fit(X_train, y_train)


# In[138]:


y_pred = regressor_onehot_2.predict(X_test)


# In[139]:


print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

score_onehot_clean_data = metrics.r2_score(y_test, y_pred)


# In[140]:


model = LinearRegression()

kfold_val = KFold(5, shuffle=True, random_state=0)

results = cross_val_score(model, X, y, cv=kfold_val)

print(results)
print(np.mean(results))
C = np.mean(results)
C


# In[141]:


df_sample = pd.DataFrame({"Age": 10,
                    "Kms_Driven": 42000,
                    "Fuel_Type_Diesel": 0,
                    "Transmission_Manual": 1,
                    "Seller_Type_Individual": 0,
                    "Owner": 1,
                    "Present_Price": [11.23],
                    "Fuel_Type_Petrol": 1,
                    "Selling_Price": 99})
df_sample


# In[142]:


df4 = pd.concat([df_onehot, df_sample])
df_sample = df4.iloc[df4.shape[0]-1:]
df_sample


# In[143]:


X_train = df_onehot.drop("Selling_Price", axis=1)
y_train = pd.DataFrame(df_onehot, columns=["Selling_Price"]).values.reshape(-1, 1)
X_test = df_sample.drop("Selling_Price", axis=1)


# In[144]:


regressor_onehot_2.fit(X_train, y_train)
y_pred_onehot_clean_data = regressor_onehot_2.predict(X_test)


# In[145]:


print(y_pred_onehot_clean_data)


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# حالت دوم
# 
# OrdinalEncoding

# In[146]:


df2


# In[147]:


orde = OrdinalEncoder()


# In[148]:


columns_to_encode = ["Fuel_Type", "Seller_Type", "Transmission"]


# In[149]:


X_ordinal = orde.fit_transform(df2[columns_to_encode])
X_ordinal


# In[150]:


X_ordinal.shape


# In[151]:


orde.categories_


# In[152]:


orde.inverse_transform([[0, 0, 0], [1, 1, 1], [2, 1, 1]])
# orde.inverse_transform([orde.categories_[0]])


# In[153]:


df_ord = df2.copy()
df_X = pd.DataFrame(X_ordinal, columns=columns_to_encode)

df_ord[columns_to_encode] = df_X[columns_to_encode]
# df_ordd[columns_to_encode] = df_X[columns_to_encode].values

df_ord


# In[154]:


X = df_ord.drop("Selling_Price", axis=1)
y = pd.DataFrame(df_ord, columns=["Selling_Price"])
y = y.values.reshape(-1, 1)


# In[155]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[156]:


regressor_ord_2 = LinearRegression()


# In[157]:


regressor_ord_2.fit(X_train, y_train)


# In[158]:


y_pred = regressor_ord_2.predict(X_test)


# In[159]:


print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

score_ordinal_clean_data =  metrics.r2_score(y_test, y_pred)


# In[160]:


model = LinearRegression()

kfold_val = KFold(5, shuffle=True, random_state=0)

results = cross_val_score(model, X, y, cv=kfold_val)

print(results)
print(np.mean(results))
D = np.mean(results)
D


# In[161]:


df_sample = pd.DataFrame({"Age": 10,
                        "Kms_Driven": 42000,
                        "Transmission": 1,
                        "Owner": 1,
                        "Present_Price": [11.23],
                        "Fuel_Type": 2,
                        "Seller_Type": 0,
                        "Selling_Price": 99
                         })
df_sample


# In[162]:


df4 = pd.concat([df_ord, df_sample])
df_sample = df4.iloc[df4.shape[0]-1:]
df_sample


# In[163]:


X_train = df_ord.drop("Selling_Price", axis=1)
y_train = pd.DataFrame(df_ord, columns=["Selling_Price"]).values.reshape(-1, 1)
X_test = df_sample.drop("Selling_Price", axis=1)


# In[164]:


regressor_ord_2.fit(X_train, y_train)
y_pred_ordinal_clean_data = regressor_ord_2.predict(X_test)


# In[165]:


print(y_pred_ordinal_clean_data)


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# مقایسه تمامی قیمت های پیش بینی شده
# 

# In[166]:


print("پیش از حذف داده‌های پرت")
print("y_pred_onehot_outliers:", round(float(y_pred_onehot_outliers), 6))
print("y_pred_ordinal_outliers:", round(float(y_pred_ordinal_outliers), 6))
print()
print("پس از حذف داده‌های پرت")
print("y_pred_onehot_clean_data:", round(float(y_pred_onehot_clean_data), 6))
print("y_pred_ordinal_clean_data:", round(float(y_pred_ordinal_clean_data), 6))


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# مقایسه تمامی دقت های پیش بینی شده
# 

# In[167]:


print("پیش از حذف داده‌های پرت")
print("score_onehot_outliers:", round(float(score_onehot_outliers), 6))
print("score_ordinal_outliers:", round(float(score_ordinal_outliers), 6))
print()
print("پس از حذف داده‌های پرت")
print("score_onehot_clean_data:", round(float(score_onehot_clean_data), 6))
print("score_ordinal_clean_data:", round(float(score_ordinal_clean_data), 6))


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# مقایسه تمامی کراس ولیدیشن های انجام گرفته
# 

# In[168]:


print("پیش از حذف داده‌های پرت")
print("cross_val_score_onehot_outliers:", round(float(A), 6))
print("cross_val_score_ordinal_outliers:", round(float(B), 6))
print()
print("پس از حذف داده‌های پرت")
print("cross_val_score_onehot_clean_data:", round(float(C), 6))
print("cross_val_score_ordinal_clean_data:", round(float(D), 6))


#  

# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
#     
# Correlation

# In[169]:


df_ord.corr()


# In[170]:


df_onehot.corr()


#  

#  

# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# به دست آوردن و بررسی فیچرهای مرتبه بالاتر

# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
#     دو حالت قابل بررسی است:<br/>
#     حالت اول این که اثر تمامی مرتبه‌های بالاتر فیچرها را بررسی نماییم <br/>
#     حالت دوم این که فقط اثر توان‌های بالاتر (دوم، سوم و ...) فیچرها را بررسی کنیم

# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# حالت اول: اثر تمامی مرتبه‌های بالاتر فیچرها را بررسی می‌نماییم

# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# مرتبه‌های سوم را به دست می‌آوریم (مرتبه‌های بالاتر از ۳ تأثیری در افزایش دقت نداشتند):

# In[171]:


poly = PolynomialFeatures(degree=3, include_bias=False)

p = poly.fit_transform(df_onehot.drop("Selling_Price", axis=1))
# p

# p.shape

df_onehot_poly = pd.DataFrame(p, columns=poly.get_feature_names_out(df_onehot.drop("Selling_Price", axis=1).columns))
df_onehot_poly


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# فیچرهایی که باعث بهبود در دقت می‌شوند را پیدا می‌کنیم و به دیتافریم اضافه می‌کنیم

# In[172]:


score_onehot_clean_data_new = score_onehot_clean_data
df_onehot_2 = df_onehot.copy()
score_onehot_poly_clean_data = []

for i in range(df_onehot.shape[1]-1, df_onehot_poly.shape[1]):
    print("*************************************")
    print("مرحله:", i)
    X = df_onehot_2.drop("Selling_Price", axis=1)
    y = pd.DataFrame(df_onehot_2, columns=["Selling_Price"])
    y = y.values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor_onehot_2_poly = LinearRegression()
    regressor_onehot_2_poly.fit(X_train, y_train)
    y_pred = regressor_onehot_2_poly.predict(X_test)
    print("new R2 Score:", metrics.r2_score(y_test, y_pred))
    score_onehot_poly_clean_data.append(metrics.r2_score(y_test, y_pred))
    print("old R2 Scores:", score_onehot_clean_data_new)

#     چاپ تمامی دقت های به دست آمده
#     print("OneHot PolyFeatures Scores:", score_onehot_poly_clean_data)
    print(df_onehot_poly.columns[i])
    
    df_onehot_2 = pd.concat([df_onehot_2, df_onehot_poly[df_onehot_poly.columns[i]]], axis=1)

    if score_onehot_poly_clean_data[-1] >= score_onehot_clean_data_new:
        score_onehot_clean_data_new = score_onehot_poly_clean_data[-1]
    else:
        df_onehot_2 = df_onehot_2.iloc[:,:-1]


df_final = df_onehot_2


# In[173]:


print()
print("فیچرهای نهایی")
print(df_final.columns)

print()
print("Final Score:", score_onehot_clean_data_new)
print("Successful")


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
#     دیتافریم نهایی شامل فیچرهای مرتبه دومی که دقت را بهبود داده‌اند

# In[174]:


df_final

# df_final.shape
# df_final.corr()


# In[175]:


X = df_final.drop("Selling_Price", axis=1)
y = pd.DataFrame(df_final, columns=["Selling_Price"])
y = y.values.reshape(-1, 1)


# In[176]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[177]:


regressor = LinearRegression()


# In[178]:


regressor.fit(X_train, y_train)


# In[179]:


y_pred = regressor.predict(X_test)


# In[180]:


print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))


# In[181]:


model = LinearRegression()

kfold_val = KFold(5, shuffle=True, random_state=0)

results = cross_val_score(model, X, y, cv=kfold_val)

print(results)
print(np.mean(results))
E = np.mean(results)
E


#  

# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# می بینیم که score دیتافریم نهایی به score به دست آمده در cross validation خیلی نزدیک‌تر شد.<br/>
# دقت مدل در ابتدا و پیش از اعمال فیچرهای غیر خطی برای هر دو حالت onehot و ordinal برابر با 0.884 بود که پس از اضافه نمودن فیچرهای غیرخطی و در دیتافریم نهایی برابر با 0.979859 و در کراس ولیدیشن نهایی برابر با 0.963783 گردید.<br/>
# 

#  

# In[182]:


# برای این که ببینیم مدل روی داده های تست چگونه عمل می کند


# In[183]:


regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)


# In[184]:


regressor.score(X_test, y_test)


# In[185]:


# پیش بینی قیمت خودروی جدید


# In[186]:


df_sample = pd.DataFrame({"Age": 10,
                        "Kms_Driven": 42000,
                        "Fuel_Type_Diesel": 0,
                        "Transmission_Manual": 1,
                        "Seller_Type_Individual": 0,
                        "Owner": 1,
                        "Present_Price": [11.23],
                        "Fuel_Type_Petrol": 1,
                        "Selling_Price": 99,
                        "Age^2": 100,
                        "Age Present_Price": 112.3,
                        "Age Kms_Driven": 420000,
                        "Age Owner": 10,
                        "Age Fuel_Type_Diesel": 0
                         })
df_sample


# In[187]:


df4 = pd.concat([df_final, df_sample])
df_sample = df4.iloc[df4.shape[0]-1:]
df_sample


# In[188]:


X_train = df_final.drop("Selling_Price", axis=1)
y_train = pd.DataFrame(df_final, columns=["Selling_Price"]).values.reshape(-1, 1)
X_test = df_sample.drop("Selling_Price", axis=1)


# In[189]:


regressor.fit(X_train, y_train)
y_pred_clean_data = regressor.predict(X_test)


# In[190]:


print(float(y_pred_clean_data))


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# اگر بخواهیم تنها توان دوم فیچرها را به حساب آوریم به صورت زیر عمل می‌کنیم:

# In[191]:


# تعریف تابعی برای پیدا کردن توان های بالاتر ستون ها و اضافه کردن آن ها به دیتافریم اصلی در صورتی که در 
# اسکور نهایی تغییر مثبتی ایجاد کنند

# محاسبه می گردند j در تابع زیر، توان ها تا مرتبه

def dataframe_pow(df_onehot, score_onehot_clean_data):
    df_onehot_2 = df_onehot.copy()
    score_onehot_poly_clean_data = []
    for i in range(df_onehot.shape[1]-1):
#         print(i)

        for j in range(2, 7):
            X = df_onehot_2.drop("Selling_Price", axis=1)
            y = pd.DataFrame(df_onehot_2, columns=["Selling_Price"])
            y = y.values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            regressor_onehot_2_poly = LinearRegression()
            regressor_onehot_2_poly.fit(X_train, y_train)
            y_pred = regressor_onehot_2_poly.predict(X_test)
            score_onehot_poly_clean_data.append(metrics.r2_score(y_test, y_pred))

            df_pow = pd.DataFrame(df_onehot_2[df_onehot_2.columns[i]]**j)
            print(df_pow)
            df_pow = df_pow.rename(columns={df_onehot_2.columns[i]: f"{df_onehot_2.columns[i]}^{j}"})
            df_onehot_2 = pd.concat([df_onehot_2, df_pow], axis=1)

            if score_onehot_poly_clean_data[-1] >= score_onehot_clean_data:
                score_onehot_clean_data = score_onehot_poly_clean_data[-1]
            else:
                df_onehot_2 = df_onehot_2.iloc[:,:-1]
            
    print()
    print("First Score:", score_onehot_poly_clean_data[0])
    print("Final Score:", score_onehot_clean_data)
    print()
    print("Successful")
    
    return df_onehot_2

# df_pow
# df_onehot_2


# In[192]:


# استفاده از تابع بالا

df_onehot_poly = dataframe_pow(df_onehot, score_onehot_clean_data)
df_onehot_poly


# In[193]:


df_onehot_poly.columns


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
#     دیتافریمی را با ترکیب دو دیتافریم قبلی (df_final و df_onehot_poly) به دست آمده تشکیل می‌دهیم

# In[194]:


df_all = pd.concat([df_final, df_onehot_poly.iloc[:,10:]], axis=1)


# In[195]:


df_all


# In[196]:


df_all.columns


# In[197]:


X = df_all.drop("Selling_Price", axis=1)
y = pd.DataFrame(df_all, columns=["Selling_Price"])
y = y.values.reshape(-1, 1)


# In[198]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[199]:


regressor = LinearRegression()


# In[200]:


regressor.fit(X_train, y_train)


# In[201]:


y_pred = regressor.predict(X_test)


# In[202]:


print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))


# In[203]:


df_sample = pd.DataFrame({"Age": 10,
                        "Kms_Driven": 42000,
                        "Fuel_Type_Diesel": 0,
                        "Transmission_Manual": 1,
                        "Seller_Type_Individual": 0,
                        "Owner": 1,
                        "Present_Price": [11.23],
                        "Fuel_Type_Petrol": 1,
                        "Selling_Price": 99,
                        "Age^2": 100,
                        "Age Present_Price": 112.3,
                        "Age Kms_Driven": 420000,
                        "Age Owner": 10,
                        "Age Fuel_Type_Diesel": 0,
                        "Age^3": 1000,
                        "Age^4": 10000
                         })
df_sample


# In[204]:


df4 = pd.concat([df_all, df_sample])
df_sample = df4.iloc[df4.shape[0]-1:]
df_sample


# In[205]:


X_train = df_all.drop("Selling_Price", axis=1)
y_train = pd.DataFrame(df_all, columns=["Selling_Price"]).values.reshape(-1, 1)
X_test = df_sample.drop("Selling_Price", axis=1)


# In[206]:


regressor.fit(X_train, y_train)
y_pred_clean_data = regressor.predict(X_test)


# In[207]:


print(float(y_pred_clean_data))


# In[208]:


model = LinearRegression()

kfold_val = KFold(5, shuffle=True, random_state=0)

results = cross_val_score(model, X, y, cv=kfold_val)

print(results)
print(np.mean(results))
F = np.mean(results)
F


# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# قیمت پیش بینی شده بر اساس دیتافریم ترکیبی اخیر از قیمت پیش بینی شده در دیتافریم نهایی کمتر شد.<br/>
# دقت حاصل از عملیات کراس ولیدیشن بر دیتافریم اخیر نیز از دقت حاصل از دیتافریم نهایی کمتر شد.<br/>
# دلیل می‌تواند این باشد که دیتافریم نهایی که از حلقه for به دست آمد بر اساس دستوراتی که برای افزایش دقت تعریف نمودیم بود.<br/>
# ولی دیتافریم اخیر را از ترکیب دیتافریم نهایی با چند فیچر توان دوم از Age به دست آوردیم.<br/>
# (مردد بودم که می‌توان پس از عملیات کراس ولیدیشن داده‌های دارای دقت پایین را از دیتاست حذف کرد یا نه، به همین دلیل آن‌ها را حذف نکردم. بعداً و در حل جناب مؤمنی مشاهده کردم که ایشان این داده‌ها را حذف کردند.)<br/>
# 

#  

# <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#33FF8A">
# باید توجه داشت که این برنامه را می‌توان خلاصه‌تر نیز نوشت،<br/>
# برخی دستورات تکراری را حذف کرد،<br/>
# برخی دستورات را که پر استفاده هستند در قالب تابع نوشت،<br/>
# برخی دستورات را که خروجی خاصی برای نمایش ندارند در قالب تنها یک سلول نوشت و... .<br/>
# به دلیل مشابه بودن روند حل مسئله با شیوه تدریس آقای مهندس مؤمنی، برنامه به صورت فوق و با ذکر تمامی جزئیات نوشته شد.

#   

#  <div dir="rtl" style="font-family:B Nazanin; font-size:20px; color:#white">
#     این برنامه پیش از مشاهده حل جناب آقای مؤمنی (جلسه نوزدهم دوره) و همچنین بدون مشاهده نمونه حل‌های سایر دانشجویان دوره نوشته شده است.<br/>
#     با این که پس از کدنویسی و نوشتن برنامه، حل جناب مؤمنی و سایر دانشجویان دوره را مشاهده کردم، ولی در این برنامه تغییری ایجاد نکردم.<br/>
#     زیرا هدفم این است که فیدبکی که از سمت پشتیبانی دوره دریافت می‌کنم با توجه به عملکرد شخصی خودم باشد.<br/>
#     البته پس از ارسال فایل برنامه به ایمیل پشتیبانی دوره، و پیش از قرار دادن آن در اکانت‌های kaggle و github برنامه را با توجه به دانش‌های جدید به روز خواهم کرد.
#     

#  
