import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



data = pd.read_csv("D:/Desktop/faketr.csv")
copy = data.copy()
copy.describe()


copy.isna().sum()

copy.dropna(subset = ["text"],inplace  = True)
copy.isna().sum()
copy.head()



x = copy["text"].values
y = copy["label"].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25 , random_state = 0)

print(y_test.shape)
print(x_test.shape)
print(x_train.shape)
print(y_train.shape)


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(stop_words="english" , max_df=0.7)
tf_train = tf.fit_transform(x_train)
tf_test = tf.transform(x_test)

tf_test.shape

from sklearn.linear_model import PassiveAggressiveClassifier
pa = PassiveAggressiveClassifier()
pa.fit(tf_train,y_train)
pred  = pa.predict(tf_test)
print(accuracy_score(pred,y_test))





dt = pd.read_csv("D:/Desktop/takete.csv")
dt.isna().sum()
cp = dt.copy()

cp["text"] = cp["text"].fillna("missing")
cp.isna().sum()


test_x = cp["text"].values
tf_x = tf.transform(test_x)
y_p = pa.predict(tf_x)


len(y_p)
print(tf_x.shape)


repeat = pd.read_csv("D:/Desktop/submit.csv")
repeat.drop("label",inplace = True ,axis = 1)
repeat.insert(1,"label", y_p,True)
repeat.to_csv("fakenews.csv",index = False)




