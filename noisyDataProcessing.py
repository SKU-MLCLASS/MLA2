# %%
# 해당 코드에서는 노이즈 데이터를 구분하고 전처리하는 작업을 진행하였습니다.
# 모듈로드
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading

from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# %%
# 데이터셋 load
np.random.seed(2021)
(X, y), (_, _) = mnist.load_data()


# dataset reshape data
def data_reshape(X_digit, y_digit):
    X_digit = X_digit[:10000].reshape(-1, 28 * 28)
    y_digit = y_digit[:10000]
    return X_digit, y_digit


# 데이터 전처리 3차원에서 2차원 형태의 쉐잎을 갖도록, 데이터 형태 단순화
X_data, y_data = data_reshape(X, y)

# %%
# print(X_data.ndim)
# print(X_data.shape)
# colors = [
#     'red', 'blue', 'black', 'orange', 'green', 'grey', 'purple', 'pink',
#     'yellow', 'gold'
# ]
# for i in range(10):
#     images = np.unique(X_data[i])
#     print(i, "중복제거:", images)
#     print(i, "중복미제거:", X_data[i])
#     # plt.plot(images, '.', color=colors[y_data[i]])
# # plt.legend(loc='best')
# # plt.show()

# %%
for i in range(10000):
    temp = X_data[i]
    temp[temp < 10] = 0
    X_data[i] = temp
print("전처리 완료")
print(X_data[0])

# %%
# 데이터 셋 준비
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y_data,
                                                    train_size=0.8,
                                                    stratify=y_data,
                                                    shuffle=True,
                                                    random_state=2021)

# %%
# 배깅
seed = 2021
# %%
# 결정 트리 분류기 500개의 배깅 앙상블
bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=seed),
                            n_estimators=500,
                            max_samples=100,
                            bootstrap=True,
                            random_state=seed)  ## max samples를 제거하면 0.944
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
# %%

# 결정 트리 분류기 하나의 성능
tree_clf = DecisionTreeClassifier(
    random_state=seed)  # 시각적 표현을 진행하려면 max_depth 3으로
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))
# %%
## KNN
from knnClassfication import *
t = KnnClassification()
t.start()
print("successfully")

# %%
