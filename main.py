# %%
from knnClassfication import *
# %% 출력 데이터 시각화 
print("변경 전")
print(X.ndim)
print(X.shape)

images = X[0:5]
labels = y[0:5]
num_row = 1
num_col = 5  # plot images
fig, axes = plt.subplots(num_row,
                         num_col,
                         figsize=(1.5 * num_col, 2 * num_row))
for i in range(5):
    ax = axes[i % num_col]  #, i % num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

print("변경 후")
print(X_data.ndim)
print(X_data.shape)
colors = ['red', 'blue', 'black', 'orange', 'green']
for i in range(5):
    images = np.unique(X_data[i])
    plt.plot(images,
             '.',
             color=colors[i],
             label='label : {0}'.format(y_data[i]))
plt.legend(loc='best')
plt.show()

# %%
# 학습 모듈 가져오기
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
seed = 2021
# %%
# 배깅

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
# 결정 트리 시각화
from sklearn.tree import export_graphviz
from graphviz import Source
export_graphviz(
    tree_clf,
    out_file="tree.dot",
    #feature_names=.feature_names,
    class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    rounded=True,
    filled=True)

Source.from_file("tree.dot")

# %%
# 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500,
                                 max_leaf_nodes=16,
                                 random_state=seed)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))

# %%
# 에이다부스트
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                             n_estimators=200,
                             learning_rate=0.5,
                             random_state=seed)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_ada))

# %%
## KNN
t = KnnClassification()
t.start()
print("successfully")

# %%
