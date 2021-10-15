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
# 데이터 셋 준비
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y_data,
                                                    train_size=0.8,
                                                    stratify=y_data,
                                                    shuffle=True,
                                                    random_state=2021)


# data visualization
def mnist_visualization(w, h):
    fig, axes = plt.subplots(w, h)
    fig.set_size_inches(12, 6)
    # 784 만큼 반복 28*28 은 총 784개 노드로 이루어져 있으므로
    for i in range(w * h):
        # visualization --> X_digit(784) 노드를 i 만큼 반복해서 28의 축을 반전하여 이미지 생성
        axes[i // h, i % h].imshow(X_data[i].reshape(-1, 28))
        axes[i // h, i % h].set_title(f"label:{y_data[i]}",
                                      fontsize=20)  # 정답 레이블 설정
        axes[i // h,
             i % h].axis('off')  # 우리는 image visualization target 이므로 축을 필요 없음
    plt.tight_layout()
    plt.show()


# %%

# data set split ||| train -> 80% test -> 20%

# %%


class KnnClassification(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # 데이터 분포도 시각화 하기 위한 데이터셋 데이터프레임으로 변환
        self.y_train_df = pd.DataFrame(data=y_train, columns=['class'])
        self.y_test_df = pd.DataFrame(data=y_test, columns=['class'])
        self.k = KNeighborsClassifier

    def run(self):
        self.total_module()

    # knn 학습 거리값 3
    def classifier(self, k):
        kc = self.k(n_neighbors=k, n_jobs=-1)
        kc.fit(X_train, y_train)

        # 예측
        prediction = kc.predict(X_test)
        print(classification_report(y_test, prediction))
        print(f"prediction acc -> {(prediction == y_test).mean()}")
        print(f"mnist image classification -> {kc.score(X_test, y_test)}")

        def data_prediction():
            # result wrong data list comprehension
            result_data = []
            wrong_data = []
            total = [result_data, wrong_data]
            for n in range(0, len(y_test)):
                if prediction[n] == y_test[n]:
                    result_data.append(n)
                else:
                    wrong_data.append(n)

            # 25개 임의로 선택 ( 맞은거 틀린거 )
            for data in total:
                sample = random.choices(population=data, k=25)
                count = 0
                nrows = ncols = 5
                plt.figure(figsize=(20, 8))
                for n in sample:
                    count += 1
                    plt.subplot(nrows, ncols, count)
                    plt.imshow(X_test[n].reshape(28, 28),
                               cmap="Greys",
                               interpolation="nearest")
                    tmp = "Label:" + str(y_test[n]) + ", Prediction:" + str(
                        prediction[n])
                    plt.title(tmp)
                plt.tight_layout()
                plt.show()

        data_prediction()

    # 거리값 마다 학습 최적의 k 값 찾기
    def range_classifier(self):
        train_acc = []
        test_acc = []
        for neg in range(1, 20, 2):
            kc = self.k(neg)
            kc.fit(X_train, y_train)
            range_prediction = kc.predict(X_test)
            knn_score = kc.score(X_test, y_test) * 100
            print(f"knn range in {neg} classification -> {knn_score}")

            # 94% 이상의 결과값만 가져오기
            # 퍼센트 바꿔도 됨
            if knn_score >= 94.0:
                print(
                    f"knn test set range in {neg} classification -> {knn_score}"
                )
                print(classification_report(y_test, range_prediction))
                # confusion metric visualization
                label = [i for i in range(0, 10)]
                cm = plot_confusion_matrix(kc,
                                           X_test,
                                           y_test,
                                           display_labels=label,
                                           cmap="Blues",
                                           normalize="true")
                cm.ax_.set_title(
                    f"Knn mnist {neg} -- {knn_score}% confusion matrix")
                plt.show()
            else:
                print("No percent score 94 ↑")
                print()

            # score 값 리스트 저장
            train_acc.append(kc.score(X_train, y_train))
            test_acc.append(kc.score(X_test, y_test))

        return train_acc, test_acc

    # 거리값 마다 학습한 데이터 시각화
    def acc_grape_visualization(self):
        class_train, class_test = self.range_classifier()
        plt.plot(class_train, "*--", label="Knn mnist train score")
        plt.plot(class_test, "^--", label="Knn mnist test score")
        plt.xlabel("k")
        plt.ylabel("classification score")
        plt.title("knn Neighbors")
        plt.legend()
        plt.show()

    # labeling data check visualization
    def data_check(self):
        self.y_train_df['class'].value_counts().plot(kind='bar',
                                                     colormap='Paired')
        plt.xlabel('Class')
        plt.ylabel('Number of samples for each category')
        plt.title('Training set')
        plt.show()

        self.y_test_df['class'].value_counts().plot(kind='bar',
                                                    colormap='Paired')
        plt.xlabel('Class')
        plt.ylabel('Number of samples for each category')
        plt.title('Testing set')
        plt.show()

    def total_module(self):
        # 데이터 확인
        mnist_visualization(5, 5)
        self.data_check()

        # 학습 시작  classifier 거리값 설정해줄것
        self.acc_grape_visualization()
        self.classifier(5)  # 거리값 설정 해줄것