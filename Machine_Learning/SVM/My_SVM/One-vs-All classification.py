from my_svm import SVM
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np

# 准备数据
digits = load_digits()
X = digits.data
y = digits.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化多个SVM分类器，每个分类器对应一个类别
classifiers = {}
for class_label in range(10):  # 假设有10个类别
    # 将当前类别标记为1，其他类别标记为-1
    y_binary = (y_train == class_label).astype(int)
    y_binary = [-1 if value == 0 else 1 for value in y_binary]
    # 检查训练文件是否存在
    classifiers_folder = './classifiers'
    if not os.path.exists(classifiers_folder):
        os.makedirs(classifiers_folder)
    train_file_path = f'./classifiers/classifier_{class_label}.pkl'
    if os.path.exists(train_file_path):
        # 如果文件存在，则加载已训练的分类器
        with open(train_file_path, 'rb') as f:
            svm_classifier = pickle.load(f)
    else:
        # 如果文件不存在，则重新训练分类器
        svm_classifier = SVM('rbf', 1, 1)
        svm_classifier.fit(X_train, y_binary)  # 训练分类器
        # 保存训练好的分类器
        with open(train_file_path, 'wb') as f:
            pickle.dump(svm_classifier, f)

    y_pre = svm_classifier.predict(X_test)
    y_class_b = (y_test == class_label).astype(int)
    y_class_b = [-1 if value == 0 else 1 for value in y_class_b]
    right_num = np.sum(y_pre == y_class_b) / len(y_test)
    print(f"数字{class_label}的正确率为{right_num}")
    classifiers[class_label] = svm_classifier

# 预测测试集
predictions = []
scores = {class_label: classifier.predict(X_test) for class_label, classifier in classifiers.items()}
for i in range(len(X_test)):
    found = False
    for k, value_list in enumerate(scores.values()):
        if value_list[i] == 1:
            if k == y_test[i]:
                predictions.append(k)
                found = True
                break
    if not found:
        predictions.append(None)

# 计算准确率
accuracy = sum(predictions == y_test) / len(y_test)
print(f'在测试集上的正确率为: {accuracy:.2f}')
