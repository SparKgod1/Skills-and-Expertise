import numpy as np
import matplotlib.pyplot as plt
from my_svm import SVM
from sklearn.datasets import make_classification
from sklearn.svm import SVC


def plot_decision_boundary(clf, X, y):
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')


if __name__ == '__main__':
    x_, y_ = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2,
                                 random_state=63, class_sep=1)
    y_ = [-1 if i == 0 else i for i in y_]
    my_classifier = SVM('rbf', 0.8, gamma=1.2)
    my_classifier.fit(x_, y_)
    y_pre = my_classifier.predict(x_)
    right_num = np.sum(y_pre == y_)
    print('手写正确率：%.2f%%' % (right_num / len(y_pre) * 100))

    sk_classifier = SVC(kernel='rbf')
    sk_classifier.fit(x_, y_)
    y_pre_ = sk_classifier.predict(x_)
    right_num = np.sum(y_pre_ == y_)
    print('调包正确率：%.2f%%' % (right_num / len(y_pre_) * 100))

    # 创建一个包含两个子图的画布
    plt.figure(figsize=(12, 6))

    # 绘制第一个子图：SVC 分类结果
    plt.subplot(1, 2, 1)
    plot_decision_boundary(sk_classifier, x_, y_)
    plt.title('SVC Classification with RBF Kernel')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # 绘制第二个子图：原始数据分布
    plt.subplot(1, 2, 2)
    plot_decision_boundary(my_classifier, x_, y_)
    plt.title('My SVM Classification with RBF Kernel')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # 显示图像
    plt.tight_layout()
    plt.show()