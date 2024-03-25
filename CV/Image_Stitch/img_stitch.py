import cv2
import numpy as np


class Match:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance


def knn_match_manual(descriptors1, descriptors2, k=2):
    matches = []

    # 遍历每个 descriptors1 中的特征描述符
    for i in range(len(descriptors1)):
        distances = np.linalg.norm(descriptors1[i] - descriptors2, axis=1)
        indices = np.argsort(distances)

        # 选择最近的 k 个邻居
        knn_matches = [Match(queryIdx=i, trainIdx=indices[j], distance=distances[indices[j]]) for j in range(k)]
        matches.append(knn_matches)

    return matches


def find_homography_manual(src_pts, dst_pts):
    A = []

    # 构建用于单应性矩阵估计的矩阵 A
    for i in range(len(src_pts)):
        x, y = src_pts[i][0][0], src_pts[i][0][1]
        u, v = dst_pts[i][0][0], dst_pts[i][0][1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    h = V[-1, :].reshape(3, 3)
    return h


def find_homography_ransac(key_points1, key_points2, matches, num_iterations=100, threshold=5.0):
    best_homography = None
    max_inliers = 0

    for _ in range(num_iterations):
        # 随机选择4个匹配点对
        random_indices = np.random.choice(len(matches), 20, replace=False)
        src_pts = np.float32([key_points1[matches[i].queryIdx].pt for i in random_indices]).reshape(-1, 1, 2)
        dst_pts = np.float32([key_points2[matches[i].trainIdx].pt for i in random_indices]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        homography = find_homography_manual(src_pts, dst_pts)

        # 测试其他匹配点
        inliers = 0
        for i, match in enumerate(matches):
            if i not in random_indices:
                src_pt = np.float32(key_points1[match.queryIdx].pt).reshape(-1, 1, 2)
                dst_pt = np.float32(key_points2[match.trainIdx].pt).reshape(-1, 1, 2)

                # 计算投影误差
                projected_pt = cv2.perspectiveTransform(src_pt, homography)
                error = np.linalg.norm(dst_pt - projected_pt)

                # 判断是否为内点
                if error < threshold:
                    inliers += 1

        # 更新最优模型
        if inliers > max_inliers:
            max_inliers = inliers
            best_homography = homography

    return best_homography


def image_stitching(image1, image2):
    # SIFT算法提取特征点
    sift = cv2.SIFT_create()
    key_points1, descriptors1 = sift.detectAndCompute(image1, None)
    key_points2, descriptors2 = sift.detectAndCompute(image2, None)

    # 计算特征点匹配
    matches = knn_match_manual(descriptors1, descriptors2, k=2)

    # 消除误匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 估计最佳单应性矩阵
    best_homography = find_homography_ransac(key_points1, key_points2, good_matches)

    # 根据变换矩阵拼接图像
    result_image = cv2.warpPerspective(image1, best_homography, (image1.shape[1] + image2.shape[1], image1.shape[0]))
    result_image[0:image2.shape[0], 0:image2.shape[1]] = image2

    return result_image


