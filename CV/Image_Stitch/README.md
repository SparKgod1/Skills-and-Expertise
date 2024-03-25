# Image Stitching
这个项目实现了使用 SIFT 算法进行图像拼接的功能。图像拼接是将多张图像拼接成一张全景图像的过程，常用于全景摄影、地图制作等领域。

# 依赖项
- Python 3.x
- OpenCV 库
- NumPy 库
# 使用示例
```
import cv2
from img_stitch import image_stitching

image1 = cv2.imread("./data/000007.jpg", cv2.IMREAD_UNCHANGED)
image2 = cv2.imread("./data/000006.jpg", cv2.IMREAD_UNCHANGED)
image3 = cv2.imread("./data/000005.jpg", cv2.IMREAD_UNCHANGED)

result_image = image_stitching(image1, image2)
result_image1 = image_stitching(result_image, image3)

cv2.imshow("Stitched Image", result_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
# 效果展示
待拼接的三张图片：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B6E7BC47C-A2F6-4140-A1FA-00CE1BC3C619%7D.png)

(当然也可以是更多张，但是考虑到进行单应性变换时，第一张图片作为目标平面，图片数量过多可能导致变换过大，从而造成图片失真)

拼接结果：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B7C1A353F-8B0C-4e3a-A26B-1771D721594F%7D.png)

# 注意事项
- 确保图片从右到左拼接
- 确保图像文件路径正确。
- 在运行代码前，确保已安装 Python 3.x、OpenCV 和 NumPy。

# 原理介绍
## SIFT算法原理
尺度不变特征转换即SIFT (Scale-invariant feature transform)。它用来侦测与描述影像中的局部性特征，它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，此算法由 David Lowe在1999年所发表，2004年完善总结。应用范围包含物体辨识、机器人地图感知与导航、影像缝合、3D模型建立、手势辨识、影像追踪和动作比对等领域。

SIFT算法的实质是在不同的尺度空间上查找关键点(特征点)，并计算出关键点的方向。SIFT所查找到的关键点是一些十分突出，不会因光照，仿射变换和噪音等因素而变化的点，如**角点、边缘点、暗区的亮点及亮区的暗点**等。

Lowe将SIFT算法分解为如下四步：

1. 尺度空间极值检测：搜索所有尺度上的图像位置。通过高斯差分函数来识别潜在的对于尺度和旋转不变的关键点。

2. 关键点定位：在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。

3. 关键点方向确定：基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而保证了对于这些变换的不变性。

4. 关键点描述：在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度作为关键点的描述符，它允许比较大的局部形状的变形或光照变化。

SIFT算法通过这些步骤可以提取出具有不变性和稳定性的图像特征点，并计算特征描述符对该特征点进行尺度和旋转不变的描述。

## 单应性矩阵原理
射影变换也叫“单应”--Homography，“Homo”前缀就是same的意思，表示“同”，homography就是用同一个源产生的graphy，中文译过来大概就是“单应”。

《从零开始学习「张氏相机标定法」（一）成像几何模型》中我们已经得到了像素坐标系和世界坐标系下的坐标映射关系：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7BC2C3AABC-4BDE-4920-B7F8-F431C80F8964%7D.png)

其中，u、v表示像素坐标系中的坐标，s表示尺度因子，fx、fy、u0、v0、γ（由于制造误差产生的两个坐标轴偏斜参数，通常很小）表示5个相机内参，R，t表示相机外参，Xw、Yw、Zw（假设标定棋盘位于世界坐标系中Zw=0的平面）表示世界坐标系中的坐标。

我们在这里引入一个新的概念：单应性（Homography）变换。可以简单的理解为它用来描述物体在世界坐标系和像素坐标系之间的位置映射关系。对应的变换矩阵称为单应性矩阵。在上述式子中，单应性矩阵定义为：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B2D7391EF-1C2C-4dd0-8EF7-611A21F71FD6%7D.png)

其中，M是内参矩阵：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B9F29B358-8788-43e7-87AC-42049024AE4B%7D.png)

利用单应性矩阵我们可以对一张图片进行视角转换，那我们把不同角度拍摄的图像都转换到同样的视角下，就可以实现图像拼接了。如下图所示，通过单应矩阵H可以将image1和image2都变换到同一个平面。

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B33AEAFC6-21BC-4871-AD32-3BF3427AF85B%7D.png)

那么如何估计单应矩阵？

首先，我们假设两张图像中的对应点对齐次坐标为(x',y',1)和(x,y,1)，单应矩阵H定义为：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/294c98c9eb8427c90f9db1b8f451e4da.png)

则有：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/10e1549705673ce00f50bcc2afed1800.png)

矩阵展开后有3个等式，将第3个等式代入前两个等式中可得：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/5db93b8a634dddd6bfe37f1badbdb47d.png)

也就是说，一个点对对应两个等式。因为这里使用的是齐次坐标系，也就是说可以进行任意尺度的缩放，因此单应矩阵H只有8个自由度。8自由度下H计算过程有两种方法。

第一种方法：直接设置h33=1，那么上述等式变为：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/eab31fd7ddfc3f418e539b38bb636dc8.png)

第二种方法：将H添加约束条件，将H矩阵模变为1，如下：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/5328b51139a920815b7bc18af501c582.png)

以第2种方法（用第1种也类似）为例继续推导，我们将如下等式（包含||H||=1约束）:

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B86624292-537C-4b74-BE26-9A5971136C3F%7D.png)

乘以分母展开，得到：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B433024AF-604D-4ce6-BA9D-263D3338185E%7D.png)

整理，得到：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B279846D2-6888-43ad-BA23-3EA3D17763AC%7D.png)

假如我们得到了两幅图片中对应的N个点对（特征点匹配对），那么可以得到如下线性方程组：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7BD8BF6682-BBA5-4636-984B-E7523C0EDC3D%7D.png)
通过奇异值分解（SVD）对矩阵 A 进行分解，可以得到：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B326557A4-B012-41c9-817E-A50E265F760A%7D.png)
将其代入约束方程中：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B3E5D7B6F-4B77-46c6-A67E-3FD85C20876D%7D.png)

由于矩阵U 和 V 是正交矩阵，它们的转置是其逆矩阵。因此，上述方程可以简化为：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7B3325851C-FD98-4c95-881F-79AD90254EE6%7D.png)

由于 Σ 是对角矩阵，它的逆矩阵也是对角矩阵，只需将对角线上的非零元素取倒数即可。因此，我们可以将上述方程进一步简化为：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%7BC68BF151-E202-4470-85AB-67A6EC8778A5%7D.png)

这里的 V 的逆的最后一行（或者说最小奇异值对应的列）实际上对应着约束方程组的最小二乘解，即满足约束条件的 H 的近似解。

