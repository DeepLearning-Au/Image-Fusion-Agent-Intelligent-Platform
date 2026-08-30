import math
import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim


def _check_gray(img):
    assert isinstance(img, np.ndarray), "输入必须是 numpy.ndarray"
    assert len(img.shape) == 2, "输入必须是灰度图 H×W"


def _check_pair(fused, img_a, img_b):
    _check_gray(fused)
    _check_gray(img_a)
    _check_gray(img_b)
    assert fused.shape == img_a.shape == img_b.shape, "fused、img_a、img_b 尺寸必须一致"


def to_uint8(img):
    img = np.asarray(img)

    # 如果输入是 0~1 浮点图，则转为 0~255
    if img.max() <= 1.5 and img.min() >= 0:
        img = img * 255.0

    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def EN(img):
    """
    Entropy, 信息熵。
    8 bit 灰度图理论最大值约为 8。
    """
    _check_gray(img)
    img = to_uint8(img)

    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def SD(img):
    """
    Standard Deviation, 标准差。
    反映灰度分布离散程度和图像对比度。
    """
    _check_gray(img)
    img = to_uint8(img).astype(np.float64)

    return float(np.std(img))


def SF(img):
    """
    Spatial Frequency, 空间频率。
    SF = sqrt(RF^2 + CF^2)
    """
    _check_gray(img)
    img = to_uint8(img).astype(np.float64)

    rf = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    cf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))

    return float(np.sqrt(rf ** 2 + cf ** 2))


def AG(img):
    """
    Average Gradient, 平均梯度。
    按 VIFB 常见口径，使用前向差分：
    AG = mean(sqrt((dx^2 + dy^2) / 2))
    """
    _check_gray(img)
    img = to_uint8(img).astype(np.float64)

    if img.shape[0] < 2 or img.shape[1] < 2:
        return 0.0

    dx = img[:-1, :-1] - img[1:, :-1]
    dy = img[:-1, :-1] - img[:-1, 1:]

    return float(np.mean(np.sqrt((dx ** 2 + dy ** 2) / 2.0)))


def _mutual_info(x, y):
    """
    Mutual Information between two 8-bit gray images.
    使用固定 256×256 联合直方图，避免 np.histogram2d 自动取范围造成不稳定。
    """
    _check_gray(x)
    _check_gray(y)
    assert x.shape == y.shape, "两幅图尺寸必须一致"

    x = to_uint8(x).astype(np.int64).ravel()
    y = to_uint8(y).astype(np.int64).ravel()

    joint = np.bincount(x * 256 + y, minlength=256 * 256).reshape(256, 256).astype(np.float64)
    pxy = joint / (joint.sum() + 1e-12)

    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    px_py = px @ py

    valid = pxy > 0

    return float(np.sum(pxy[valid] * np.log2(pxy[valid] / (px_py[valid] + 1e-12))))


def MI(fused, img_a, img_b):
    """
    Fusion MI:
    MI = MI(A,F) + MI(B,F)
    """
    _check_pair(fused, img_a, img_b)

    return _mutual_info(img_a, fused) + _mutual_info(img_b, fused)


def CC(fused, img_a, img_b):
    """
    Correlation Coefficient.
    取融合图与两个源图相关系数的平均值。
    """
    _check_pair(fused, img_a, img_b)

    fused = to_uint8(fused).astype(np.float64)
    img_a = to_uint8(img_a).astype(np.float64)
    img_b = to_uint8(img_b).astype(np.float64)

    def corr(x, y):
        x = x - np.mean(x)
        y = y - np.mean(y)
        denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)) + 1e-12
        return float(np.sum(x * y) / denom)

    return float((corr(fused, img_a) + corr(fused, img_b)) / 2.0)


def PSNR(fused, img_a, img_b):
    """
    按 VIFB 公式：
    MSE = (MSE(A,F) + MSE(B,F)) / 2
    PSNR = 10 * log10(255^2 / MSE)

    注意：不是先算两个 PSNR 再平均。
    """
    _check_pair(fused, img_a, img_b)

    fused = to_uint8(fused).astype(np.float64)
    img_a = to_uint8(img_a).astype(np.float64)
    img_b = to_uint8(img_b).astype(np.float64)

    mse_a = np.mean((img_a - fused) ** 2)
    mse_b = np.mean((img_b - fused) ** 2)
    mse = (mse_a + mse_b) / 2.0

    if mse <= 1e-12:
        return float("inf")

    return float(10.0 * np.log10((255.0 ** 2) / mse))


def SSIM(fused, img_a, img_b):
    """
    按 VIFB 口径：
    SSIM = SSIM(A,F) + SSIM(B,F)
    因此该值范围大致为 0~2。

    如果你希望显示为 0~1，则改成：
    return float((ssim_a + ssim_b) / 2.0)
    """
    _check_pair(fused, img_a, img_b)

    fused = to_uint8(fused)
    img_a = to_uint8(img_a)
    img_b = to_uint8(img_b)

    ssim_a = ssim(img_a, fused, data_range=255)
    ssim_b = ssim(img_b, fused, data_range=255)

    return float(ssim_a + ssim_b)


def _edge_array(img):
    """
    Qabf 中使用 Sobel 算子计算边缘强度和方向。
    """
    _check_gray(img)
    img = to_uint8(img).astype(np.float64)

    h1 = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float64)

    h3 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)

    gx = convolve2d(img, h3, mode="same", boundary="symm")
    gy = convolve2d(img, h1, mode="same", boundary="symm")

    g = np.sqrt(gx ** 2 + gy ** 2)

    a = np.zeros_like(img)
    a[gx == 0] = math.pi / 2
    idx = gx != 0
    a[idx] = np.arctan(gy[idx] / gx[idx])

    return g, a


def Qabf(fused, img_a, img_b):
    """
    Qabf / QABF 边缘信息保持指标。
    数值越大，说明源图边缘信息向融合图传递得越充分。
    """
    _check_pair(fused, img_a, img_b)

    gA, aA = _edge_array(img_a)
    gB, aB = _edge_array(img_b)
    gF, aF = _edge_array(fused)

    def get_q(a_src, g_src, a_f, g_f):
        Tg, kg, Dg = 0.9994, -15, 0.5
        Ta, ka, Da = 0.9879, -22, 0.8

        G = np.zeros_like(g_src)

        idx1 = g_src > g_f
        idx2 = g_src == g_f
        idx3 = g_src < g_f

        G[idx1] = g_f[idx1] / (g_src[idx1] + 1e-12)
        G[idx2] = 1.0
        G[idx3] = g_src[idx3] / (g_f[idx3] + 1e-12)

        A = 1.0 - np.abs(a_src - a_f) / (math.pi / 2)

        Qg = Tg / (1.0 + np.exp(kg * (G - Dg)))
        Qa = Ta / (1.0 + np.exp(ka * (A - Da)))

        return Qg * Qa

    QAF = get_q(aA, gA, aF, gF)
    QBF = get_q(aB, gB, aF, gF)

    numerator = np.sum(QAF * gA + QBF * gB)
    denominator = np.sum(gA + gB) + 1e-12

    return float(numerator / denominator)


def sobel_edge(img):
    """
    仅用于可视化边缘，不属于正式融合评价指标。
    """
    _check_gray(img)
    img = to_uint8(img).astype(np.float64)

    sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)

    sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float64)

    gx = convolve2d(img, sx, mode="same", boundary="symm")
    gy = convolve2d(img, sy, mode="same", boundary="symm")

    edge = np.sqrt(gx ** 2 + gy ** 2)
    edge = edge / (edge.max() + 1e-12) * 255.0

    return edge.astype(np.uint8)


def calculate_metrics(fused, img_a, img_b):
    _check_pair(fused, img_a, img_b)

    metrics = {
        "EN 信息熵": EN(fused),
        "SD 标准差": SD(fused),
        "SF 空间频率": SF(fused),
        "AG 平均梯度": AG(fused),
        "MI 互信息": MI(fused, img_a, img_b),
        "CC 相关系数": CC(fused, img_a, img_b),
        "PSNR 峰值信噪比": PSNR(fused, img_a, img_b),
        "SSIM 结构相似性": SSIM(fused, img_a, img_b),
        "Qabf 边缘保持": Qabf(fused, img_a, img_b),
    }

    # 注意：这个综合评分不是论文通用权威指标，只适合你的系统里做展示排序。
    score = (
        min(metrics["EN 信息熵"] / 8.0, 1.0) * 20
        + min(metrics["SF 空间频率"] / 25.0, 1.0) * 20
        + min(metrics["AG 平均梯度"] / 15.0, 1.0) * 15
        + min(metrics["MI 互信息"] / 8.0, 1.0) * 20
        + min(metrics["SSIM 结构相似性"] / 2.0, 1.0) * 15
        + min(metrics["Qabf 边缘保持"] / 1.0, 1.0) * 10
    )

    metrics["综合质量评分"] = float(score)

    return metrics