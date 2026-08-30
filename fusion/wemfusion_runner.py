import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import settings


# =========================================================
# 1. 加入 MambaDFuse 源码路径
# =========================================================

if str(settings.MAMBADFUSE_ROOT) not in sys.path:
    sys.path.insert(0, str(settings.MAMBADFUSE_ROOT))

try:
    from models.network import MambaDFuse
except Exception as e:
    raise ImportError(
        "无法导入 MambaDFuse。请确认：\n"
        "1. third_party/MambaDFuse 是否存在；\n"
        "2. third_party/MambaDFuse/models/network.py 是否存在；\n"
        "3. 当前环境是否安装 mamba_ssm、causal_conv1d、timm、einops。\n"
        f"原始错误：{e}"
    )


# =========================================================
# 2. 基础工具函数
# =========================================================

def get_device():
    if getattr(settings, "DEVICE", "cuda") == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def read_gray_image(path):
    """
    读取图像为灰度 float32，范围 [0, 1]。
    光学图像和 SAR 图像都转成 1 通道输入。
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def read_rgb_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def np_to_tensor(gray):
    """
    H,W -> 1,1,H,W
    """
    return torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)


def tensor_to_uint8(tensor):
    """
    1,1,H,W or 1,H,W -> H,W uint8
    """
    if tensor.dim() == 4:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:
        tensor = tensor[0]

    arr = tensor.detach().float().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def pad_to_window(img_a, img_b, window_size=8):
    """
    按官方 test_MambaDFuse.py 的方式，把输入 pad 到 window_size 的倍数。
    """
    _, _, h_old, w_old = img_a.size()

    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old

    if h_pad > 0:
        img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
        img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]

    if w_pad > 0:
        img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
        img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]

    return img_a, img_b, h_old, w_old


def save_gray(path, arr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_color_fused(optical_rgb, fused_gray, save_path):
    """
    用光学图像的色彩 + 融合图的亮度，生成彩色融合图。
    这样视觉上更像光学-SAR融合结果，而不是纯灰度图。
    """
    optical_rgb = cv2.resize(
        optical_rgb,
        (fused_gray.shape[1], fused_gray.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    ycrcb = cv2.cvtColor(optical_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = fused_gray
    fused_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    Image.fromarray(fused_rgb).save(save_path)


def edge_map(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    if mag.max() > 0:
        mag = mag / mag.max() * 255.0

    return mag.astype(np.uint8)


def save_diff(opt_gray, sar_gray, save_path):
    sar_resized = cv2.resize(
        sar_gray,
        (opt_gray.shape[1], opt_gray.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    diff = cv2.absdiff(opt_gray, sar_resized)
    Image.fromarray(diff).save(save_path)


# =========================================================
# 3. 简单指标计算
# =========================================================

def entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def spatial_frequency(img):
    img = img.astype(np.float32)
    rf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    cf = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def average_gradient(img):
    img = img.astype(np.float32)
    gx = img[:, 1:] - img[:, :-1]
    gy = img[1:, :] - img[:-1, :]
    gx = gx[:-1, :]
    gy = gy[:, :-1]
    return float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2.0)))


def mutual_information(a, b):
    a = a.flatten()
    b = b.flatten()

    hist_2d, _, _ = np.histogram2d(a, b, bins=256)
    pxy = hist_2d / (hist_2d.sum() + 1e-12)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    px_py = px[:, None] * py[None, :]
    nz = pxy > 0

    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / (px_py[nz] + 1e-12))))


def corrcoef(a, b):
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)

    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)

    if mse < 1e-12:
        return 99.0

    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def simple_ssim(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (7, 7), 1.5)
    mu_b = cv2.GaussianBlur(b, (7, 7), 1.5)

    sigma_a = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_a * mu_b

    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2) + 1e-12
    )

    return float(np.mean(ssim_map))


def qabf_approx(a, b, f):
    """
    简化版边缘保持指标，用 Sobel 边缘相关性近似。
    """
    ea = edge_map(a).astype(np.float32)
    eb = edge_map(b).astype(np.float32)
    ef = edge_map(f).astype(np.float32)

    return float((max(corrcoef(ea, ef), 0) + max(corrcoef(eb, ef), 0)) / 2.0)


def compute_metrics(opt_gray, sar_gray, fused_gray):
    sar_resized = cv2.resize(
        sar_gray,
        (opt_gray.shape[1], opt_gray.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    mi = mutual_information(opt_gray, fused_gray) + mutual_information(sar_resized, fused_gray)
    cc = (corrcoef(opt_gray, fused_gray) + corrcoef(sar_resized, fused_gray)) / 2.0
    ssim = (simple_ssim(opt_gray, fused_gray) + simple_ssim(sar_resized, fused_gray)) / 2.0
    qabf = qabf_approx(opt_gray, sar_resized, fused_gray)

    en = entropy(fused_gray)
    sd = float(np.std(fused_gray))
    sf = spatial_frequency(fused_gray)
    ag = average_gradient(fused_gray)
    psnr_value = (psnr(opt_gray, fused_gray) + psnr(sar_resized, fused_gray)) / 2.0

    score = (
        min(en / 8.0, 1.0) * 20
        + min(sf / 30.0, 1.0) * 20
        + min(ag / 15.0, 1.0) * 20
        + min(mi / 6.0, 1.0) * 20
        + min(qabf, 1.0) * 20
    )

    return {
        "EN 信息熵": en,
        "SD 标准差": sd,
        "SF 空间频率": sf,
        "AG 平均梯度": ag,
        "MI 互信息": mi,
        "CC 相关系数": cc,
        "PSNR 峰值信噪比": psnr_value,
        "SSIM 结构相似性": ssim,
        "Qabf 边缘保持": qabf,
        "综合质量评分": float(score),
    }


# =========================================================
# 4. 模型加载
# =========================================================

_MODEL_CACHE = None


def build_model():
    model = MambaDFuse(
        upscale=settings.MAMBADFUSE_SCALE,
        in_chans=settings.MAMBADFUSE_IN_CHANNEL,
        img_size=settings.MAMBADFUSE_IMG_SIZE,
        window_size=settings.MAMBADFUSE_WINDOW_SIZE,
        img_range=settings.MAMBADFUSE_IMG_RANGE,
        depths=[6, 6, 6, 6],
        embed_dim=settings.MAMBADFUSE_EMBED_DIM,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler=None,
        resi_connection="1conv",
    )

    return model


def find_weight_path():
    """
    官方测试脚本中实际加载的是 iter_number + "_E.pth"。
    这里优先 E，找不到再尝试 G。
    """
    e_path = Path(settings.MAMBADFUSE_E_PATH)
    g_path = Path(settings.MAMBADFUSE_G_PATH)

    if e_path.exists():
        return e_path

    if g_path.exists():
        return g_path

    raise FileNotFoundError(
        "没有找到 MambaDFuse 权重文件。\n"
        f"请检查：\n{e_path}\n或：\n{g_path}"
    )


def load_model():
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    device = get_device()
    model = build_model()

    ckpt_path = find_weight_path()

    print(f"Loading MambaDFuse weight from: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt, dict) and "params" in ckpt:
        state_dict = ckpt["params"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    _MODEL_CACHE = model

    return model


# =========================================================
# 5. 推理入口：保持 run_wemfusion 名字不变
# =========================================================

def infer_whole(model, opt_tensor, sar_tensor):
    device = get_device()

    opt_tensor = opt_tensor.to(device)
    sar_tensor = sar_tensor.to(device)

    window_size = settings.MAMBADFUSE_WINDOW_SIZE

    opt_pad, sar_pad, h_old, w_old = pad_to_window(opt_tensor, sar_tensor, window_size)

    with torch.no_grad():
        output = model(opt_pad, sar_pad)

    output = output[..., :h_old, :w_old]

    return output


def infer_tile(model, opt_tensor, sar_tensor, tile=256, overlap=32):
    """
    分块推理，适合大图，避免显存过高。
    """
    device = get_device()
    window_size = settings.MAMBADFUSE_WINDOW_SIZE

    opt_tensor = opt_tensor.to(device)
    sar_tensor = sar_tensor.to(device)

    b, c, h, w = opt_tensor.shape

    tile = min(tile, h, w)

    if tile % window_size != 0:
        tile = (tile // window_size) * window_size

    if tile < window_size:
        return infer_whole(model, opt_tensor, sar_tensor)

    stride = tile - overlap

    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

    output = torch.zeros_like(opt_tensor).to(device)
    weight = torch.zeros_like(opt_tensor).to(device)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            opt_patch = opt_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            sar_patch = sar_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]

            with torch.no_grad():
                out_patch = model(opt_patch, sar_patch)

            mask = torch.ones_like(out_patch)

            output[..., h_idx:h_idx + tile, w_idx:w_idx + tile] += out_patch
            weight[..., h_idx:h_idx + tile, w_idx:w_idx + tile] += mask

    output = output / (weight + 1e-8)

    return output


def run_wemfusion(
    optical_path,
    sar_path,
    output_dir=None,
    use_tile=False,
    tile=256,
    overlap=32,
):
    """
    兼容原系统的融合入口。
    虽然函数名叫 run_wemfusion，但内部实际调用的是 MambaDFuse。
    """
    start_time = time.time()

    optical_path = str(optical_path)
    sar_path = str(sar_path)

    output_dir = Path(output_dir or settings.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    opt_gray_float = read_gray_image(optical_path)
    sar_gray_float = read_gray_image(sar_path)

    # 尺寸对齐：以光学图像尺寸为准
    h, w = opt_gray_float.shape

    if sar_gray_float.shape != opt_gray_float.shape:
        sar_gray_float = cv2.resize(
            sar_gray_float,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

    opt_tensor = np_to_tensor(opt_gray_float)
    sar_tensor = np_to_tensor(sar_gray_float)

    model = load_model()

    if use_tile:
        fused_tensor = infer_tile(model, opt_tensor, sar_tensor, tile=tile, overlap=overlap)
    else:
        fused_tensor = infer_whole(model, opt_tensor, sar_tensor)

    fused_gray = tensor_to_uint8(fused_tensor)

    opt_gray_uint8 = (np.clip(opt_gray_float, 0, 1) * 255).astype(np.uint8)
    sar_gray_uint8 = (np.clip(sar_gray_float, 0, 1) * 255).astype(np.uint8)
    optical_rgb = read_rgb_image(optical_path)

    fused_gray_path = output_dir / f"mambadfuse_gray_{timestamp}.png"
    fused_color_path = output_dir / f"mambadfuse_color_{timestamp}.png"
    edge_optical_path = output_dir / f"edge_optical_{timestamp}.png"
    edge_sar_path = output_dir / f"edge_sar_{timestamp}.png"
    edge_fused_path = output_dir / f"edge_fused_{timestamp}.png"
    diff_path = output_dir / f"diff_opt_sar_{timestamp}.png"

    save_gray(fused_gray_path, fused_gray)
    save_color_fused(optical_rgb, fused_gray, fused_color_path)

    save_gray(edge_optical_path, edge_map(opt_gray_uint8))
    save_gray(edge_sar_path, edge_map(sar_gray_uint8))
    save_gray(edge_fused_path, edge_map(fused_gray))
    save_diff(opt_gray_uint8, sar_gray_uint8, diff_path)

    metrics = compute_metrics(opt_gray_uint8, sar_gray_uint8, fused_gray)

    elapsed = time.time() - start_time

    result = {
        "optical_path": optical_path,
        "sar_path": sar_path,
        "fused_gray_path": str(fused_gray_path),
        "fused_color_path": str(fused_color_path),
        "edge_optical_path": str(edge_optical_path),
        "edge_sar_path": str(edge_sar_path),
        "edge_fused_path": str(edge_fused_path),
        "diff_path": str(diff_path),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "device": str(get_device()),
        "model_name": "MambaDFuse / Official",
    }

    print("融合完成！")
    print("模型：MambaDFuse")
    print("灰度融合图：", fused_gray_path)
    print("彩色融合图：", fused_color_path)
    print("推理设备：", result["device"])
    print("推理耗时：", elapsed)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--optical", type=str, required=True)
    parser.add_argument("--sar", type=str, required=True)
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)

    args = parser.parse_args()

    result = run_wemfusion(
        optical_path=args.optical,
        sar_path=args.sar,
        use_tile=args.tile,
        tile=args.tile_size,
        overlap=args.overlap,
    )

    print(result)