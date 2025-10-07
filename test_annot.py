# %%
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import os
import cv2
import insightface
# %%
@dataclass
class FaceRecord:
    label: str
    embedding: np.ndarray  # L2-normalized embedding
    bbox: Optional[Tuple[int, int, int, int]] = None
    score: Optional[float] = None

def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """PIL.Image -> OpenCV BGR ndarray"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Assumes both a and b are already L2-normalized."""
    return float(np.dot(a, b))

class FaceLabeler:
    """
    轻量人脸处理器：
    - init 不接收参考图；仅初始化人脸模型。
    - detect(image): 在输入图中检测所有人脸，提取特征并缓存（不编号）。
    - label(): 对累计的人脸特征做去重聚类，生成唯一编号（从 0 开始），返回 {编号: 代表图PIL}。
    - match(image): 输入图中选取最大的人脸，与已编号的人脸库比对，返回最相似的编号。
    """
    def __init__(
        self,
        model_name: str = "buffalo_l",
        provider: Optional[List[str]] = None,
        match_threshold: float = 0.38,  # 聚类阈值（仅用于 label 聚类）
        text_font_scale: float = 0.7,
        text_thickness: int = 2,
        root: Optional[str] = None,
    ):
        """
        参数说明：
        - model_name: InsightFace 模型包名（如 'buffalo_l'、'antelopev2'）。
        - provider: onnxruntime providers 列表，None 则由 InsightFace 默认选择。
        - root: InsightFace 模型缓存目录（离线环境可指向本地已下载模型）。
        """
        # 尝试构建 InsightFace 应用，带降级与友好报错
        self.app = None
        last_err = None
        candidates = []
        # 优先使用用户传入的 model_name
        candidates.append((model_name, True))
        # 备选回退：buffalo_l（若不同）
        if model_name != "buffalo_l":
            candidates.append(("buffalo_l", True))
        # 最后回退：不指定 name，仅依赖 allowed_modules
        candidates.append((None, False))

        # 若未显式提供 root，尝试从环境变量读取
        if root is None:
            env_root = os.environ.get("INSIGHTFACE_HOME") or os.environ.get("INSIGHTFACE_ROOT")
            if env_root:
                root = env_root

        def build_app(name_opt, use_name_flag, providers_opt):
            kwargs = {"providers": providers_opt, "allowed_modules": ["detection", "recognition"]}
            if root is not None:
                kwargs["root"] = root
            if use_name_flag:
                kwargs["name"] = name_opt
            return insightface.app.FaceAnalysis(**kwargs)

        built = False
        for name, use_name in candidates:
            prov_candidates = []
            if provider is not None:
                prov_candidates.append(provider)
            else:
                prov_candidates.append(None)
                prov_candidates.append(["CPUExecutionProvider"])  # 显式 CPU 回退
            for prov in prov_candidates:
                try:
                    self.app = build_app(name, use_name, prov)
                    self.app.prepare(ctx_id=0, det_size=(640, 640))
                    built = True
                    break
                except Exception as e:
                    last_err = e
                    self.app = None
            if built:
                break
        if self.app is None:
            raise RuntimeError(
                "初始化 InsightFace 失败：未能加载包含检测/识别的模型包。"
                "请确认已联网下载或提供本地模型目录 root；"
                "或尝试使用默认 model_name='buffalo_l'。\n原始错误:" + repr(last_err)
            )

        # 仅用于 label() 聚类的阈值；match() 的阈值从参数传入
        self.cluster_threshold = float(match_threshold)
        self.text_font_scale = text_font_scale
        self.text_thickness = text_thickness

        # 最近一次 detect 的原图与结果（用于 get_faces 裁剪）
        self._last_image: Optional[Image.Image] = None
        self._last_results: Optional[List[Dict]] = None
        
        # 累计的人脸库（未去重）：每次 detect 追加
        # 元素：{"embedding": np.ndarray, "bbox": [x1,y1,x2,y2], "score": float, "crop": PIL.Image}
        self._all_faces: List[Dict] = []

        # 去重后的身份库：label(str) -> {"centroid": np.ndarray, "crops": List[PIL.Image], "best_crop": PIL.Image}
        self._identities: Dict[str, Dict] = {}

    def _face_area(self, bbox: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _cluster_faces(self):
        """基于余弦相似度的简单在线聚类，更新 self._identities。"""
        clusters: List[Dict] = []  # 顺序即标签顺序
        for rec in self._all_faces:
            emb = rec["embedding"]
            # 找与现有簇的最高相似度
            best_idx, best_sim = -1, -1.0
            for idx, c in enumerate(clusters):
                sim = float(np.dot(emb, c["centroid"]))
                if sim > best_sim:
                    best_sim, best_idx = sim, idx
            if best_sim >= self.cluster_threshold and best_idx >= 0:
                c = clusters[best_idx]
                # 更新簇
                c["embeds"].append(emb)
                # 更新质心（均值后归一化）
                centroid = np.mean(np.stack(c["embeds"], axis=0), axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                c["centroid"] = centroid
                # 保留面积最大的代表图
                if self._face_area(rec["bbox"]) > self._face_area(c["best_bbox"]):
                    c["best_bbox"] = tuple(rec["bbox"])  # type: ignore
                    c["best_crop"] = rec["crop"]
                c["crops"].append(rec["crop"])
            else:
                # 新簇
                clusters.append({
                    "embeds": [emb],
                    "centroid": emb,
                    "crops": [rec["crop"]],
                    "best_crop": rec["crop"],
                    "best_bbox": tuple(rec["bbox"])  # type: ignore
                })
        # 写回 identities，编号从 0 开始的字符串
        identities: Dict[str, Dict] = {}
        for i, c in enumerate(clusters):
            lab = str(i)
            identities[lab] = {
                "centroid": c["centroid"],
                "crops": c["crops"],
                "best_crop": c["best_crop"],
            }
        self._identities = identities

    def detect(self, image: Image.Image) -> Dict:
        """
        输入：PIL.Image
        仅检测并缓存所有人脸（不编号）。
        返回：{"faces": [{"bbox":..., "score":...}], "annotated": PIL.Image}
        """
        # 记录原图，便于后续裁剪人脸
        self._last_image = image.copy()
        bgr = _pil_to_bgr(image)
        faces = self.app.get(bgr)

        results = []
        canvas = bgr.copy()

        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox.astype(int))
            emb = np.array(f.normed_embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)

            # 缓存到全局库
            w, h = self._last_image.size
            xa = max(0, min(x1, w))
            ya = max(0, min(y1, h))
            xb = max(0, min(x2, w))
            yb = max(0, min(y2, h))
            if xb <= xa or yb <= ya:
                continue
            crop = self._last_image.crop((xa, ya, xb, yb))
            rec = {
                "embedding": emb,
                "bbox": [xa, ya, xb, yb],
                "score": float(getattr(f, "det_score", getattr(f, "score", 0.0))),
                "crop": crop,
            }
            self._all_faces.append(rec)

            # 画框（不写文本）
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

            results.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(getattr(f, "det_score", getattr(f, "score", 0.0))),
            })

        annotated = _bgr_to_pil(canvas)
        # 保存结果，供 get_faces 使用
        self._last_results = results
        return {"faces": results, "annotated": annotated}

    def get_faces(self) -> Dict[str, Image.Image]:
        """
        返回最近一次 detect 的人脸裁剪图：{idx(str): PIL.Image}
        - idx 为本次检测的人脸顺序编号（从 0 开始）。
        - 在未调用 detect 前调用会抛出 ValueError。
        """
        if self._last_image is None or self._last_results is None:
            raise ValueError("尚未调用 detect，无法获取人脸裁剪图。")

        img = self._last_image
        w, h = img.size

        crops: Dict[str, Image.Image] = {}
        for i, item in enumerate(self._last_results):
            x1, y1, x2, y2 = item["bbox"]
            x1 = max(0, min(int(x1), w))
            y1 = max(0, min(int(y1), h))
            x2 = max(0, min(int(x2), w))
            y2 = max(0, min(int(y2), h))
            if x2 <= x1 or y2 <= y1:
                continue
            crops[str(i)] = img.crop((x1, y1, x2, y2))

        return crops

    def label(self) -> Dict[str, Image.Image]:
        """
        对已收集的人脸进行去重聚类，生成唯一编号（从 0 开始），
        返回 {label(str): 代表图 PIL.Image}。
        """
        if not self._all_faces:
            return {}
        self._cluster_faces()
        return {lab: info["best_crop"] for lab, info in self._identities.items()}

    def match(self, image: Image.Image, threshold: float) -> Optional[str]:
        """
        识别输入图中最大的人脸，与已编号的人脸中最相似的编号。
        - threshold: 相似性阈值，若所有相似度均低于该阈值，返回 None。
        需先调用 label() 建立身份库；若库为空或未检测到人脸则抛出 ValueError。
        """
        if not self._identities:
            raise ValueError("身份库为空。请先调用 detect 收集样本并执行 label()。")

        bgr = _pil_to_bgr(image)
        faces = self.app.get(bgr)
        if not faces:
            raise ValueError("输入图中未检测到人脸。")

        # 选取最大的人脸
        def area_of(face) -> int:
            x1, y1, x2, y2 = map(int, face.bbox.astype(int))
            return max(0, x2 - x1) * max(0, y2 - y1)
        faces.sort(key=area_of, reverse=True)
        f = faces[0]
        emb = np.array(f.normed_embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        # 与每个身份质心计算相似度
        best_label, best_sim = None, -1.0
        for lab, info in self._identities.items():
            sim = float(np.dot(emb, info["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_label = lab
        if best_label is None or best_sim < float(threshold):
            return None
        return best_label

#%% 简单用法示例（按需注释/调整）
labeler = FaceLabeler(match_threshold=0.5, model_name='antelopev2')  # 使用默认 'buffalo_l'
imgs = [Image.open(f"test_output/scene_{i}.png") for i in range(4)]
for img in imgs:
    labeler.detect(img)
unique = labeler.label()  # {"0": PIL.Image, "1": PIL.Image, ...}
query = Image.open("test_output/scene_0.png")
# %%
best = labeler.match(imgs[1], threshold=10)
unique.get(best)
#%%
