import os
import base64
import random
import string
from pathlib import Path

os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
os.environ.setdefault('KMP_WARNINGS', '0')

import logging
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify

try:
    import torch
    from ultralytics import YOLO
except Exception:
    torch = None
    YOLO = None


#MKLDNN cpu fix
if torch is not None:
    try:
        torch.backends.mkldnn.enabled = False
    except Exception:
        pass


app = Flask(__name__)

# Логгер для вывода в консоль
logger = logging.getLogger('captcha_solver')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO').upper())

CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
CONF_THRESHOLD = float(os.getenv('YOLO_CONF', '0.2'))
IOU_NMS = float(os.getenv('YOLO_IOU', '0.2'))

CAPTCHA_DIR = Path(__file__).parent / 'captchas'
CAPTCHA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = Path(__file__).parent / 'best.pt'

# Низкая уверенность: сохранять сюда
LOW_CONF_THRESHOLD = float(os.getenv('LOW_CONF_THRESHOLD', '0.75'))
LOW_CONF_DIR = CAPTCHA_DIR / 'something'
LOW_CONF_DIR.mkdir(parents=True, exist_ok=True)

def get_bool_env(key: str, default: str = '1') -> bool:
    val = os.getenv(key, default)
    return str(val).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')

# Флаги сохранения
SAVE_NORMAL_CAPTCHAS = get_bool_env('SAVE_NORMAL_CAPTCHAS', '0')
SAVE_SOMETHING_CAPTCHAS = get_bool_env('SAVE_SOMETHING_CAPTCHAS', '1')


def select_device():
    if torch is not None and getattr(torch, 'cuda', None) is not None and torch.cuda.is_available():
        return 0  # GPU 0
    return 'cpu'


def generate_random_filename(extension: str = 'png', length: int = 10) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length)) + f'.{extension}'


def load_model():
    if YOLO is None:
        raise RuntimeError('Ultralytics YOLO не установлен')
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f'Файл весов не найден: {WEIGHTS_PATH}')
    return YOLO(str(WEIGHTS_PATH))


model = load_model()
device = select_device()


def compute_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-6
    return inter / union


def nms_cross_class(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=int)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = compute_iou_xyxy(boxes[i], boxes[rest])
        rest = rest[ious < iou_thr]
        order = rest
    return np.array(keep, dtype=int)


def sort_left_to_right(boxes: np.ndarray, class_ids: np.ndarray, scores: np.ndarray):
    idxs = np.argsort(boxes[:, 0])
    return boxes[idxs], class_ids[idxs], scores[idxs]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_b64 = data['image']
    if isinstance(image_b64, str) and 'base64,' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    try:
        image_binary = base64.b64decode(image_b64)
    except Exception:
        return jsonify({'error': 'Invalid base64 image'}), 400

    size_bytes = len(image_binary)
    if size_bytes < 2048:
        try:
            logger.info(f"received_empty size_bytes={size_bytes}")
        except Exception:
            pass
        return jsonify({'error': 'empty image'}), 400

    temp_filename = generate_random_filename('png')
    save_path = CAPTCHA_DIR / temp_filename
    with open(save_path, 'wb') as f:
        f.write(image_binary)

    img_array = np.frombuffer(image_binary, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        # fallback
        img = cv2.imread(str(save_path))
        if img is None:
            return jsonify({'error': 'Unable to decode image'}), 400

    try:
        h, w = img.shape[:2]
        logger.info(f"received='{save_path.name}' size_bytes={len(image_binary)} resolution={w}x{h}")
    except Exception:
        logger.info(f"received='{save_path.name}' size_bytes={len(image_binary)}")

    recognized_text = ''
    agg_confidence = None  # минимальная уверенность среди детектов
    t0 = time.perf_counter()
    results = model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        iou=IOU_NMS,
        agnostic_nms=True,
        device=device,
        verbose=False,
    )
    infer_ms = (time.perf_counter() - t0) * 1000.0

    if results:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()

            conf_mask = scores >= CONF_THRESHOLD
            boxes_xyxy = boxes_xyxy[conf_mask]
            class_ids = class_ids[conf_mask]
            scores = scores[conf_mask]

            keep = nms_cross_class(boxes_xyxy, scores, iou_thr=IOU_NMS)
            boxes_xyxy = boxes_xyxy[keep]
            class_ids = class_ids[keep]
            scores = scores[keep]

            if len(boxes_xyxy) > 0:
                boxes_xyxy, class_ids, scores = sort_left_to_right(boxes_xyxy, class_ids, scores)
                detected_chars = []
                det_summaries = []
                for cls_id in class_ids:
                    if 0 <= int(cls_id) < len(CLASS_NAMES):
                        detected_chars.append(CLASS_NAMES[int(cls_id)])
                    else:
                        detected_chars.append('?')
                recognized_text = ''.join(detected_chars)

                # минимальная уверенность по символам
                try:
                    if len(scores) > 0:
                        agg_confidence = float(np.min(scores))
                except Exception:
                    pass

                try:
                    for (x1, y1, x2, y2), cid, sc in zip(boxes_xyxy, class_ids, scores):
                        ch = CLASS_NAMES[int(cid)] if 0 <= int(cid) < len(CLASS_NAMES) else '?'
                        det_summaries.append(f"{ch}({sc:.2f})[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
                    logger.info(
                        f"detections={len(class_ids)} details=" + (', '.join(det_summaries) if det_summaries else '-')
                    )
                except Exception:
                    pass
            else:
                logger.info("detections=0")

    agg_str = f"{agg_confidence:.2f}" if agg_confidence is not None else 'n/a'
    logger.info(
        f"solved='{recognized_text}' time_ms={infer_ms:.1f} device={device} conf={CONF_THRESHOLD} iou={IOU_NMS} agg_conf={agg_str} thr={LOW_CONF_THRESHOLD} save_normal={int(SAVE_NORMAL_CAPTCHAS)} save_something={int(SAVE_SOMETHING_CAPTCHAS)}"
    )

    should_low_bucket = (agg_confidence is None) or (agg_confidence < LOW_CONF_THRESHOLD)

    # Сохранение управляется флагами: SAVE_SOMETHING_CAPTCHAS и SAVE_NORMAL_CAPTCHAS
    should_save_something = should_low_bucket and SAVE_SOMETHING_CAPTCHAS
    should_save_normal = (not should_low_bucket) and SAVE_NORMAL_CAPTCHAS and bool(recognized_text)

    if should_save_something:
        if recognized_text:
            target_name = f"{recognized_text}.png"
            target_path = LOW_CONF_DIR / target_name
            if target_path.exists():
                idx = 1
                while True:
                    candidate = LOW_CONF_DIR / f"{recognized_text}_{idx}.png"
                    if not candidate.exists():
                        target_path = candidate
                        break
                    idx += 1
        else:
            target_path = LOW_CONF_DIR / save_path.name
        try:
            save_path.rename(target_path)
        except Exception:
            pass
        else:
            logger.info(f"saved_as='{target_path.as_posix()}'")
    elif should_save_normal:
        target_name = f"{recognized_text}.png"
        target_path = CAPTCHA_DIR / target_name
        if target_path.exists():
            idx = 1
            while True:
                candidate = CAPTCHA_DIR / f"{recognized_text}_{idx}.png"
                if not candidate.exists():
                    target_path = candidate
                    break
                idx += 1
        try:
            save_path.rename(target_path)
        except Exception:
            pass
        else:
            logger.info(f"saved_as='{target_path.as_posix()}'")
    else:
        # Сохранение отключено — удаляем временный файл
        try:
            save_path.unlink()
        except Exception:
            pass

    return jsonify({'predicted_label': recognized_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


