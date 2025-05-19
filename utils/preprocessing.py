import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


def selective_search(image, mode='fast'):
    """
    Sabit sayıda basit bölge önerisi oluşturan alternatif fonksiyon
    OpenCV ximgproc kütüphanesini gerektirmez

    Args:
        image: Numpy dizisi olarak görüntü (RGB)
        mode: Bu versiyonda kullanılmıyor

    Returns:
        boxes: [x, y, width, height] formatında dikdörtgen önerileri
    """
    height, width = image.shape[:2]
    boxes = []

    # 1. Basit kayan pencere önerileri
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    aspect_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    window_sizes = [64, 96, 128, 192, 256]

    for window_size in window_sizes:
        for scale in scales:
            w = int(window_size * scale)
            for ar in aspect_ratios:
                h = int(w / ar)

                if w > width or h > height:
                    continue

                for y in range(0, height - h, max(1, h // 3)):
                    for x in range(0, width - w, max(1, w // 3)):
                        boxes.append([x, y, w, h])

    # 2. Görüntü bölme temelli öneriler
    for h_count in range(1, 5):  # Yatay bölme sayısı
        for v_count in range(1, 5):  # Dikey bölme sayısı
            cell_width = width // h_count
            cell_height = height // v_count

            for i in range(v_count):
                for j in range(h_count):
                    x = j * cell_width
                    y = i * cell_height
                    w = cell_width
                    h = cell_height
                    boxes.append([x, y, w, h])

                    # Daha büyük öneriler ekle
                    if j < h_count - 1 and i < v_count - 1:
                        w = cell_width * 2
                        h = cell_height * 2
                        boxes.append([x, y, w, h])

    # 3. Rastgele öneriler
    np.random.seed(42)  # Tekrarlanabilirlik için
    for _ in range(500):
        x = np.random.randint(0, width - 20)
        y = np.random.randint(0, height - 20)
        w = np.random.randint(20, min(width - x, 300))
        h = np.random.randint(20, min(height - y, 300))
        boxes.append([x, y, w, h])

    # Toplamda en fazla 2000 öneri
    if len(boxes) > 2000:
        indices = np.random.choice(len(boxes), 2000, replace=False)
        boxes = [boxes[i] for i in indices]

    return np.array(boxes)


def compute_iou(box1, box2):
    """
    İki kutu arasındaki IoU'yu hesapla
    box formatı: [x, y, width, height]
    """
    # Kesişim alanını hesapla
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Birleşim alanını hesapla
    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]
    union = area_box1 + area_box2 - intersection

    return intersection / union if union > 0 else 0


def crop_and_resize(image, box, output_size=(224, 224)):
    """
    Görüntüden bir bölgeyi kırp ve yeniden boyutlandır
    """
    x, y, w, h = box
    # Görüntü sınırlarını kontrol et
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(image.shape[1] - x, int(w))
    h = min(image.shape[0] - y, int(h))

    # Bölgeyi kırp
    cropped = image[y:y + h, x:x + w]

    # Yeniden boyutlandır
    resized = cv2.resize(cropped, output_size)

    return resized


def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Non-Maximum Suppression uygula
    """
    if len(boxes) == 0:
        return []

    # Kutuları koordinatlara göre sırala
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]

    keep = []

    while boxes.size > 0:
        # En yüksek skorlu kutuyu tut
        keep.append(indices[0])

        # Diğer tüm kutularla IoU hesapla
        ious = np.array([compute_iou(boxes[0], box) for box in boxes[1:]])

        # Eşik değerinden düşük IoU'lu kutuları tut
        mask = ious < threshold
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        indices = indices[1:][mask]

    return keep