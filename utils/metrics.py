import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from utils.preprocessing import compute_iou


def calculate_ap(y_true, y_scores, iou_threshold=0.5):
    """
    Bir sınıf için Ortalama Kesinlik (AP) hesapla
    """
    # Skorlara göre sırala
    indices = np.argsort(y_scores)[::-1]

    true_positives = np.zeros(len(y_scores))
    false_positives = np.zeros(len(y_scores))

    total_gt_boxes = len(y_true)

    # Her tahmin için TP/FP değerleri hesapla
    for i, idx in enumerate(indices):
        pred_box = y_scores[idx]['box']
        max_iou = 0
        max_iou_idx = -1

        # En yüksek IoU'lu gt kutuyu bul
        for j, gt_box in enumerate(y_true):
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_idx = j

        # IoU eşik değerinden büyükse TP, değilse FP
        if max_iou >= iou_threshold and max_iou_idx != -1:
            true_positives[i] = 1
            # Bu gt kutuyu kaldır (bir gt kutu yalnızca bir tahmine eşlenebilir)
            y_true.pop(max_iou_idx)
        else:
            false_positives[i] = 1

    # Birikimli toplam
    cum_true_positives = np.cumsum(true_positives)
    cum_false_positives = np.cumsum(false_positives)

    # Kesinlik ve geri çağırma hesapla
    precision = cum_true_positives / (cum_true_positives + cum_false_positives)
    recall = cum_true_positives / total_gt_boxes

    # Kesinlik-geri çağırma eğrisi alanını hesapla (AP)
    ap = average_precision_score(y_true, y_scores)

    return ap, precision, recall


def calculate_map(predictions, ground_truths, class_names, iou_threshold=0.5):
    """
    Tüm sınıflar için Ortalama Kesinlik (mAP) hesapla
    """
    aps = []

    for class_id in range(len(class_names)):
        # Bu sınıf için tahminleri ve gerçek değerleri filtrele
        class_preds = [p for p in predictions if p['class_id'] == class_id]
        class_gts = [gt for gt in ground_truths if gt['class_id'] == class_id]

        if len(class_gts) == 0:
            continue

        # Bu sınıf için AP hesapla
        ap, _, _ = calculate_ap(class_gts, class_preds, iou_threshold)
        aps.append(ap)

    # Ortalama AP (mAP) hesapla
    mAP = np.mean(aps) if aps else 0

    return mAP, {class_names[i]: ap for i, ap in enumerate(aps)}