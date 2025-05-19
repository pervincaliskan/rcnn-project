import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from models.rcnn import RCNN
from models.bbox_regressor import BBoxRegressor
from utils.dataset import create_data_loaders
from utils.preprocessing import crop_and_resize, non_max_suppression
from utils.metrics import calculate_map
from utils.visualization import visualize_detections
from config import CONFIG


def detect_objects(model, bbox_regressor, image, proposals, device, config):
    """
    Bir görüntüde nesneleri tespit et
    """
    model.eval()
    bbox_regressor.eval()

    detections = []

    with torch.no_grad():
        # Her öneri için
        batch_features = []
        batch_proposals = []

        for prop in proposals:
            # Bölgeyi kırp ve yeniden boyutlandır
            try:
                cropped = crop_and_resize(image.cpu().numpy(), prop)
                batch_features.append(cropped)
                batch_proposals.append(prop)
            except Exception:
                continue

        if not batch_features:
            return detections

        # Mini-batch değerlendirme
        batch_size = 64
        num_batches = len(batch_features) // batch_size + (1 if len(batch_features) % batch_size != 0 else 0)

        all_scores = []
        all_labels = []
        all_boxes = []

        for mini_batch in range(num_batches):
            start_idx = mini_batch * batch_size
            end_idx = min((mini_batch + 1) * batch_size, len(batch_features))

            mini_batch_features = torch.stack(batch_features[start_idx:end_idx]).to(device)
            mini_batch_proposals = batch_proposals[start_idx:end_idx]

            # İleri geçiş
            features, cls_scores = model(mini_batch_features)
            reg_preds = bbox_regressor(features)

            # Sınıf skorlarını ve etiketlerini al
            cls_probs = torch.softmax(cls_scores, dim=1)
            max_scores, pred_labels = cls_probs.max(1)

            # Sınırlayıcı kutuları düzelt
            for i in range(len(mini_batch_proposals)):
                if pred_labels[i] == 0:  # Arka plan sınıfı, atla
                    continue

                score = max_scores[i].item()
                label = pred_labels[i].item()

                if score < config['score_threshold']:
                    continue

                # Öneri kutusunu düzelt
                box = mini_batch_proposals[i]
                dx, dy, dw, dh = reg_preds[i].cpu().numpy()

                # Kutu merkezini ve boyutlarını al
                width = box[2]
                height = box[3]
                ctr_x = box[0] + 0.5 * width
                ctr_y = box[1] + 0.5 * height

                # Kutuyu düzelt
                pred_ctr_x = dx * width + ctr_x
                pred_ctr_y = dy * height + ctr_y
                pred_w = np.exp(dw) * width
                pred_h = np.exp(dh) * height

                # [x, y, w, h] formatına dönüştür
                pred_box = [
                    pred_ctr_x - 0.5 * pred_w,
                    pred_ctr_y - 0.5 * pred_h,
                    pred_w,
                    pred_h
                ]

                all_scores.append(score)
                all_labels.append(label)
                all_boxes.append(pred_box)

        # Non-maximum suppression uygula
        if all_boxes:
            all_boxes = np.array(all_boxes)
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)

            # Her sınıf için ayrı NMS uygula
            for label in np.unique(all_labels):
                cls_indices = np.where(all_labels == label)[0]
                cls_boxes = all_boxes[cls_indices]
                cls_scores = all_scores[cls_indices]

                keep = non_max_suppression(cls_boxes, cls_scores, config['nms_threshold'])

                for idx in keep:
                    detections.append((
                        cls_boxes[idx],
                        label,
                        cls_scores[idx]
                    ))

    return detections


def evaluate_model(model, bbox_regressor, test_loader, config, device, visualize=False):
    """
    Test veri setinde modeli değerlendir
    """
    model.eval()
    bbox_regressor.eval()

    all_predictions = []
    all_ground_truths = []

    for batch_idx, batch in enumerate(tqdm(test_loader, desc='Değerlendirme')):
        images = batch['image'].to(device)
        gt_boxes = batch['boxes']
        gt_labels = batch['labels']
        proposals = batch['proposals']
        img_ids = batch['img_id']

        for i in range(len(images)):
            image = images[i]
            img_gt_boxes = gt_boxes[i]
            img_gt_labels = gt_labels[i]
            img_proposals = proposals[i]
            img_id = img_ids[i]

            # Görüntüdeki nesneleri tespit et
            detections = detect_objects(
                model, bbox_regressor, image, img_proposals, device, config
            )

            # Tahminleri kaydet
            for box, label, score in detections:
                all_predictions.append({
                    'image_id': img_id,
                    'box': box,
                    'class_id': label,
                    'score': score
                })

            # Temel gerçeklikleri kaydet
            for j in range(len(img_gt_boxes)):
                all_ground_truths.append({
                    'image_id': img_id,
                    'box': img_gt_boxes[j],
                    'class_id': img_gt_labels[j]
                })

            # Tespitleri görselleştir
            if visualize and batch_idx % 10 == 0:
                visualize_detections(
                    image.cpu().numpy().transpose(1, 2, 0),
                    detections,
                    config['classes'],
                    save_path=os.path.join(config['results_dir'], f'detection_{img_id}.png'),
                    show=False
                )

    # mAP hesapla
    mAP, class_aps = calculate_map(
        all_predictions,
        all_ground_truths,
        config['classes'],
        iou_threshold=config['iou_threshold']
    )

    return mAP, class_aps


def main():
    # Argümanları ayrıştır
    parser = argparse.ArgumentParser(description='R-CNN Değerlendirme')
    parser.add_argument('--config', type=str, default='config.py', help='Konfigürasyon dosyası')
    parser.add_argument('--checkpoint', type=str, required=True, help='Değerlendirilecek model')
    parser.add_argument('--visualize', action='store_true', help='Tespitleri görselleştir')
    args = parser.parse_args()

    # Konfigürasyonu yükle
    config = CONFIG

    # GPU kullanılabilirliğini kontrol et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # Veri yükleyicileri oluştur
    _, _, test_loader = create_data_loaders(config)
    print(f"Test örnekleri: {len(test_loader.dataset)}")

    # Model oluştur
    model = RCNN(
        num_classes=config['num_classes'],
        backbone_name=config['backbone'],
        pretrained=False
    ).to(device)

    bbox_regressor = BBoxRegressor(input_dim=config['feature_dim']).to(device)

    # Kontrol noktasını yükle
    if os.path.isfile(args.checkpoint):
        print(f"Kontrol noktasından yükleniyor: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        bbox_regressor.load_state_dict(checkpoint['bbox_regressor_state_dict'])
    else:
        print(f"Kontrol noktası bulunamadı: {args.checkpoint}")
        return

    # Modeli değerlendir
    mAP, class_aps = evaluate_model(
        model, bbox_regressor, test_loader, config, device, args.visualize
    )

    # Sonuçları yazdır
    print(f"mAP: {mAP:.4f}")
    print("Sınıf AP değerleri:")
    for cls, ap in class_aps.items():
        print(f"{cls}: {ap:.4f}")

    # Sonuçları kaydet
    results = {
        'mAP': mAP,
        'class_aps': class_aps
    }

    with open(os.path.join(config['results_dir'], 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()