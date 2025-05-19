import os
import cv2
import torch
import numpy as np
import argparse
from models.rcnn import RCNN
from models.bbox_regressor import BBoxRegressor
from utils.preprocessing import selective_search, crop_and_resize, non_max_suppression
from utils.visualization import visualize_detections
from config import CONFIG


def load_model(checkpoint_path, config, device):
    """
    Kaydedilmiş modeli yükle
    """
    # Model oluştur
    model = RCNN(
        num_classes=config['num_classes'],
        backbone_name=config['backbone'],
        pretrained=False
    ).to(device)

    bbox_regressor = BBoxRegressor(input_dim=config['feature_dim']).to(device)

    # Kontrol noktasını yükle
    if os.path.isfile(checkpoint_path):
        print(f"Kontrol noktasından yükleniyor: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        bbox_regressor.load_state_dict(checkpoint['bbox_regressor_state_dict'])
    else:
        print(f"Kontrol noktası bulunamadı: {checkpoint_path}")
        return None, None

    return model, bbox_regressor


def preprocess_image(image, transform=None):
    """
    Görüntüyü ön işle
    """
    # BGR'dan RGB'ye dönüştür
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Dönüşümleri uygula
    if transform:
        image = transform(image)
    else:
        # Basit normalizasyon
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).float()

    return image


def detect_objects_in_image(model, bbox_regressor, image_path, config, device, output_path=None):
    """
    Bir görüntüdeki nesneleri tespit et
    """
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        return None

    # Bölge önerilerini oluştur
    proposals = selective_search(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Görüntüyü ön işle
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Batch boyutu ekle

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
                cropped = crop_and_resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), prop, output_size=(224, 224))
                cropped_tensor = preprocess_image(cropped)
                batch_features.append(cropped_tensor)
                batch_proposals.append(prop)
            except Exception as e:
                continue

        if not batch_features:
            return []

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
                    max(0, pred_ctr_x - 0.5 * pred_w),
                    max(0, pred_ctr_y - 0.5 * pred_h),
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

    # Tespitleri görselleştir
    if output_path is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualize_detections(
            rgb_image,
            detections,
            config['classes'],
            save_path=output_path,
            show=False
        )

    return detections


def main():
    # Argümanları ayrıştır
    parser = argparse.ArgumentParser(description='R-CNN Nesne Tespiti')
    parser.add_argument('--config', type=str, default='config.py', help='Konfigürasyon dosyası')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model kontrol noktası')
    parser.add_argument('--input', type=str, required=True, help='Giriş görüntüsü veya klasörü')
    parser.add_argument('--output', type=str, default='results', help='Çıkış klasörü')
    args = parser.parse_args()

    # Konfigürasyonu yükle
    config = CONFIG

    # GPU kullanılabilirliğini kontrol et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # Model yükle
    model, bbox_regressor = load_model(args.checkpoint, config, device)
    if model is None or bbox_regressor is None:
        return

    # Çıkış klasörünü oluştur
    os.makedirs(args.output, exist_ok=True)

    # Giriş bir klasör mü yoksa dosya mı kontrol et
    if os.path.isdir(args.input):
        # Klasördeki tüm görüntüleri işle
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                       if os.path.splitext(f)[1].lower() in image_extensions]

        for image_path in image_files:
            print(f"İşleniyor: {image_path}")
            output_path = os.path.join(args.output, os.path.basename(image_path))
            detect_objects_in_image(model, bbox_regressor, image_path, config, device, output_path)
    else:
        # Tek bir görüntüyü işle
        print(f"İşleniyor: {args.input}")
        output_path = os.path.join(args.output, os.path.basename(args.input))
        detections = detect_objects_in_image(model, bbox_regressor, args.input, config, device, output_path)

        # Sonuçları yazdır
        print(f"Tespit edilen nesneler: {len(detections)}")
        for box, label, score in detections:
            class_name = config['classes'][label]
            print(f"{class_name}: {score:.4f} at {box}")


if __name__ == "__main__":
    main()