import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import argparse
import json

from models.rcnn import RCNN
from models.bbox_regressor import BBoxRegressor
from utils.dataset import create_data_loaders
from utils.preprocessing import crop_and_resize, compute_iou
from utils.visualization import plot_training_history
from config import CONFIG


def train_one_epoch(model, bbox_regressor, train_loader, optimizer, criterion, device, epoch, num_epochs):
    """
    Bir eğitim döngüsü çalıştır
    """
    model.train()
    bbox_regressor.train()

    train_loss = 0.0
    cls_loss = 0.0
    reg_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        gt_boxes = batch['boxes']  # Liste olarak
        gt_labels = batch['labels']  # Liste olarak
        proposals = batch['proposals']  # Liste olarak

        # Her görüntüdeki her bölge önerisi için
        batch_features = []
        batch_cls_labels = []
        batch_reg_targets = []
        batch_proposals = []

        for i in range(len(images)):
            img = images[i]
            img_gt_boxes = gt_boxes[i]  # Doğrudan liste öğesi
            img_gt_labels = gt_labels[i]  # Doğrudan liste öğesi
            img_proposals = proposals[i]  # Doğrudan liste öğesi

            # Her öneri için IoU hesapla
            img_batch_features = []
            img_batch_cls_labels = []
            img_batch_reg_targets = []
            img_batch_proposals = []

            for prop in img_proposals:
                # IoU hesapla
                ious = [compute_iou(prop, gt_box) for gt_box in img_gt_boxes]
                max_iou = max(ious) if ious else 0
                max_iou_idx = np.argmax(ious) if ious else -1

                # Etiket ata (IoU > 0.5 -> pozitif, < 0.3 -> negatif)
                if max_iou >= 0.5:
                    cls_label = img_gt_labels[max_iou_idx]
                    # Sınırlayıcı kutu regresyon hedefi hesapla
                    gt_box = img_gt_boxes[max_iou_idx]
                    reg_target = bbox_transform(prop, gt_box)
                elif max_iou < 0.3:
                    cls_label = 0  # Arka plan
                    reg_target = np.zeros(4, dtype=np.float32)
                else:
                    continue  # Gri bölge, atla

                # Bölgeyi kırp ve yeniden boyutlandır
                try:
                    cropped = crop_and_resize(img.cpu().numpy(), prop)
                    # Tensor'a dönüştür ve cihaza taşı
                    cropped_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
                    # Normalize et
                    cropped_tensor = torch.nn.functional.normalize(
                        cropped_tensor,
                        mean=torch.tensor([0.485, 0.456, 0.406]),
                        std=torch.tensor([0.229, 0.224, 0.225])
                    )
                    img_batch_features.append(cropped_tensor)
                    img_batch_cls_labels.append(cls_label)
                    img_batch_reg_targets.append(reg_target)
                    img_batch_proposals.append(prop)
                except Exception as e:
                    # print(f"Hata: {e}")
                    continue

            # Eğitim için mini-batch oluştur
            if img_batch_features:
                batch_features.extend(img_batch_features)
                batch_cls_labels.extend(img_batch_cls_labels)
                batch_reg_targets.extend(img_batch_reg_targets)
                batch_proposals.extend(img_batch_proposals)

        if not batch_features:
            continue

        # Mini-batch eğitimi
        batch_size = 64  # Mini-batch boyutu
        num_batches = len(batch_features) // batch_size + (1 if len(batch_features) % batch_size != 0 else 0)

        for mini_batch in range(num_batches):
            start_idx = mini_batch * batch_size
            end_idx = min((mini_batch + 1) * batch_size, len(batch_features))

            # Boş batch kontrolü
            if start_idx >= len(batch_features):
                continue

            try:
                mini_batch_features = torch.stack(batch_features[start_idx:end_idx]).to(device)
                mini_batch_cls_labels = torch.tensor(batch_cls_labels[start_idx:end_idx]).to(device)
                mini_batch_reg_targets = torch.tensor(batch_reg_targets[start_idx:end_idx]).to(device)
            except Exception as e:
                # print(f"Batch oluşturma hatası: {e}")
                continue

            # İleri geçiş
            optimizer.zero_grad()
            features, cls_scores = model(mini_batch_features)
            reg_preds = bbox_regressor(features)

            # Kayıpları hesapla
            cls_criterion = nn.CrossEntropyLoss()
            reg_criterion = nn.SmoothL1Loss()

            batch_cls_loss = cls_criterion(cls_scores, mini_batch_cls_labels)
            batch_reg_loss = reg_criterion(reg_preds, mini_batch_reg_targets)

            # Toplam kayıp
            loss = batch_cls_loss + batch_reg_loss

            # Geri yayılım
            loss.backward()
            optimizer.step()

            # Kayıpları topla
            train_loss += loss.item()
            cls_loss += batch_cls_loss.item()
            reg_loss += batch_reg_loss.item()

        # İlerleme çubuğunu güncelle
        progress_bar.set_postfix({
            'loss': train_loss / (batch_idx + 1),
            'cls_loss': cls_loss / (batch_idx + 1),
            'reg_loss': reg_loss / (batch_idx + 1)
        })

    # Ortalama kayıpları hesapla
    train_loss /= len(train_loader)
    cls_loss /= len(train_loader)
    reg_loss /= len(train_loader)

    return {
        'loss': train_loss,
        'cls_loss': cls_loss,
        'reg_loss': reg_loss
    }


def validate(model, bbox_regressor, val_loader, criterion, device):
    """
    Doğrulama kümesinde modeli değerlendir
    """
    model.eval()
    bbox_regressor.eval()

    val_loss = 0.0
    cls_loss = 0.0
    reg_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validating')):
            images = batch['image'].to(device)
            gt_boxes = batch['boxes']  # Liste olarak
            gt_labels = batch['labels']  # Liste olarak
            proposals = batch['proposals']  # Liste olarak

            # Her görüntüdeki her bölge önerisi için
            batch_features = []
            batch_cls_labels = []
            batch_reg_targets = []

            for i in range(len(images)):
                img = images[i]
                img_gt_boxes = gt_boxes[i]
                img_gt_labels = gt_labels[i]
                img_proposals = proposals[i]

                # Her öneri için IoU hesapla
                for prop in img_proposals:
                    ious = [compute_iou(prop, gt_box) for gt_box in img_gt_boxes]
                    max_iou = max(ious) if ious else 0
                    max_iou_idx = np.argmax(ious) if ious else -1

                    # Etiket ata
                    if max_iou >= 0.5:
                        cls_label = img_gt_labels[max_iou_idx]
                        gt_box = img_gt_boxes[max_iou_idx]
                        reg_target = bbox_transform(prop, gt_box)
                    elif max_iou < 0.3:
                        cls_label = 0  # Arka plan
                        reg_target = np.zeros(4, dtype=np.float32)
                    else:
                        continue  # Gri bölge, atla

                    # Bölgeyi kırp ve yeniden boyutlandır
                    try:
                        cropped = crop_and_resize(img.cpu().numpy(), prop)
                        # Tensor'a dönüştür ve normalize et
                        cropped_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
                        cropped_tensor = torch.nn.functional.normalize(
                            cropped_tensor,
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225])
                        )
                        batch_features.append(cropped_tensor)
                        batch_cls_labels.append(cls_label)
                        batch_reg_targets.append(reg_target)
                    except Exception:
                        continue

            if not batch_features:
                continue

            # Mini-batch değerlendirme
            batch_size = 64
            num_batches = len(batch_features) // batch_size + (1 if len(batch_features) % batch_size != 0 else 0)

            for mini_batch in range(num_batches):
                start_idx = mini_batch * batch_size
                end_idx = min((mini_batch + 1) * batch_size, len(batch_features))

                # Boş batch kontrolü
                if start_idx >= len(batch_features):
                    continue

                try:
                    mini_batch_features = torch.stack(batch_features[start_idx:end_idx]).to(device)
                    mini_batch_cls_labels = torch.tensor(batch_cls_labels[start_idx:end_idx]).to(device)
                    mini_batch_reg_targets = torch.tensor(batch_reg_targets[start_idx:end_idx]).to(device)
                except Exception as e:
                    # print(f"Doğrulama batch hatası: {e}")
                    continue

                # İleri geçiş
                features, cls_scores = model(mini_batch_features)
                reg_preds = bbox_regressor(features)

                # Kayıpları hesapla
                cls_criterion = nn.CrossEntropyLoss()
                reg_criterion = nn.SmoothL1Loss()

                batch_cls_loss = cls_criterion(cls_scores, mini_batch_cls_labels)
                batch_reg_loss = reg_criterion(reg_preds, mini_batch_reg_targets)

                loss = batch_cls_loss + batch_reg_loss

                # Kayıpları topla
                val_loss += loss.item()
                cls_loss += batch_cls_loss.item()
                reg_loss += batch_reg_loss.item()

                # Doğruluk hesapla
                _, predicted = cls_scores.max(1)
                total += mini_batch_cls_labels.size(0)
                correct += predicted.eq(mini_batch_cls_labels).sum().item()

    # Ortalama kayıpları ve doğruluğu hesapla
    val_loss /= (len(val_loader) if len(val_loader) > 0 else 1)
    cls_loss /= (len(val_loader) if len(val_loader) > 0 else 1)
    reg_loss /= (len(val_loader) if len(val_loader) > 0 else 1)
    accuracy = 100.0 * correct / total if total > 0 else 0

    return {
        'loss': val_loss,
        'cls_loss': cls_loss,
        'reg_loss': reg_loss,
        'accuracy': accuracy
    }


def bbox_transform(ex_box, gt_box):
    """
    Sınırlayıcı kutu regresyon hedeflerini hesapla
    ex_box: öneri kutusu [x, y, w, h]
    gt_box: temel gerçeklik kutusu [x, y, w, h]
    """
    ex_width = ex_box[2]
    ex_height = ex_box[3]
    ex_ctr_x = ex_box[0] + 0.5 * ex_width
    ex_ctr_y = ex_box[1] + 0.5 * ex_height

    gt_width = gt_box[2]
    gt_height = gt_box[3]
    gt_ctr_x = gt_box[0] + 0.5 * gt_width
    gt_ctr_y = gt_box[1] + 0.5 * gt_height

    # Hedef değerleri hesapla
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_width if ex_width > 0 else 0
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_height if ex_height > 0 else 0
    targets_dw = np.log(gt_width / ex_width) if ex_width > 0 else 0
    targets_dh = np.log(gt_height / ex_height) if ex_height > 0 else 0

    return np.array([targets_dx, targets_dy, targets_dw, targets_dh], dtype=np.float32)


def save_checkpoint(model, bbox_regressor, optimizer, epoch, loss, accuracy, save_path):
    """
    Model kontrol noktası kaydet
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'bbox_regressor_state_dict': bbox_regressor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, save_path)
    print(f"Kontrol noktası kaydedildi: {save_path}")


def main():
    # Argümanları ayrıştır
    parser = argparse.ArgumentParser(description='R-CNN Eğitimi')
    parser.add_argument('--config', type=str, default='config.py', help='Konfigürasyon dosyası')
    parser.add_argument('--resume', type=str, default=None, help='Eğitimi devam ettirmek için kontrol noktası')
    args = parser.parse_args()

    # Konfigürasyonu yükle
    config = CONFIG

    # Sonuç klasörünü oluştur
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['model_save_path'], exist_ok=True)

    # GPU kullanılabilirliğini kontrol et
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Cihaz: {device}")

    # Veri yükleyicileri oluştur
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f"Eğitim örnekleri: {len(train_loader.dataset)}")
    print(f"Doğrulama örnekleri: {len(val_loader.dataset)}")
    print(f"Test örnekleri: {len(test_loader.dataset)}")

    # Model oluştur
    model = RCNN(
        num_classes=config['num_classes'],
        backbone_name=config['backbone'],
        pretrained=config['pretrained']
    ).to(device)

    bbox_regressor = BBoxRegressor(input_dim=config['feature_dim']).to(device)

    # Optimize edici
    params = list(model.parameters()) + list(bbox_regressor.parameters())
    optimizer = optim.SGD(
        params,
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # Öğrenme oranı planlayıcısı
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Kayıp fonksiyonu
    criterion = nn.CrossEntropyLoss()

    # Eğitim geçmişi
    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_reg_loss': [],
        'val_loss': [],
        'val_cls_loss': [],
        'val_reg_loss': [],
        'val_accuracy': []
    }

    start_epoch = 0

    # Eğitimi devam ettir
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Kontrol noktasından yükleniyor: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            bbox_regressor.load_state_dict(checkpoint['bbox_regressor_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Devam ediliyor: epoch {start_epoch}")
        else:
            print(f"Kontrol noktası bulunamadı: {args.resume}")

    # Eğitim döngüsü
    print("Eğitime başlanıyor...")
    for epoch in range(start_epoch, config['num_epochs']):
        # Eğitim
        train_metrics = train_one_epoch(
            model, bbox_regressor, train_loader, optimizer, criterion, device, epoch, config['num_epochs']
        )

        # Doğrulama
        val_metrics = validate(model, bbox_regressor, val_loader, criterion, device)

        # Öğrenme oranını güncelle
        scheduler.step()

        # Metrikleri kaydet
        history['train_loss'].append(train_metrics['loss'])
        history['train_cls_loss'].append(train_metrics['cls_loss'])
        history['train_reg_loss'].append(train_metrics['reg_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_cls_loss'].append(val_metrics['cls_loss'])
        history['val_reg_loss'].append(val_metrics['reg_loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        # Sonuçları yazdır
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(
            f"Train Loss: {train_metrics['loss']:.4f} (Cls: {train_metrics['cls_loss']:.4f}, Reg: {train_metrics['reg_loss']:.4f})")
        print(
            f"Val Loss: {val_metrics['loss']:.4f} (Cls: {val_metrics['cls_loss']:.4f}, Reg: {val_metrics['reg_loss']:.4f})")
        print(f"Val Accuracy: {val_metrics['accuracy']:.2f}%")

        # Kontrol noktası kaydet
        save_checkpoint(
            model, bbox_regressor, optimizer, epoch,
            val_metrics['loss'], val_metrics['accuracy'],
            os.path.join(config['model_save_path'], f'checkpoint_epoch_{epoch + 1}.pth')
        )

        # En iyi modeli kaydet (en düşük doğrulama kaybına sahip)
        if len(history['val_loss']) > 0 and val_metrics['loss'] == min(history['val_loss']):
            save_checkpoint(
                model, bbox_regressor, optimizer, epoch,
                val_metrics['loss'], val_metrics['accuracy'],
                os.path.join(config['model_save_path'], 'best_model.pth')
            )

    # Eğitim geçmişini görselleştir
    plot_training_history(
        history,
        save_path=os.path.join(config['results_dir'], 'training_history.png')
    )

    # Eğitim geçmişini kaydet
    with open(os.path.join(config['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f)

    print("Eğitim tamamlandı!")


if __name__ == "__main__":
    main()