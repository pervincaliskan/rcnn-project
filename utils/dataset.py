import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from utils.preprocessing import selective_search


class VOCDataset(Dataset):
    """
    Pascal VOC veri seti için veri yükleyici
    """

    def __init__(self, img_dir, annotation_dir, transform=None, proposal_method='selective_search'):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.proposal_method = proposal_method

        # Sınıf adlarının listesi (Pascal VOC için)
        self.classes = [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Görüntü dosyalarını bul
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Görüntüyü yükle
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Görüntüyü sabit boyuta yeniden boyutlandır
        resized_image = cv2.resize(image, (224, 224))

        # Etiketleri yükle
        img_id = os.path.splitext(self.img_files[idx])[0]
        anno_path = os.path.join(self.annotation_dir, img_id + '.xml')
        boxes, labels = self._parse_voc_annotation(anno_path)

        # Kutuları yeniden boyutlandırılmış görüntüye uygun şekilde ölçekle
        if len(boxes) > 0:
            # Orijinal görüntü boyutlarını al
            orig_height, orig_width = image.shape[:2]
            # Yeni boyutlar
            new_height, new_width = 224, 224

            # Ölçekleme faktörlerini hesapla
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height

            # Kutuları ölçekle
            scaled_boxes = boxes.copy()
            for i in range(len(boxes)):
                scaled_boxes[i][0] = boxes[i][0] * scale_x  # x
                scaled_boxes[i][1] = boxes[i][1] * scale_y  # y
                scaled_boxes[i][2] = boxes[i][2] * scale_x  # width
                scaled_boxes[i][3] = boxes[i][3] * scale_y  # height

            boxes = scaled_boxes

        # Bölge önerilerini oluştur
        proposals = selective_search(resized_image)

        # Görüntü dönüşümlerini uygula
        if self.transform:
            resized_image = self.transform(resized_image)

        return {
            'image': resized_image,
            'boxes': boxes,
            'labels': labels,
            'proposals': proposals,
            'img_id': img_id
        }

    def _parse_voc_annotation(self, anno_path):
        """
        Pascal VOC XML dosyasını ayrıştırma
        """
        tree = ET.parse(anno_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            if label not in self.classes:
                continue

            label_idx = self.class_to_idx[label]

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # [x, y, width, height] formatına dönüştür
            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            labels.append(label_idx)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def custom_collate_fn(batch):
    """
    Farklı boyutlardaki tensörler için özel collate fonksiyonu
    """
    # Boş bir sözlük oluştur
    batch_dict = {}

    # Batch'deki her veri türü için ayrı bir liste oluştur
    for key in batch[0].keys():
        batch_dict[key] = [item[key] for item in batch]

    # image tensörleri genellikle aynı boyuttadır, onları stack edebiliriz
    if torch.is_tensor(batch_dict['image'][0]):
        batch_dict['image'] = torch.stack(batch_dict['image'], 0)

    # Diğer öğeler (boxes, labels, proposals) farklı boyutlarda olabilir,
    # bu yüzden onları liste olarak bırakıyoruz

    return batch_dict


def create_data_loaders(config):
    """
    Eğitim, doğrulama ve test veri yükleyicilerini oluştur
    """
    from torch.utils.data import random_split
    from torchvision import transforms

    # Görüntü dönüşümleri
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Veri setini oluştur
    dataset = VOCDataset(
        img_dir=config['image_dir'],
        annotation_dir=config['annotation_dir'],
        transform=transform
    )

    # Veri setini böl
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.15)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Veri yükleyicileri oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # MultiProcessing devre dışı
        collate_fn=custom_collate_fn  # Özel collate fonksiyonu kullan
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # MultiProcessing devre dışı
        collate_fn=custom_collate_fn  # Özel collate fonksiyonu kullan
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # MultiProcessing devre dışı
        collate_fn=custom_collate_fn  # Özel collate fonksiyonu kullan
    )

    return train_loader, val_loader, test_loader