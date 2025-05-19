# RCNN Object Detection Project

Pascal VOC veri seti üzerinde eğitilmiş bir RCNN (Region-based Convolutional Neural Network) nesne tanıma modeli.

## Proje Yapısı

- `models/`: Model tanımlamaları ve ilgili kod
  - `backbone.py`: Özellik çıkarıcı ağ mimarisi
  - `bbox_regressor.py`: Sınırlayıcı kutu regresyonu için model
  - `rcnn.py`: Ana RCNN modeli

- `utils/`: Yardımcı fonksiyonlar
  - `dataset.py`: Veri yükleme ve işleme
  - `metrics.py`: Değerlendirme metrikleri
  - `preprocessing.py`: Görüntü ön işleme
  - `visualization.py`: Sonuçları görselleştirme

- `config.py`: Konfigürasyon parametreleri
- `detect.py`: Nesne algılama için kod
- `evaluate.py`: Modeli değerlendirme
- `train.py`: Eğitim kodu

## Veri Seti

Bu proje Pascal VOC veri setini kullanmaktadır. Veri seti proje ile birlikte paylaşılmamıştır. Aşağıdaki kaynaklardan indirilebilir:

- [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

İndirilen veri setini `data/` klasörüne aşağıdaki şekilde yerleştirin:
- `data/annotations/`: XML formatında nesne etiketleri
- `data/images/`: JPG formatında görüntüler

## Kurulum

```bash
# Sanal ortam oluşturun
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya
.venv\Scripts\activate  # Windows

# Gereksinimleri yükleyin
pip install -r requirements.txt
