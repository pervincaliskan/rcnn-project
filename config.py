# Proje konfigürasyonu
CONFIG = {
    # Veri
    'data_dir': 'data/',
    'image_dir': 'data/images/',
    'annotation_dir': 'data/annotations/',
    'processed_dir': 'data/processed/',
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,

    # Model
    'backbone': 'resnet18',  # ['resnet18', 'resnet34', 'resnet50', 'vgg16']
    'pretrained': True,
    'feature_dim': 512,

    # Eğitim
    'batch_size': 2,
    'num_epochs': 5,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 5e-4,

    # Nesne Tespiti
    'num_classes': 20,  # Pascal VOC için 20, COCO için 80, özel veri setinize göre ayarlayın
    'iou_threshold': 0.5,
    'nms_threshold': 0.3,
    'score_threshold': 0.5,
    'max_proposals': 2000,

    # Çıktılar
    'model_save_path': 'models/saved/',
    'results_dir': 'results/',
}