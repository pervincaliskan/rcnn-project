import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os


def plot_training_history(history, save_path=None):
    """
    Eğitim geçmişini görselleştir
    """
    plt.figure(figsize=(12, 4))

    # Eğitim ve doğrulama kaybı
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Doğrulama doğruluğu
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def visualize_detections(image, detections, class_names, save_path=None, show=True):
    """
    Nesne tespitlerini görselleştir
    """
    plt.figure(figsize=(10, 10))

    # Numpy dizisine dönüştür
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()

    plt.imshow(image)

    # Her tespiti çiz
    for box, label_id, score in detections:
        x, y, w, h = box

        # Sınırlayıcı kutuyu çiz
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        # Sınıf adını ve skoru ekle
        class_name = class_names[label_id]
        plt.text(x, y - 5, f'{class_name}: {score:.2f}',
                 color='white', fontsize=10,
                 bbox=dict(facecolor='red', alpha=0.7))

    plt.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()