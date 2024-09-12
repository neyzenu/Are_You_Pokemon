import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


model = tf.keras.models.load_model('pokemon.h5')
label_classes = np.load('pokemon.npy', allow_pickle=True).item()  # Etiket sınıflarını yükle


class_labels = {v: k for k, v in label_classes.items()}


def classify_image(image_path, model, class_labels):
    image = load_img(image_path, target_size=(128, 128))  # Resmi 128x128 boyutuna getir
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalizasyon

    prediction = model.predict(image)
    predicted_class_idx = np.argmax(prediction)  # En yüksek olasılığa sahip sınıf
    confidence = np.max(prediction)  # En yüksek olasılığı al
    predicted_class_label = class_labels[predicted_class_idx]

    return predicted_class_label, confidence

# Yeni bir resmi sınıflandır ve eminlik yüzdesini göster
result_label, confidence = classify_image('Your Test Image', model, class_labels)
confidence_percentage = confidence * 100
print(f' This picture {result_label} belonging to the classe confidence_percentage %{confidence_percentage:.2f} sure.')
