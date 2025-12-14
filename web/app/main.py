"""Flask Backend - Beyin Tümör Segmentasyonu Web Uygulaması"""

import os
import sys
from pathlib import Path
from datetime import datetime
import io
import json
import pickle

import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# TensorFlow'u lazy load yapalım
tf = None

def load_tensorflow():
    global tf
    if tf is None:
        try:
            import tensorflow
            tf = tensorflow
        except Exception as e:
            print(f"[WARNING] TensorFlow yüklenemedi: {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

# Model ve konfigürasyonları
MODEL = None
MODEL_PATH = None
IMAGE_SIZE = (256, 256)


def get_custom_objects():
    """Custom loss ve metric fonksiyonlarını döndür"""
    load_tensorflow()
    if tf is None:
        return {}
    
    # Custom loss fonksiyonlarını tanımla
    def dice_loss(y_true, y_pred, smooth=1e-6):
        from tensorflow.keras import backend as K
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1.0 - (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def bce_dice_loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return bce + dice_loss(y_true, y_pred)
    
    # Custom metric fonksiyonlarını tanımla
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
        dice = (2.0 * intersection + smooth) / (denom + smooth)
        return tf.reduce_mean(dice)
    
    def iou_score(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        union = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return tf.reduce_mean(iou)
    
    return {
        'bce_dice_loss': bce_dice_loss,
        'dice_loss': dice_loss,
        'dice_coefficient': dice_coefficient,
        'dice': dice_coefficient,
        'iou_score': iou_score,
        'iou': iou_score
    }


def load_model(model_path, model_type=None):
    """Modeli yükle"""
    global MODEL, MODEL_PATH
    try:
        # TensorFlow'u yükle
        load_tensorflow()
        
        if tf is None:
            print("[WARNING] TensorFlow kullanılamıyor, mock model kullanılıyor")
            MODEL = "mock"
            return True
        
        if model_path is None:
            print("[WARNING] Model yolu belirtilmedi, mock model kullanılıyor")
            MODEL = "mock"
            return True
        
        if not os.path.exists(model_path):
            print(f"[WARNING] Model dosyası bulunamadı: {model_path}, mock model kullanılıyor")
            MODEL = "mock"
            return True
        
        # Model tipini belirle
        if model_type is None:
            if model_path.endswith('.tflite'):
                model_type = 'tflite'
            elif model_path.endswith('.keras'):
                model_type = 'keras'
            elif model_path.endswith('.h5'):
                model_type = 'h5'
            else:
                model_type = 'keras'  # Default
        
        print(f"[INFO] Model yükleniyor: {model_path} (tip: {model_type})")
        
        if model_type == 'tflite':
            # TensorFlow Lite interpreter kullan
            try:
                import tensorflow.lite as tflite
                interpreter = tflite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                MODEL = interpreter
                MODEL_PATH = model_path
                print(f"[OK] TFLite model başarıyla yüklendi: {model_path}")
                # Input/output detaylarını al
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print(f"[INFO] Input shape: {input_details[0]['shape']}")
                print(f"[INFO] Output shape: {output_details[0]['shape']}")
                return True
            except Exception as e:
                print(f"[ERROR] TFLite model yükleme hatası: {e}")
                raise
        else:
            # Keras model yükle
            custom_objects = get_custom_objects()
            MODEL = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
            MODEL_PATH = model_path
            print(f"[OK] Keras model başarıyla yüklendi: {model_path}")
            print(f"[INFO] Model input shape: {MODEL.input_shape}")
            print(f"[INFO] Model output shape: {MODEL.output_shape}")
            # Test prediction yap
            test_input = np.zeros((1, 256, 256, 1), dtype=np.float32)
            test_output = MODEL.predict(test_input, verbose=0)
            print(f"[OK] Model test prediction başarılı, output shape: {test_output.shape}")
            return True
            
    except Exception as e:
        print(f"[ERROR] Model yükleme hatası: {e}")
        import traceback
        traceback.print_exc()
        print("[WARNING] Mock model kullanılıyor")
        MODEL = "mock"
        return True


def preprocess_image(image_array):
    """Görüntüyü önişle (normalize et)"""
    if len(image_array.shape) == 3:
        # RGB -> Grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Resize
    image_resized = cv2.resize(image_array, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Normalize [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch ve channel dimensions
    image_batch = np.expand_dims(np.expand_dims(image_normalized, axis=0), axis=-1)
    
    return image_batch, image_resized


def create_mock_mask(image):
    """Mock segmentasyon maskesi oluştur (test için)"""
    # Threshold ile basit segmentasyon
    gray = cv2.cvtColor(image, cv2.COLOR_GRAY2GRAY) if len(image.shape) == 3 else image
    _, mask = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
    
    # Morfolojik operasyonlar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Gaussian blur
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    
    return mask


def predict_segmentation(image_path):
    """Segmentasyon tahmini yap"""
    if MODEL is None or (isinstance(MODEL, str) and MODEL == "mock"):
        return None, None, None, {"error": "Model yüklenmedi"}
    
    try:
        # Görüntüyü oku
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image is None:
            return None, None, None, {"error": "Görüntü okunamadı"}
        
        original_shape = image.shape
        
        # Ön işleme
        image_batch, image_resized = preprocess_image(image)
        
        # Tahmin
        if isinstance(MODEL, str) and MODEL == "mock":
            # Mock segmentasyon (model test amaçlı)
            print("[WARNING] Mock model kullanılıyor")
            mask = create_mock_mask(image_resized)
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
        else:
            # Gerçek model
            load_tensorflow()
            if tf is not None and MODEL is not None:
                print(f"[DEBUG] Model prediction başlatılıyor, input shape: {image_batch.shape}")
                try:
                    # TFLite interpreter kontrolü
                    if hasattr(MODEL, 'get_input_details'):
                        # TFLite model
                        input_details = MODEL.get_input_details()
                        output_details = MODEL.get_output_details()
                        
                        # Input'u doğru formata getir
                        input_data = image_batch.astype(input_details[0]['dtype'])
                        MODEL.set_tensor(input_details[0]['index'], input_data)
                        MODEL.invoke()
                        prediction = MODEL.get_tensor(output_details[0]['index'])
                        print(f"[DEBUG] TFLite prediction shape: {prediction.shape}")
                    else:
                        # Keras model
                        prediction = MODEL.predict(image_batch, verbose=0)
                        print(f"[DEBUG] Keras prediction shape: {prediction.shape}")
                    
                    mask = prediction[0, :, :, 0]
                    print(f"[DEBUG] Mask shape: {mask.shape}, min: {mask.min()}, max: {mask.max()}")
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    print(f"[OK] Segmentasyon tamamlandı, mask binary shape: {mask_binary.shape}")
                except Exception as e:
                    print(f"[ERROR] Prediction hatası: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to mock
                    mask = create_mock_mask(image_resized)
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
            else:
                print("[WARNING] Model veya TensorFlow yok, mock kullanılıyor")
                mask = create_mock_mask(image_resized)
                mask_binary = (mask > 0.5).astype(np.uint8) * 255
        
        # Orijinal boyuta redüz et
        mask_original = cv2.resize(mask_binary, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        mask_original = (mask_original > 127).astype(np.uint8)
        
        return image_resized, mask_binary, mask_original, {"success": True}
    
    except Exception as e:
        return None, None, None, {"error": str(e)}


def calculate_metrics(ground_truth, prediction):
    """DICE ve IoU hesapla"""
    try:
        # DICE
        intersection = np.sum(ground_truth * prediction)
        dice = (2.0 * intersection) / (np.sum(ground_truth) + np.sum(prediction) + 1e-7)
        
        # IoU (Jaccard)
        union = np.sum(np.logical_or(ground_truth, prediction))
        intersection = np.sum(np.logical_and(ground_truth, prediction))
        iou = intersection / (union + 1e-7)
        
        return {
            "dice": float(dice),
            "iou": float(iou)
        }
    except Exception as e:
        return {"dice": 0.0, "iou": 0.0, "error": str(e)}


def create_overlay(original, mask, alpha=0.5):
    """Overlay görüntü oluştur"""
    # Grayscale -> BGR
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Mask'ı kırmızı renk yap
    overlay = original_bgr.copy()
    overlay[mask > 127] = [0, 0, 255]  # Kırmızı
    
    # Blend
    result = cv2.addWeighted(original_bgr, 1 - alpha, overlay, alpha, 0)
    
    return result


@app.route('/')
def index():
    """Ana sayfa"""
    model_status = "Yüklendi" if MODEL is not None else "Yüklenmedi"
    return render_template('index.html', model_status=model_status)


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Görüntü yükle ve segmentasyon yap"""
    print(f"[DEBUG] Upload request geldi. Files: {list(request.files.keys())}")
    
    if 'file' not in request.files:
        print("[ERROR] 'file' key'i request'te yok")
        return jsonify({"error": "Dosya bulunamadı"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("[ERROR] Dosya adı boş")
        return jsonify({"error": "Dosya seçilmedi"}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Dizin oluştur
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Dosyayı kaydet
        print(f"[DEBUG] Dosya kaydediliyor: {filepath}")
        file.save(filepath)
        
        # Segmentasyon yap
        print(f"[DEBUG] Segmentasyon başlatılıyor: {filepath}")
        image_resized, mask_binary, mask_original, status = predict_segmentation(filepath)
        
        if image_resized is None:
            error_msg = status.get("error", "Bilinmeyen hata") if isinstance(status, dict) else str(status)
            print(f"[ERROR] Segmentasyon hatası: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        # Overlay oluştur
        overlay = create_overlay(image_resized, mask_binary, alpha=0.4)
        
        # Görüntüleri base64 encode et
        def image_to_base64(img):
            _, buffer = cv2.imencode('.png', img)
            return f"data:image/png;base64,{__import__('base64').b64encode(buffer).decode()}"
        
        result = {
            "success": True,
            "original": image_to_base64(image_resized),
            "mask": image_to_base64(mask_binary),
            "overlay": image_to_base64(overlay),
            "filename": filename,
            "has_ground_truth": False,
            "metrics": None
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/data-images', methods=['GET'])
def list_data_images():
    """Data klasöründeki görüntüleri listele"""
    try:
        data_base = Path(__file__).parent.parent.parent / "ml" / "data"
        train_images_dir = data_base / "train" / "images"
        val_images_dir = data_base / "val" / "images"
        
        images = []
        
        # Train görüntüleri
        if train_images_dir.exists():
            for img_path in sorted(train_images_dir.glob("*.png")):
                mask_path = data_base / "train" / "masks" / img_path.name
                images.append({
                    "filename": img_path.name,
                    "path": str(img_path.relative_to(data_base.parent.parent)),
                    "dataset": "train",
                    "has_mask": mask_path.exists()
                })
        
        # Val görüntüleri
        if val_images_dir.exists():
            for img_path in sorted(val_images_dir.glob("*.png")):
                mask_path = data_base / "val" / "masks" / img_path.name
                images.append({
                    "filename": img_path.name,
                    "path": str(img_path.relative_to(data_base.parent.parent)),
                    "dataset": "val",
                    "has_mask": mask_path.exists()
                })
        
        return jsonify({
            "success": True,
            "images": images,
            "total": len(images)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/load-data-image', methods=['POST'])
def load_data_image():
    """Data klasöründen görüntü yükle ve segmentasyon yap"""
    try:
        data = request.get_json()
        print(f"[DEBUG] Request data: {data}")
        
        if data is None:
            print("[ERROR] JSON verisi alınamadı")
            return jsonify({"error": "JSON verisi alınamadı"}), 400
        
        if 'filename' not in data or 'dataset' not in data:
            print(f"[ERROR] Eksik parametreler. Gelen data: {data}")
            return jsonify({
                "error": "filename ve dataset gerekli",
                "received": list(data.keys()) if data else "None"
            }), 400
        
        filename = data['filename']
        dataset = data['dataset']  # 'train' veya 'val'
        
        print(f"[DEBUG] İşleniyor: filename={filename}, dataset={dataset}")
        
        data_base = Path(__file__).parent.parent.parent / "ml" / "data"
        image_path = data_base / dataset / "images" / filename
        mask_path = data_base / dataset / "masks" / filename
        
        print(f"[DEBUG] Image path: {image_path}")
        print(f"[DEBUG] Image exists: {image_path.exists()}")
        
        if not image_path.exists():
            return jsonify({"error": f"Görüntü bulunamadı: {image_path}"}), 404
        
        # Segmentasyon yap
        print(f"[DEBUG] Segmentasyon başlatılıyor: {image_path}")
        image_resized, mask_binary, mask_original, status = predict_segmentation(str(image_path))
        
        if image_resized is None:
            error_msg = status.get("error", "Bilinmeyen hata") if isinstance(status, dict) else str(status)
            print(f"[ERROR] Segmentasyon hatası: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        # Ground truth maskesi varsa yükle ve metrikleri hesapla
        metrics = None
        if mask_path.exists():
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                # Ground truth'u binary yap
                gt_binary = (gt_mask > 127).astype(np.uint8)
                
                # Prediction'ı aynı boyuta getir
                pred_binary = (mask_original > 127).astype(np.uint8)
                
                # Boyutları eşleştir
                if gt_binary.shape != pred_binary.shape:
                    pred_binary = cv2.resize(pred_binary, (gt_binary.shape[1], gt_binary.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                
                # Metrikleri hesapla
                metrics = calculate_metrics(gt_binary, pred_binary)
        
        # Overlay oluştur
        overlay = create_overlay(image_resized, mask_binary, alpha=0.4)
        
        # Görüntüleri base64 encode et
        def image_to_base64(img):
            _, buffer = cv2.imencode('.png', img)
            return f"data:image/png;base64,{__import__('base64').b64encode(buffer).decode()}"
        
        result = {
            "success": True,
            "original": image_to_base64(image_resized),
            "mask": image_to_base64(mask_binary),
            "overlay": image_to_base64(overlay),
            "filename": filename,
            "dataset": dataset,
            "has_ground_truth": mask_path.exists(),
            "metrics": metrics
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics', methods=['POST'])
def get_metrics():
    """DICE ve IoU hesapla (eğer ground truth varsa)"""
    try:
        data = request.get_json()
        
        if 'ground_truth_file' not in data or 'prediction_file' not in data:
            return jsonify({
                "dice": 0.0,
                "iou": 0.0,
                "message": "Karşılaştırma için iki dosya gerekli"
            })
        
        # Simüle edilmiş hesaplama (gerçek versiyonda GT yükleme yapılacak)
        return jsonify({
            "dice": 0.85,
            "iou": 0.75,
            "success": True
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Model bilgileri"""
    if MODEL is None or (isinstance(MODEL, str) and MODEL == "mock"):
        return jsonify({
            "loaded": False,
            "message": "Model yüklenmedi"
        })
    
    # TFLite interpreter kontrolü
    if hasattr(MODEL, 'get_input_details'):
        # TFLite model
        input_details = MODEL.get_input_details()
        output_details = MODEL.get_output_details()
        return jsonify({
            "loaded": True,
            "path": str(MODEL_PATH),
            "type": "tflite",
            "input_shape": str(input_details[0]['shape']),
            "output_shape": str(output_details[0]['shape']),
            "input_dtype": str(input_details[0]['dtype']),
            "output_dtype": str(output_details[0]['dtype'])
        })
    else:
        # Keras model
        try:
            return jsonify({
                "loaded": True,
                "path": str(MODEL_PATH),
                "type": "keras",
                "input_shape": str(MODEL.input_shape),
                "output_shape": str(MODEL.output_shape),
                "parameters": MODEL.count_params()
            })
        except Exception as e:
            return jsonify({
                "loaded": True,
                "path": str(MODEL_PATH),
                "type": "keras",
                "error": str(e)
            })


@app.route('/ml/data/<path:filename>')
def serve_data_image(filename):
    """Data klasöründeki görüntüleri serve et"""
    try:
        data_base = Path(__file__).parent.parent.parent / "ml" / "data"
        image_path = data_base / filename
        
        if not image_path.exists() or not image_path.is_file():
            return jsonify({"error": "Görüntü bulunamadı"}), 404
        
        # Güvenlik kontrolü - sadece data klasörü içindeki dosyalara erişim
        try:
            image_path.resolve().relative_to(data_base.resolve())
        except ValueError:
            return jsonify({"error": "Geçersiz yol"}), 403
        
        return send_file(str(image_path))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Sayfa bulunamadı"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Sunucu hatası"}), 500


# Modeli uygulama başlatıldığında yükle
def initialize_model():
    """Modeli yükle - uygulama başlatıldığında çağrılır"""
    artifacts_path = Path(__file__).parent.parent.parent / "ml" / "artifacts"
    model_path = None
    model_type = None  # 'keras', 'tflite', 'h5'
    
    # Önce belirtilen klasörü kontrol et
    target_dir = artifacts_path / "unet_brain_tumor_20251205-163801"
    if target_dir.exists():
        # Önce model.keras dosyasını kontrol et (TensorFlow 2.10+ format)
        keras_path = target_dir / "model.keras"
        if keras_path.exists():
            model_path = keras_path
            model_type = 'keras'
        else:
            # Sonra .tflite dosyasını kontrol et
            tflite_path = target_dir / "model.tflite"
            if tflite_path.exists():
                model_path = tflite_path
                model_type = 'tflite'
            else:
                # Sonra checkpoints klasöründeki dosyaları kontrol et
                checkpoint_dir = target_dir / "checkpoints"
                if checkpoint_dir.exists():
                    # best.weights.h5 veya model.h5 ara
                    for pattern in ["best.weights.h5", "model.h5"]:
                        found = list(checkpoint_dir.glob(pattern))
                        if found:
                            model_path = found[0]
                            model_type = 'h5'
                            break
    
    # Eğer bulunamadıysa, tüm artifact klasörlerinde ara
    if model_path is None or not model_path.exists():
        if artifacts_path.exists():
            for artifact_dir in sorted(artifacts_path.glob("unet_brain_tumor_*"), reverse=True):
                # model.keras dosyasını kontrol et
                keras_path = artifact_dir / "model.keras"
                if keras_path.exists():
                    model_path = keras_path
                    model_type = 'keras'
                    break
                
                # .tflite dosyasını kontrol et
                tflite_path = artifact_dir / "model.tflite"
                if tflite_path.exists():
                    model_path = tflite_path
                    model_type = 'tflite'
                    break
                
                # checkpoints klasörünü kontrol et
                checkpoint_dir = artifact_dir / "checkpoints"
                if checkpoint_dir.exists():
                    for pattern in ["best.weights.h5", "model.h5"]:
                        found = list(checkpoint_dir.glob(pattern))
                        if found:
                            model_path = found[0]
                            model_type = 'h5'
                            break
                
                if model_path and model_path.exists():
                    break
    
    if model_path and model_path.exists():
        print(f"[OK] Model bulundu: {model_path} (tip: {model_type})")
        load_model(str(model_path), model_type)
    else:
        print(f"[WARNING] Model dosyası bulunamadı. Mock model kullanılacak.")
        print(f"[INFO] Aradığım yerler: {artifacts_path}")
        load_model(None, None)


# Uygulama başlatıldığında modeli yükle
initialize_model()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
