"""GPU kontrolü ve TensorFlow GPU durumu kontrol scripti."""

import sys

try:
    import tensorflow as tf
except ModuleNotFoundError as _e:
    tf = None
    _TF_IMPORT_ERROR = _e


def _get_tf_version():
 
    # If tensorflow failed to import, try to obtain metadata, otherwise
    # return a clear 'not-installed' marker.
    if tf is None:
        try:
            try:
                from importlib.metadata import version as _version
            except Exception:
                from importlib_metadata import version as _version  # type: ignore
            return _version("tensorflow")
        except Exception:
            return "not-installed"

    # Preferred attribute
    v = getattr(tf, "__version__", None)
    if v:
        return v

    # Try importlib.metadata (py3.8+) or its backport
    try:
        try:
            from importlib.metadata import version as _version
        except Exception:
            from importlib_metadata import version as _version  # type: ignore
        return _version("tensorflow")
    except Exception:
        pass

    # Try pkg_resources as a last resort
    try:
        import pkg_resources

        return pkg_resources.get_distribution("tensorflow").version
    except Exception:
        return "unknown"


def main():
    print("=" * 60)
    print("TensorFlow GPU Kontrolü")
    print("=" * 60)
    
    print(f"\nTensorFlow Versiyonu: {_get_tf_version()}")

    if tf is None:
        print("\n[HATA] TensorFlow import edilemedi.")
        try:
            print(f"  Detay: {_TF_IMPORT_ERROR}")
        except Exception:
            pass
        print("  Cozum: Sanal ortamda TensorFlow'u kurun:")
        print("    - `pip install -r requirements.txt`")
        print("    - veya `pip install tensorflow`")
        return 2
    
    # Fiziksel GPU cihazlarını listele
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    
    print(f"\nFiziksel Cihazlar:")
    print(f"  CPU: {len(cpus)} adet")
    for cpu in cpus:
        print(f"    - {cpu.name}")
    
    print(f"  GPU: {len(gpus)} adet")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"    - {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                if details:
                    print(f"      Detaylar: {details}")
            except Exception:
                pass
    else:
        print("    [UYARI] GPU bulunamadi!")
    
    # CUDA ve cuDNN bilgisi
    print(f"\nCUDA Kutuphaneleri:")
    try:
        print(f"  CUDA: {tf.sysconfig.get_build_info()['cuda_version']}")
        print(f"  cuDNN: {tf.sysconfig.get_build_info()['cudnn_version']}")
    except (KeyError, AttributeError):
        print("  [UYARI] CUDA/cuDNN bilgisi alinamadi (CPU-only build olabilir)")
    
    # Test: Basit bir işlemi GPU'da çalıştırmayı dene
    print(f"\nGPU Test:")
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"  [OK] GPU'da matris carpimi basarili!")
            print(f"  Sonuc: {c.numpy()}")
    except Exception as e:
        print(f"  [HATA] GPU testi basarisiz: {e}")
        print(f"  [BILGI] CPU kullanilacak.")
    
    print("\n" + "=" * 60)
    
    if gpus:
        print("[OK] GPU kullanilabilir! Egitim GPU'da calisacak.")
        return 0
    else:
        print("[UYARI] GPU bulunamadi. Egitim CPU'da calisacak (daha yavas).")
        print("[BILGI] GPU kullanmak icin:")
        print("   1. NVIDIA GPU'nun kurulu oldugundan emin ol")
        print("   2. CUDA Toolkit ve cuDNN'in kurulu oldugundan emin ol")
        print("   3. TensorFlow GPU versiyonunun kurulu oldugundan emin ol")
        return 1


if __name__ == "__main__":
    sys.exit(main())

