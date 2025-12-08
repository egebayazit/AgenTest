import requests
import base64
import json
import os

# --- AYARLAR (Buraları kendi resmine göre düzenle) ---
SERVER_URL = "http://127.0.0.1:8000"
IMAGE_PATH = "test_screenshot.png"

# TEST 1: Bu koordinatlarda ne var? (Paint'ten bakıp buraya yaz)
TEST_X = 234
TEST_Y = 56

# TEST 2: Bu element nerede? (Resimde gördüğün bir isim yaz)
TEST_ELEMENT_NAME = "Whatsapp" 

def encode_image_to_base64(path):
    if not os.path.exists(path):
        print(f"HATA: '{path}' dosyası bulunamadı!")
        exit()
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_get_id_from_ods(b64_img):
    print("\n--- TEST 1: Koordinattan İsim Bulma (/get-id-from-ods) ---")
    print(f"Sorgulanan Koordinat: ({TEST_X}, {TEST_Y})")
    
    payload = {
        "base64_image": b64_img,
        "x": TEST_X,  # DÜZELTİLDİ: Değişken kullanıldı
        "y": TEST_Y   # DÜZELTİLDİ: Değişken kullanıldı
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/get-id-from-ods", json=payload)
        if response.status_code == 200:
            print("✅ BAŞARILI:", json.dumps(response.json(), indent=2))
        else:
            print("❌ HATA:", response.status_code, response.text)
    except Exception as e:
        print(f"❌ Bağlantı Hatası: {e}")

def test_get_coords_from_ods(b64_img):
    print("\n--- TEST 2: İsimden Koordinat Bulma (/get-coords-from-ods) ---")
    print(f"Aranan Element: '{TEST_ELEMENT_NAME}'")
    
    payload = {
        "base64_image": b64_img,
        "content_id": TEST_ELEMENT_NAME # DÜZELTİLDİ: Değişken kullanıldı
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/get-coords-from-ods", json=payload)
        if response.status_code == 200:
            print("✅ BAŞARILI:", json.dumps(response.json(), indent=2))
        else:
            print("❌ HATA:", response.status_code, response.text)
    except Exception as e:
        print(f"❌ Bağlantı Hatası: {e}")

if __name__ == "__main__":
    # 1. Resmi Base64'e çevir
    print(f"Resim yükleniyor: {IMAGE_PATH}...")
    b64_image = encode_image_to_base64(IMAGE_PATH)
    print("Resim Base64 formatına çevrildi.")

    # 2. Testleri Çalıştır
    test_get_id_from_ods(b64_image)
    test_get_coords_from_ods(b64_image)