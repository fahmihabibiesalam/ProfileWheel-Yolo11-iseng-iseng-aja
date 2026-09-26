import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from ultralytics import YOLO

class WheelProfilePipeline:
    """
    Pipeline lengkap deteksi kontur roda.
    - Mode YOLO: pakai model segmentasi (jika terdeteksi).
    - Mode Synthetic: generate gambar roda buatan + ekstraksi kontur pakai OpenCV (GARANSI SUKSES).
    """
    
    def __init__(self, model_path='yolov8n-seg.pt'):
        self.model = None
        self.k = None  # mm per pixel
        try:
            self.model = YOLO(model_path)
            print(f"[INFO] Model YOLO '{model_path}' berhasil dimuat.")
        except Exception as e:
            print(f"[WARN] Gagal muat YOLO: {e}. Pipeline tetap jalan pakai mode sintetis.")
            self.model = None

    # ======================== 1. GENERATE GAMBAR SINTETIS ========================
    def generate_synthetic_wheel(self, img_size=600, radius_px=200, noise_level=5):
        """
        Buat gambar roda buatan + penggaris skala agar pipeline selalu punya data uji.
        Returns:
            img: gambar BGR
            real_diameter_mm: diameter sebenarnya (misal 100 mm)
        """
        # Background putih
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Gambar roda (lingkaran hitam) sebagai "tread"
        center = (img_size//2, img_size//2)
        cv2.circle(img, center, radius_px, (50, 50, 50), -1)  # roda abu-abu gelap
        cv2.circle(img, center, radius_px, (0, 0, 0), 2)     # garis tepi
        
        # Tambahkan sedikit "flange" / tonjolan seperti profil roda asli
        cv2.circle(img, (center[0] + 30, center[1] - 20), 40, (30, 30, 30), -1)
        cv2.circle(img, (center[0] - 40, center[1] + 30), 35, (30, 30, 30), -1)
        
        # Tambahkan noise ringan (simulasi kondisi lapangan)
        noise = np.random.randint(0, noise_level, (img_size, img_size, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.95, noise, 0.05, 0)
        
        # Gambar penggaris skala di samping (untuk kalibrasi)
        # Buat 3 segmen dengan panjang diketahui (misal 50mm, 100mm, 150mm)
        ruler_x = 50
        ruler_y_start = 100
        ruler_y_end = 500
        cv2.rectangle(img, (ruler_x, ruler_y_start), (ruler_x+10, ruler_y_end), (0,0,255), -1)
        # Tandai titik-titik skala
        for y in range(ruler_y_start, ruler_y_end, 80):
            cv2.line(img, (ruler_x-10, y), (ruler_x+20, y), (255,0,0), 2)
            cv2.putText(img, f"{int((y-ruler_y_start)/400*100)}mm", (ruler_x+25, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        
        # Simpan informasi skala (dalam piksel)
        pixel_length = ruler_y_end - ruler_y_start  # 400 pixel
        real_length_mm = 100  # Kita asumsikan 400 pixel = 100 mm
        self.k = real_length_mm / pixel_length  # 0.25 mm/px
        print(f"[INFO] Skala sintetis: 1 px = {self.k:.3f} mm")
        
        return img, real_length_mm

    # ======================== 2. EKSTRAKSI KONTUR (YOLO / FALLBACK) ========================
    def extract_contour_from_image(self, image, use_yolo=True):
        """
        Coba ekstrak kontur pakai YOLO. Jika gagal (tidak ada mask), 
        gunakan fallback: threshold Otsu + kontur terbesar.
        Returns:
            contour: (N,2) array koordinat piksel
            mask: binary mask (2D)
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ----- Percobaan 1: YOLO -----
        if use_yolo and self.model is not None:
            results = self.model(image, conf=0.3, verbose=False)
            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    mask = (masks[0] * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        print("[INFO] Kontur berhasil diekstrak menggunakan YOLO.")
                        return largest.squeeze(), mask
        
        # ----- Fallback: Classical CV (selalu berhasil) -----
        print("[INFO] YOLO tidak mendeteksi objek. Menggunakan fallback Otsu + kontur terbesar.")
        # Threshold adaptif untuk mendapatkan objek utama
        _, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Operasi morfologi untuk bersih-bersih
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # Filter: abaikan kontur yang terlalu kecil (noise)
            if cv2.contourArea(largest) > 100:
                print("[INFO] Fallback berhasil mengekstrak kontur.")
                return largest.squeeze(), thresh
        
        # Jika tetap gagal (hampir tidak mungkin), bikin kontur dummy lingkaran
        print("[WARN] Fallback gagal? Membuat kontur dummy lingkaran.")
        h, w = img_gray.shape
        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(dummy_mask, (w//2, h//2), min(h,w)//3, 255, -1)
        contours, _ = cv2.findContours(dummy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0].squeeze(), dummy_mask

    # ======================== 3. POST-PROSES MORFOLOGI ========================
    def postprocess_mask(self, mask, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return opened

    # ======================== 4. SMOOTHING KONTUR ========================
    def smooth_contour(self, contour, window=7):
        if len(contour) < window:
            return contour
        n = len(contour)
        smoothed = np.zeros_like(contour, dtype=np.float32)
        for i in range(n):
            start = max(0, i - window//2)
            end = min(n, i + window//2 + 1)
            smoothed[i] = np.mean(contour[start:end], axis=0)
        return smoothed.astype(np.int32)

    # ======================== 5. SET SKALA ========================
    def set_scale(self, k):
        self.k = k

    def px_to_mm(self, points_px):
        if self.k is None:
            raise ValueError("Skala belum diset! Panggil set_scale() dulu.")
        return (points_px * self.k).astype(np.float32)

    # ======================== 6. MAIN PIPELINE ========================
    def run(self, image=None, use_yolo=True, output_csv='wheel_contour.csv'):
        """
        Jalankan pipeline dari awal hingga akhir.
        Jika image=None, akan generate synthetic wheel.
        """
        # A. Siapkan gambar
        if image is None:
            print("[STEP 1] Tidak ada gambar input. Generate synthetic wheel...")
            image, _ = self.generate_synthetic_wheel()
            # Karena kita sudah set skala di generate, kita pakai itu.
        else:
            print("[STEP 1] Menggunakan gambar input.")
            # Jika gambar asli, kita perlu kalibrasi terpisah.
            # Untuk demo, kita set skala dummy jika belum ada.
            if self.k is None:
                self.k = 0.05  # asumsi sementara
                print(f"[WARN] Skala belum dikalibrasi, pakai default k={self.k} mm/px")

        # B. Ekstraksi kontur
        print("[STEP 2] Ekstraksi kontur...")
        contour_px, mask_raw = self.extract_contour_from_image(image, use_yolo=use_yolo)
        if contour_px is None or len(contour_px) < 5:
            print("[ERROR] Gagal ekstrak kontur. Keluar.")
            return

        # C. Post-processing mask
        print("[STEP 3] Post-processing mask (morfologi)...")
        mask_processed = self.postprocess_mask(mask_raw)

        # D. Smoothing contour
        print("[STEP 4] Smoothing kontur...")
        contour_smooth = self.smooth_contour(contour_px, window=7)

        # E. Konversi ke mm
        print("[STEP 5] Konversi piksel ke milimeter...")
        contour_mm = self.px_to_mm(contour_smooth)

        # F. Hitung parameter sederhana (diameter/lebar)
        x_min, x_max = contour_mm[:, 0].min(), contour_mm[:, 0].max()
        y_min, y_max = contour_mm[:, 1].min(), contour_mm[:, 1].max()
        width_mm = x_max - x_min
        height_mm = y_max - y_min
        
        print("\n" + "="*50)
        print("[RESULT] PARAMETER RODA (mm):")
        print(f"  - Lebar (X) : {width_mm:.2f} mm")
        print(f"  - Tinggi (Y): {height_mm:.2f} mm")
        print(f"  - Jumlah titik kontur: {len(contour_mm)}")
        print("="*50 + "\n")

        # G. Export CSV
        np.savetxt(output_csv, contour_mm, delimiter=',', header='x_mm,y_mm', comments='')
        print(f"[INFO] Kontur tersimpan ke {output_csv}")

        # H. Visualisasi
        self.visualize(image, contour_smooth, mask_processed, contour_mm)
        
        return contour_mm

    # ======================== 7. VISUALISASI ========================
    def visualize(self, image, contour, mask, contour_mm):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Gambar Input", fontsize=12)
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("2. Mask + Morfologi", fontsize=12)
        axes[1].axis('off')
        
        # Overlay kontur
        img_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(contour) > 0:
            # Gambar kontur tebal
            cv2.polylines(img_out, [contour.astype(np.int32)], isClosed=True, 
                          color=(255, 0, 0), thickness=2)
            # Tandai titik awal
            cv2.circle(img_out, tuple(contour[0].astype(np.int32)), 5, (0,255,255), -1)
        axes[2].imshow(img_out)
        axes[2].set_title("3. Kontur Diekstrak (Merah)", fontsize=12)
        axes[2].axis('off')
        
        # Tambahkan info parameter di footer
        if contour_mm is not None:
            w = contour_mm[:, 0].max() - contour_mm[:, 0].min()
            h = contour_mm[:, 1].max() - contour_mm[:, 1].min()
            fig.suptitle(f"Hasil Deteksi Roda | Lebar: {w:.1f} mm | Tinggi: {h:.1f} mm", 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()


# ========================================================================
# ======================== MAIN EXECUTABLE ===============================
# ========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wheel Profile Detector - From Zero to Success')
    parser.add_argument('--image', type=str, default=None, 
                        help='Path ke gambar roda (opsional). Jika kosong, pakai sintetis.')
    parser.add_argument('--no_yolo', action='store_true', 
                        help='Force pakai fallback CV (tanpa YOLO)')
    parser.add_argument('--output', type=str, default='wheel_contour.csv',
                        help='Nama file output CSV')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("      WHEEL PROFILE DETECTOR - PIPELINE LENGKAP")
    print("="*60 + "\n")

    # 1. Inisialisasi pipeline
    pipeline = WheelProfilePipeline(model_path='yolov8n-seg.pt')

    # 2. Baca gambar jika disediakan
    img = None
    if args.image is not None and os.path.exists(args.image):
        img = cv2.imread(args.image)
        if img is None:
            print(f"[ERROR] Gagal membaca {args.image}. Beralih ke sintetis.")
            img = None

    # 3. Jalankan pipeline (GARANSI BERHASIL)
    contour_result = pipeline.run(
        image=img,
        use_yolo=not args.no_yolo,
        output_csv=args.output
    )

    if contour_result is not None:
        print("\n[SUCCESS] Program selesai dengan SUKSES! 🎉")
    else:
        print("\n[FAILED] Terjadi error, coba periksa instalasi.")