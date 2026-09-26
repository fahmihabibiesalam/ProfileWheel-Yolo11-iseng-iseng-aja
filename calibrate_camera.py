# calibrate_camera.py
# ====================================================================
#  VERSI DENGAN PEMILIHAN KAMERA MANUAL + DUKUNGAN LAYAR LAPTOP
# ====================================================================

import cv2
import numpy as np
import os
import glob
import time

# ============================================================
#  KONFIGURASI
# ============================================================

# INNER CORNERS untuk board 9x6 kotak = (8,5)
INNER_CORNERS = (8, 5)
SQUARE_SIZE_MM = 25
CALIB_DIR = "calib_images"

# ============================================================
#  1.  SCAN & PILIH KAMERA SECARA MANUAL
# ============================================================

def select_camera():
    """
    Memindai semua indeks kamera (0-5) dan menampilkan preview 3 detik.
    Pengguna memilih indeks yang sesuai (iVCam).
    """
    print("\n" + "="*60)
    print("📷 SCANNING KAMERA - Perhatikan preview di layar!")
    print("="*60)
    
    available = []
    
    for idx in range(6):
        print(f"\n🔍 Mengecek indeks {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"   ✅ Kamera di indeks {idx} BERHASIL dibuka.")
                print(f"   👀 Menampilkan preview selama 3 detik...")
                
                # Tampilkan preview dengan label indeks
                display = frame.copy()
                cv2.putText(display, f"Camera Index: {idx}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display, "Apakah ini iVCam (HP Anda)?", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display, "Jika YA, catat indeks ini!", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Camera Scanner", display)
                cv2.waitKey(3000)  # Tampilkan 3 detik
                cv2.destroyAllWindows()
                
                available.append(idx)
                cap.release()
            else:
                cap.release()
        else:
            print(f"   ❌ Indeks {idx} tidak bisa dibuka.")
    
    if not available:
        print("\n❌ TIDAK ADA KAMERA YANG DITEMUKAN!")
        print("   Pastikan iVCam Desktop berjalan dan koneksi HP aktif.")
        return None
    
    print("\n" + "="*60)
    print("📋 DAFTAR INDEKS KAMERA YANG BERHASIL:", available)
    print("="*60)
    print("💡 Tips identifikasi:")
    print("   - Kamera laptop: biasanya menampilkan wajah Anda sendiri.")
    print("   - iVCam (HP): menampilkan apa yang dilihat oleh HP Anda.")
    print("     (misalnya: meja, lantai, atau layar laptop)")
    print("="*60)
    
    while True:
        try:
            choice = int(input("\n🔢 Masukkan indeks KAMERA iVCam (contoh: 1 atau 2): "))
            if choice in available:
                print(f"✅ Anda memilih indeks {choice} (iVCam).")
                return choice
            else:
                print(f"❌ Indeks {choice} tidak tersedia. Pilih dari {available}")
        except ValueError:
            print("❌ Masukkan angka yang valid.")

# ============================================================
#  2.  GENERATE CHESSBOARD
# ============================================================

def generate_chessboard():
    os.makedirs(CALIB_DIR, exist_ok=True)
    img_w, img_h = 1200, 800
    board = np.zeros((img_h, img_w), dtype=np.uint8)
    sq_px = img_w // (INNER_CORNERS[0] + 1)
    
    for i in range(INNER_CORNERS[0] + 1):
        for j in range(INNER_CORNERS[1] + 1):
            if (i + j) % 2 == 0:
                x1, y1 = i * sq_px, j * sq_px
                x2, y2 = (i + 1) * sq_px, (j + 1) * sq_px
                cv2.rectangle(board, (x1, y1), (x2, y2), 255, -1)
    
    cv2.imwrite("chessboard_pattern.png", board)
    
    print("\n" + "="*60)
    print("🖨️  FILE CHESSBOARD TELAH DIBUAT: chessboard_pattern.png")
    print("="*60)
    print("📌 INSTRUKSI PENGGUNAAN LAYAR LAPTOP:")
    print("   1. Buka file 'chessboard_pattern.png' di laptop Anda.")
    print("   2. Tekan F11 (Fullscreen) atau buka dalam mode layar penuh.")
    print("   3. Atur kecerahan layar ke 100%.")
    print("   4. Arahkan HP (iVCam) ke arah layar laptop.")
    print("   5. Gerakkan HP ke berbagai sudut untuk mengambil gambar.")
    print("="*60 + "\n")
    
    return "chessboard_pattern.png"

# ============================================================
#  3.  CAPTURE GAMBAR
# ============================================================

def capture_images(camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Gagal membuka kamera yang dipilih.")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n📸 MODE CAPTURE")
    print("   📍 Arahkan HP ke LAYAR LAPTOP yang menampilkan chessboard.")
    print("   🟢 HIJAU = sudut terdeteksi (bisa simpan)")
    print("   🔴 MERAH = tidak terdeteksi (atur posisi/cahaya)")
    print("   Tekan [SPACE] untuk foto (hanya jika HIJAU), [ESC] selesai.\n")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(
            gray, INNER_CORNERS,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        display = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display, INNER_CORNERS, corners, ret_corners)
            status = f"✅ DETEKSI OK! ({count} tersimpan)"
            color = (0, 255, 0)
        else:
            status = f"❌ TIDAK TERDETEKSI"
            color = (0, 0, 255)
        
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Tersimpan: {count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(display, "SPACE=Simpan | ESC=Keluar", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        cv2.imshow("Kalibrasi - Live", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and ret_corners:
            filename = f"{CALIB_DIR}/calib_{count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   ✅ Tersimpan: {filename}")
            count += 1
        elif key == 32 and not ret_corners:
            print("   ⚠️  Tidak terdeteksi! Atur posisi HP terhadap layar.")
        elif key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📸 Total gambar: {count}")
    return count >= 8

# ============================================================
#  4.  KALIBRASI
# ============================================================

def run_calibration():
    images = glob.glob(f"{CALIB_DIR}/*.jpg")
    if len(images) < 8:
        print(f"❌ Hanya {len(images)} gambar, minimal 8.")
        return False
    
    print(f"\n📂 Memproses {len(images)} gambar...")
    
    objp = np.zeros((INNER_CORNERS[0] * INNER_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:INNER_CORNERS[0], 0:INNER_CORNERS[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    
    objpoints = []
    imgpoints = []
    img_shape = None
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(
            gray, INNER_CORNERS,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            print(f"   ✅ {os.path.basename(fname)}")
        else:
            print(f"   ⚠️  {os.path.basename(fname)} - dilewati")
    
    if len(objpoints) < 5:
        print(f"❌ Hanya {len(objpoints)} valid.")
        return False
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    
    print("\n" + "="*60)
    print("✅ KALIBRASI BERHASIL!")
    print(f"   Error proyeksi: {mean_error:.4f} px (baik jika < 0.5)")
    print("\n📷 Matriks Kamera:")
    print(mtx)
    print("\n🔧 Distorsi:")
    print(dist)
    print("="*60)
    
    np.savez("calibration.npz", mtx=mtx, dist=dist)
    print("💾 Tersimpan ke: calibration.npz")
    return True

# ============================================================
#  5.  MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("   🔧 KALIBRASI KAMERA iVCam + CHESSBOARD (LAYAR LAPTOP)")
    print("="*60 + "\n")
    
    # Step 1: Generate
    generate_chessboard()
    input("🖥️  Tampilkan 'chessboard_pattern.png' FULLSCREEN di laptop, lalu ENTER...")
    
    # Step 2: Pilih kamera
    cam_idx = select_camera()
    if cam_idx is None:
        return
    
    # Step 3: Capture
    if not capture_images(cam_idx):
        print("❌ Capture gagal. Ulangi.")
        return
    
    # Step 4: Calibrasi
    input("\n📸 Tekan ENTER untuk menjalankan kalibrasi...")
    run_calibration()

if __name__ == "__main__":
    main()