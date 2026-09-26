# calibrate_camera.py
import cv2
import numpy as np
import os
import glob

# === CONFIGURATION ===
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (width, height)
SQUARE_SIZE_MM = 25       # Actual size of each chessboard square in mm
CALIB_IMAGES_DIR = "calib_images"

# 1. Generate and print the chessboard (run this once)
def generate_chessboard():
    os.makedirs(CALIB_IMAGES_DIR, exist_ok=True)
    img_w, img_h = 1200, 800
    board = np.zeros((img_h, img_w), dtype=np.uint8)
    sq_px = img_w // (CHESSBOARD_SIZE[0] + 1)
    
    for i in range(CHESSBOARD_SIZE[0] + 1):
        for j in range(CHESSBOARD_SIZE[1] + 1):
            if (i + j) % 2 == 0:
                x1, y1 = i * sq_px, j * sq_px
                x2, y2 = (i + 1) * sq_px, (j + 1) * sq_px
                cv2.rectangle(board, (x1, y1), (x2, y2), 255, -1)
    
    cv2.imwrite("chessboard_pattern.png", board)
    print(f"✅ Chessboard saved as 'chessboard_pattern.png'. Print it on A4 paper.")
    print(f"   Each square is ~{sq_px} px. Actual size should be {SQUARE_SIZE_MM} mm.")

# 2. Capture calibration images using iVCam
def capture_calibration_images():
    cap = cv2.VideoCapture(0)  # iVCam is usually index 0
    if not cap.isOpened():
        print("❌ iVCam not found. Please check the connection.")
        return
    
    print("\n📸 Press [SPACE] to capture, [ESC] to stop. Move the chessboard around.")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: continue
        cv2.putText(frame, f"Images: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Capture Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            cv2.imwrite(f"{CALIB_IMAGES_DIR}/calib_{count:03d}.jpg", frame)
            print(f"✅ Captured {count}")
            count += 1
        elif key == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"📸 Captured {count} images.")

# 3. Perform calibration
def calibrate():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    images = glob.glob(f"{CALIB_IMAGES_DIR}/*.jpg")
    
    if len(images) < 10:
        print(f"❌ Need at least 10 images. Found {len(images)}.")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    if len(objpoints) < 5:
        print("❌ Not enough valid chessboard detections.")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    print("\n" + "="*50)
    print("✅ CALIBRATION SUCCESSFUL!")
    print(f"Reprojection Error: {ret:.4f} pixels")
    print("Camera Matrix (K):\n", mtx)
    print("Distortion Coefficients (k1,k2,p1,p2,k3):\n", dist)
    print("="*50)
    
    # Save the parameters
    np.savez("calibration.npz", mtx=mtx, dist=dist)
    print("💾 Calibration saved to 'calibration.npz'")

if __name__ == "__main__":
    # Uncomment the steps sequentially:
    # generate_chessboard()      # Step 1: Generate pattern
    # capture_calibration_images() # Step 2: Capture images (run after printing)
    calibrate()               # Step 3: Run calibration