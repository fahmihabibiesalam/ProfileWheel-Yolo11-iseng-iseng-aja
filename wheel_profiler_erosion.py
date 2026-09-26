# wheel_profiler_erosion.py
# ====================================================================
#  ACCURATE REALTIME WHEEL EROSION (PENGIKISAN) MONITORING
#  Features:
#  1. Multi-Segment Scale Calibration (averaged for accuracy)
#  2. Rigid 2D Contour Alignment (corrects for camera shifts)
#  3. Per-Point Erosion Depth Calculation (Heatmap ready)
#  4. P/B/H Delta Tracking
# ====================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from ultralytics import YOLO
import sys
import time
from scipy.spatial import cKDTree

# ============================================================
#  CONFIGURATION
# ============================================================
CALIB_FILE = "calibration.npz"
YOLO_MODEL_PATH = "yolov8n-seg.pt"  # or "best.pt" if fine-tuned
IVCAM_INDEX = 1                    # Change this to your iVCam index
FRAME_SKIP = 2                     # Process every 2nd frame

# Standard measurement height for Flange Thickness (B)
FLANGE_MEASURE_HEIGHT = 12.0  # mm

class ErosionProfiler:
    def __init__(self):
        print("\n" + "="*70)
        print("🚆 RAIL TRANSIT SMART EYE - EROSION MONITOR v3.0")
        print("   Accurate Wear Measurement with Contour Alignment")
        print("="*70)

        # --- Load Camera Calibration ---
        if os.path.exists(CALIB_FILE):
            data = np.load(CALIB_FILE)
            self.mtx = data['mtx']
            self.dist = data['dist']
            print("✅ Camera calibration loaded.")
        else:
            print("❌ Calibration file missing! Run calibrate_camera.py first.")
            sys.exit(1)

        # --- Load YOLO ---
        if os.path.exists(YOLO_MODEL_PATH):
            self.model = YOLO(YOLO_MODEL_PATH)
            print(f"✅ YOLO model loaded: {YOLO_MODEL_PATH}")
        else:
            print(f"⚠️  YOLO not found. Falling back to robust Canny.")
            self.model = None

        # --- Measurement State ---
        self.k = None                     # Scale factor (mm/px)
        self.baseline_contour_mm = None   # Reference (new wheel) profile
        self.baseline_params = None       # {'P':, 'B':, 'H':}
        self.baseline_set = False

        self.current_contour_mm = None
        self.current_params = None
        self.erosion_depth_mm = None      # Array of erosion depths per point

        # --- Video Capture ---
        self.cap = cv2.VideoCapture(IVCAM_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"❌ Cannot open iVCam at index {IVCAM_INDEX}.")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"📷 Camera ready: {int(self.cap.get(3))}x{int(self.cap.get(4))}\n")

    # ============================================================
    #  STEP 1: ACCURATE SCALE CALIBRATION (Multi-Segment Avg)
    # ============================================================
    def calibrate_scale_accurate(self, frame):
        """
        User clicks 3 different known lengths on a ruler.
        The script averages the resulting k values for maximum accuracy.
        """
        print("\n📏 ACCURATE SCALE CALIBRATION (Multi-Segment)")
        print("   Click 2 points for a known distance (e.g., 0mm to 50mm).")
        print("   Repeat for 3 different segments. The script averages them.")
        print("   Press [ESC] to skip.")
        
        clone = frame.copy()
        points = []
        segments = []
        
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(clone, str(len(points)), (x+10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.imshow("Accurate Scale", clone)

        cv2.imshow("Accurate Scale", clone)
        cv2.setMouseCallback("Accurate Scale", mouse_cb)

        collected_k = []
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                self.k = 0.1
                print(f"⚠️  Scale skipped. Using dummy k={self.k:.4f} mm/px")
                return
            
            if key == 13 and len(points) == 2:  # ENTER to finalize a segment
                cv2.destroyWindow("Accurate Scale")
                px_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
                try:
                    real_mm = float(input(f"Enter real distance for this segment (mm): "))
                    k_seg = real_mm / px_dist
                    collected_k.append(k_seg)
                    print(f"   Segment {len(collected_k)}: k = {k_seg:.4f} mm/px")
                except ValueError:
                    print("   Invalid input. Segment discarded.")
                
                # Reset points for next segment
                points = []
                clone = frame.copy()
                cv2.imshow("Accurate Scale", clone)
                cv2.setMouseCallback("Accurate Scale", mouse_cb)
                
                if len(collected_k) >= 3:
                    break

        cv2.destroyAllWindows()
        
        # Average the collected K values for high accuracy
        self.k = np.mean(collected_k)
        std_k = np.std(collected_k)
        print(f"\n✅ Scale finalized: 1 px = {self.k:.4f} mm")
        print(f"   (Std Dev: {std_k:.5f} - Lower is better)")

    # ============================================================
    #  STEP 2: UNDISTORTION
    # ============================================================
    def undistort(self, img):
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, self.mtx, self.dist, None, new_mtx)
        x, y, w, h = roi
        return dst[y:y+h, x:x+w]

    # ============================================================
    #  STEP 3: CONTOUR EXTRACTION (YOLO + Morphology)
    # ============================================================
    def extract_contour(self, img):
        """Extracts the wheel profile contour."""
        mask_binary = None
        
        # --- YOLO attempt ---
        if self.model is not None:
            results = self.model(img, conf=0.4, verbose=False)
            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    mask_binary = (masks[0] * 255).astype(np.uint8)

        # --- Fallback: Adaptive Canny (robust to lighting) ---
        if mask_binary is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # Otsu's threshold for adaptive edge detection
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            edges = cv2.Canny(gray, 30, 100)
            mask_binary = cv2.bitwise_or(edges, thresh)
            kernel = np.ones((7, 7), np.uint8)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

        # --- Post-processing (Slide 2.2.3) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return None

        # Smoothing
        epsilon = 0.001 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        contour_points = approx.squeeze()
        
        if contour_points.ndim == 1:
            contour_points = contour_points.reshape(-1, 2)
        return contour_points

    # ============================================================
    #  STEP 4: P/B/H CALCULATION
    # ============================================================
    def calculate_pbh(self, contour_px):
        if self.k is None:
            return None
        contour_mm = contour_px * self.k
        self.current_contour_mm = contour_mm

        # Flange tip (lowest point in Y)
        tip_idx = np.argmax(contour_mm[:, 1])
        flange_tip = contour_mm[tip_idx]
        tip_y = flange_tip[1]

        # Tread line (top 10% of points)
        y_min = contour_mm[:, 1].min()
        tread_points = contour_mm[contour_mm[:, 1] <= y_min + 5.0 * self.k]
        tread_y = np.mean(tread_points[:, 1]) if len(tread_points) > 0 else y_min

        # H (Rim Width)
        H = tread_points[:, 0].max() - tread_points[:, 0].min() if len(tread_points) > 0 else 0

        # P (Flange Height)
        P = tip_y - tread_y

        # B (Thickness at FLANGE_MEASURE_HEIGHT above tread)
        measure_y = tread_y + FLANGE_MEASURE_HEIGHT
        near_points = contour_mm[np.abs(contour_mm[:, 1] - measure_y) < 2.0 * self.k]
        if len(near_points) >= 2:
            B = near_points[:, 0].max() - near_points[:, 0].min()
        else:
            B = flange_tip[0] - tread_points[:, 0].min() if len(tread_points) > 0 else 0

        return {'P': P, 'B': B, 'H': H, 'flange_tip': flange_tip, 'tread_y': tread_y}

    # ============================================================
    #  STEP 5: CONTOUR ALIGNMENT (CRITICAL FOR ACCURATE EROSION)
    # ============================================================
    def align_contours(self, baseline_mm, current_mm):
        """
        Aligns the current contour to the baseline using Rigid 2D Translation.
        Aligns based on the geometric centers of the bounding boxes.
        This removes X/Y shift errors caused by camera vibration.
        """
        # Compute centroids of bounding boxes
        bbox_b = np.array([baseline_mm[:, 0].min(), baseline_mm[:, 1].min(),
                           baseline_mm[:, 0].max(), baseline_mm[:, 1].max()])
        bbox_c = np.array([current_mm[:, 0].min(), current_mm[:, 1].min(),
                           current_mm[:, 0].max(), current_mm[:, 1].max()])
        
        center_b = np.array([(bbox_b[0] + bbox_b[2])/2, (bbox_b[1] + bbox_b[3])/2])
        center_c = np.array([(bbox_c[0] + bbox_c[2])/2, (bbox_c[1] + bbox_c[3])/2])
        
        # Translation vector (shift current to match baseline)
        dx = center_b[0] - center_c[0]
        dy = center_b[1] - center_c[1]
        
        aligned_current = current_mm + np.array([dx, dy])
        return aligned_current

    # ============================================================
    #  STEP 6: EROSION DEPTH CALCULATION (Per-Point)
    # ============================================================
    def compute_erosion(self, baseline_mm, current_mm_aligned):
        """
        For every point on the current profile, find the nearest point on the baseline.
        The distance (in mm) is the erosion depth.
        Positive distance = Material removed (erosion).
        """
        # Build KD-Tree for baseline for fast nearest-neighbor search
        tree = cKDTree(baseline_mm)
        
        # For each point in current, find distance to nearest baseline point
        distances, indices = tree.query(current_mm_aligned, k=1)
        
        # Positive distance means the current point is "pushed in" (worn)
        # We need a sign. If current is inside the baseline (towards center), it's erosion.
        # We use the normals. Simplified: we check the direction from baseline to current.
        # We'll just return the absolute distance for the heatmap, but for P/B/H deltas we use subtraction.
        erosion_depths = distances  # in mm
        
        return erosion_depths

    # ============================================================
    #  STEP 7: VISUALIZATION WITH EROSION HEATMAP
    # ============================================================
    def draw_overlay(self, frame, contour_px, params, aligned_contour_mm, erosion_depths):
        display = frame.copy()
        h, w = display.shape[:2]

        # 1. Draw Current Contour (Green)
        if contour_px is not None and len(contour_px) > 5:
            cv2.drawContours(display, [contour_px.astype(np.int32)], -1, (0, 255, 0), 2)

        # 2. Draw Baseline Contour (Blue)
        if self.baseline_contour_mm is not None and self.k is not None:
            baseline_px = (self.baseline_contour_mm / self.k).astype(np.int32)
            cv2.drawContours(display, [baseline_px], -1, (255, 0, 0), 2)

        # 3. Erosion Heatmap Overlay (Red overlay where erosion > 0.5mm)
        if erosion_depths is not None and contour_px is not None and len(contour_px) == len(erosion_depths):
            for i, pt in enumerate(contour_px):
                if erosion_depths[i] > 0.5:  # Erosion > 0.5 mm
                    cv2.circle(display, tuple(pt.astype(np.int32)), 2, (0, 0, 255), -1)
                elif erosion_depths[i] > 0.2:
                    cv2.circle(display, tuple(pt.astype(np.int32)), 1, (0, 255, 255), -1)

        # 4. Display P/B/H and Erosion Stats
        text_lines = []
        if params:
            text_lines.append(f"P: {params['P']:.2f} mm")
            text_lines.append(f"B: {params['B']:.2f} mm")
            text_lines.append(f"H: {params['H']:.2f} mm")
        
        if self.baseline_set and self.baseline_params is not None and params is not None:
            delta_P = self.baseline_params['P'] - params['P']
            delta_B = self.baseline_params['B'] - params['B']
            delta_H = self.baseline_params['H'] - params['H']
            text_lines.append("--- EROSION (Δ) ---")
            text_lines.append(f"ΔP: +{delta_P:.2f}mm" if delta_P > 0 else f"ΔP: {delta_P:.2f}mm")
            text_lines.append(f"ΔB: +{delta_B:.2f}mm" if delta_B > 0 else f"ΔB: {delta_B:.2f}mm")
            text_lines.append(f"ΔH: +{delta_H:.2f}mm" if delta_H > 0 else f"ΔH: {delta_H:.2f}mm")
            if erosion_depths is not None:
                max_erosion = np.max(erosion_depths)
                avg_erosion = np.mean(erosion_depths)
                text_lines.append(f"Max Erosion: {max_erosion:.2f} mm")
                text_lines.append(f"Avg Erosion: {avg_erosion:.2f} mm")
        else:
            text_lines.append("---")
            text_lines.append("Press 'b' to set BASELINE (New wheel)")

        # Draw text
        y = 30
        for line in text_lines:
            color = (255, 255, 255)
            if "EROSION" in line or "Δ" in line:
                color = (0, 255, 255)
            if "Max Erosion" in line:
                val = float(line.split(":")[1].strip().split(" ")[0])
                color = (0, 0, 255) if val > 1.5 else (0, 255, 255)
            cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

        # Legend
        cv2.putText(display, "Green: Current | Blue: Baseline | Red: Erosion >0.5mm", 
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display

    # ============================================================
    #  MAIN VIDEO LOOP
    # ============================================================
    def run(self):
        # --- Initial Scale Calibration ---
        ret, first_frame = self.cap.read()
        if not ret:
            print("❌ Cannot read camera.")
            return
        first_frame = self.undistort(first_frame)
        self.calibrate_scale_accurate(first_frame)

        print("\n" + "="*70)
        print("▶️  EROSION MONITOR STARTED")
        print("   [b] Set Baseline (Healthy Wheel)")
        print("   [q] Quit")
        print("="*70)

        frame_counter = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % FRAME_SKIP != 0:
                display = self.undistort(frame)
                cv2.imshow("Erosion Monitor", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            img_undist = self.undistort(frame)

            # --- Extract Contour ---
            contour_px = self.extract_contour(img_undist)
            
            params = None
            aligned_current_mm = None
            erosion_depths = None

            if contour_px is not None and self.k is not None:
                params = self.calculate_pbh(contour_px)
                
                if self.baseline_set and self.baseline_contour_mm is not None:
                    # ALIGNMENT: Remove camera shift
                    aligned_current_mm = self.align_contours(self.baseline_contour_mm, self.current_contour_mm)
                    # EROSION CALC
                    erosion_depths = self.compute_erosion(self.baseline_contour_mm, aligned_current_mm)

            # --- Display ---
            display = self.draw_overlay(img_undist, contour_px, params, aligned_current_mm, erosion_depths)
            cv2.imshow("Erosion Monitor", display)

            # --- Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                if self.current_contour_mm is not None and params is not None:
                    self.baseline_contour_mm = self.current_contour_mm.copy()
                    self.baseline_params = params.copy()
                    self.baseline_set = True
                    print(f"\n✅ BASELINE SET.")
                    print(f"   P: {params['P']:.2f} | B: {params['B']:.2f} | H: {params['H']:.2f}")
                    print("   Tracking erosion now...")
                else:
                    print("⚠️  No contour detected.")

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Erosion monitoring stopped.")


# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    profiler = ErosionProfiler()
    profiler.run()