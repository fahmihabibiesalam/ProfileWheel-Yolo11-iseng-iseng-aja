# wheel_measurement_pro.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

class RealWheelProfiler:
    def __init__(self, calib_file="calibration.npz"):
        # Load camera calibration
        if os.path.exists(calib_file):
            data = np.load(calib_file)
            self.mtx = data['mtx']
            self.dist = data['dist']
            print("✅ Calibration loaded.")
        else:
            print("⚠️  No calibration file found. Running without distortion correction.")
            self.mtx = None
            self.dist = None

        # Connect to iVCam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open iVCam. Make sure the app is running.")
        
        print(f"📷 iVCam connected. Resolution: {int(self.cap.get(3))}x{int(self.cap.get(4))}")
        
        # Scale factor (mm per pixel)
        self.scale_k = None
        self.last_contour_mm = None
        self.roi = None  # Region of Interest
        self.contour_cache = None

    def undistort(self, img):
        if self.mtx is not None and self.dist is not None:
            h, w = img.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img, self.mtx, self.dist, None, new_mtx)
            x, y, w, h = roi
            return dst[y:y+h, x:x+w]
        return img

    def select_roi(self, frame):
        """Let user draw a rectangle around the wheel."""
        print("📐 Please draw a rectangle around the wheel profile. Press SPACE or ENTER to confirm.")
        roi = cv2.selectROI("Select Wheel ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Wheel ROI")
        if roi[2] > 10 and roi[3] > 10:
            self.roi = roi
            print(f"✅ ROI selected: {roi}")
        else:
            print("⚠️  ROI selection cancelled. Using full frame.")

    def set_scale_from_ruler(self, frame):
        """Click two points on a ruler in the image to set mm/px scale."""
        print("\n📏 SCALE CALIBRATION MODE")
        print("   1. Click on the 0mm mark of the ruler.")
        print("   2. Click on the 100mm (or known distance) mark.")
        print("   3. Press ENTER after clicking both points.")
        print("   Press ESC to cancel.")
        
        img_disp = frame.copy()
        points = []
        
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 2:
                    points.append((x, y))
                    cv2.circle(img_disp, (x, y), 6, (0, 255, 255), -1)
                    cv2.putText(img_disp, f"P{len(points)}", (x+10, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv2.imshow("Set Scale", img_disp)
        
        cv2.imshow("Set Scale", img_disp)
        cv2.setMouseCallback("Set Scale", mouse_cb)
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow("Set Scale")
                return False
            if key == 13 and len(points) == 2:  # ENTER
                break
        
        cv2.destroyWindow("Set Scale")
        
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        px_dist = np.sqrt(dx**2 + dy**2)
        
        print(f"📏 Pixel distance: {px_dist:.2f} px")
        try:
            real_mm = float(input("Enter the real distance between these two points (in mm): "))
            self.scale_k = real_mm / px_dist
            print(f"✅ Scale set: 1 pixel = {self.scale_k:.4f} mm")
            return True
        except ValueError:
            print("❌ Invalid input.")
            return False

    def extract_profile(self, img, low_thresh=50, high_thresh=150):
        """
        Extract the detailed wheel profile using Canny edge detection.
        Returns the contour points and the edge mask.
        """
        # Apply ROI if selected
        if self.roi is not None:
            x, y, w, h = self.roi
            roi_img = img[y:y+h, x:x+w]
        else:
            roi_img = img
            x, y = 0, 0

        # Preprocess
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # Morphological closing to connect broken edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None, edges, roi_img
        
        # Filter by area (remove tiny noise)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
        if not valid_contours:
            return None, edges, roi_img
        
        # Take the largest contour (the wheel profile)
        largest = max(valid_contours, key=cv2.contourArea)
        
        # Smooth the contour slightly using approxPolyDP
        epsilon = 0.002 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        # Shift coordinates back to original image space if ROI was used
        if self.roi is not None:
            approx[:, 0, 0] += x
            approx[:, 0, 1] += y
        
        return approx.squeeze(), edges, roi_img

    def measure_and_visualize(self, frame):
        """Main processing pipeline with interactive trackbars."""
        frame_undist = self.undistort(frame)
        
        # Create a window with trackbars for Canny thresholds
        cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Low", "Tuning", 50, 200, lambda x: None)
        cv2.createTrackbar("High", "Tuning", 150, 300, lambda x: None)
        
        print("\n🔧 Tuning Mode. Adjust sliders until the wheel profile is clearly outlined.")
        print("   Press 'q' to proceed with current settings.")
        print("   Press 'r' to select ROI again.")
        print("   Press 's' to calibrate scale.")
        print("   Press 'c' to export contour CSV.")
        
        while True:
            # Get current trackbar values
            low = cv2.getTrackbarPos("Low", "Tuning")
            high = cv2.getTrackbarPos("High", "Tuning")
            
            # Extract profile
            contour, edges, roi = self.extract_profile(frame_undist, low, high)
            
            # Display
            display = frame_undist.copy()
            if contour is not None and len(contour) > 10:
                cv2.drawContours(display, [contour.astype(np.int32)], -1, (0, 255, 0), 2)
                self.contour_cache = contour
                info = f"Points: {len(contour)}"
                if self.scale_k is not None:
                    # Show approximate diameter in mm (if contour is closed)
                    area = cv2.contourArea(contour.astype(np.int32))
                    if area > 0:
                        equiv_diam = np.sqrt(4 * area / np.pi) * self.scale_k
                        info += f" | Est. Diameter: {equiv_diam:.1f} mm"
                cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            # Show ROI
            if self.roi is not None:
                x, y, w, h = self.roi
                cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.imshow("Tuning", display)
            cv2.imshow("Edges", edges)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.roi = None
                self.select_roi(frame_undist)
            elif key == ord('s'):
                if self.set_scale_from_ruler(frame_undist):
                    print("✅ Scale calibration successful!")
            elif key == ord('c'):
                if self.contour_cache is not None and self.scale_k is not None:
                    self.save_contour()
                elif self.contour_cache is None:
                    print("⚠️  No contour detected. Adjust Canny thresholds.")
                else:
                    print("⚠️  Please calibrate scale first (press 's').")
        
        cv2.destroyAllWindows()
        return self.contour_cache

    def save_contour(self):
        """Export the contour in millimeters as CSV."""
        if self.contour_cache is None or self.scale_k is None:
            return
        
        contour_mm = self.contour_cache * self.scale_k
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wheel_profile_{timestamp}.csv"
        
        # Sort points to be continuous (clockwise) for proper shape plotting
        # Simple sorting by angle around centroid
        cx, cy = np.mean(contour_mm, axis=0)
        angles = np.arctan2(contour_mm[:, 1] - cy, contour_mm[:, 0] - cx)
        sorted_idx = np.argsort(angles)
        sorted_contour = contour_mm[sorted_idx]
        
        np.savetxt(filename, sorted_contour, delimiter=',', 
                   header='x_mm,y_mm', comments='')
        print(f"💾 Profile saved to '{filename}' with {len(sorted_contour)} points.")
        
        # Plot and show the profile
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_contour[:, 0], sorted_contour[:, 1], 'b-', linewidth=1.5)
        plt.title(f"Wheel Profile - {timestamp}")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(f"wheel_profile_{timestamp}.png")
        plt.show()

    def run(self):
        """Main loop."""
        print("\n" + "="*50)
        print("🚀 REAL WHEEL PROFILE MEASUREMENT")
        print("="*50)
        
        # Capture a single high-quality frame (you can switch to video later)
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            return
        
        # Optionally select ROI
        self.select_roi(frame)
        
        # Start measurement with trackbars
        self.measure_and_visualize(frame)
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Measurement finished.")

if __name__ == "__main__":
    profiler = RealWheelProfiler("calibration.npz")
    profiler.run()