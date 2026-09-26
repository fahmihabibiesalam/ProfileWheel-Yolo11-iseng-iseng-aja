wheel_profiler_erosion.py

📖 Detailed Explanation of Steps (In English)
Step 1: Accurate Multi-Segment Scale Calibration
Why it matters: Clicking one segment is inaccurate.
What it does: You click 3 different known distances on a ruler placed flush against the wheel. The script calculates k = real_mm / pixel_distance for each segment and takes the average. This eliminates manual clicking errors and parallax.

Step 2: Camera Undistortion
Uses the calibration.npz from your earlier chessboard calibration to remove lens distortion, ensuring the pixels represent true straight lines.

Step 3: Contour Extraction (YOLO + Morphology)
Attempts YOLO segmentation first.

If YOLO fails (e.g., bad lighting), it falls back to a robust adaptive Canny edge detector that works well in varying light conditions.
Applies Closing (fill holes) and Opening (remove noise) to clean the mask before extracting the final contour.

Step 4: P/B/H Calculation
P (Flange Height): Vertical distance from the lowest point (flange tip) to the tread line.
B (Flange Thickness): Horizontal thickness measured exactly at 12mm above the tread line (standard UIC/AAR specification).
H (Rim Width): Horizontal width of the top tread surface.

Step 5: Contour Alignment (The "Accuracy" Enabler)
The wheel might shift slightly in the video frame due to vibration or camera movement.
The script calculates the geometric center (centroid of the bounding box) of the baseline profile and the current profile.
It translates the current contour so its center perfectly matches the baseline's center.
This ensures we are comparing the exact same physical location on the wheel, removing false wear readings caused by shifts.

Step 6: Erosion Depth Calculation (Per-Point)
After alignment, the script builds a KD-Tree from the baseline contour points.
For every single point on the current contour, it finds the nearest point on the baseline.
The Euclidean distance (in mm) between these points is the erosion depth at that location.

Interpretation:
0.0 mm = No wear.
1.2 mm = The wheel has eroded (worn down) by 1.2 mm at that specific point.

Step 7: Real-Time Visualization (Heatmap Overlay)
Green Contour: Current wheel profile.
Blue Contour: Baseline (healthy) wheel profile.
Red Dots: Points where erosion is > 0.5 mm (severe wear highlighted).
Yellow Dots: Points where erosion is between 0.2 mm and 0.5 mm (moderate wear).

Text Overlay: Shows current P/B/H, along with ΔP, ΔB, ΔH (the total erosion in mm for those specific parameters), plus the Maximum and Average erosion depth across the entire profile.

🧪 How to Run It
Install scipy (needed for the KD-Tree):

python wheel_profiler_erosion.py
Calibrate Scale: When prompted, click 3 segments on a ruler, type the real distances (e.g., 50mm, 100mm, 50mm). Press ENTER after each.

Set Baseline: Point the camera at a brand new (healthy) wheel. Press b. The Blue contour appears.

Measure Erosion: Now replace the wheel with a worn wheel (or shift the camera to a worn section). The Green contour appears. The script instantly shows:

The Red/Yellow dots highlighting the worn areas.

The numerical erosion values (ΔP, ΔB, ΔH, Max Erosion, Avg Erosion).
