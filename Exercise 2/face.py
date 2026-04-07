import dearpygui.dearpygui as dpg
import sys
import time
import math
import cv2
import os
import pickle
import numpy as np

# YOUR JOB: Implement processFace to compute x,y,z from face bounding box

from DataPlot import DataPlot
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'calibration_data.pkl')
FACE_WIDTH_MM = 140  # Average width of a human face in mm

class FaceGui:
	def __init__(self):
		# Create all data plots to hold 1000 points before scrolling
		self.xyzplot = DataPlot(("x", "y", "z"), 1000)

		# Initialize face detector
		from ultralytics import YOLO
		self.model = YOLO("yolo26n.pt")
		self.video_capture = cv2.VideoCapture(0)
		self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		print(f"Video resolution: {self.width}x{self.height}")

		# Load calibration data and initialize undistortion maps
		with open(CALIBRATION_FILE, 'rb') as f:
			calibration_data = pickle.load(f)

		mtx = calibration_data['camera_matrix']
		dist = calibration_data['distortion_coefficients']
		print(f"Camera matrix:\n{mtx}")
		print(f"Distortion coefficients:\n{dist}")
		self.mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1, (self.width, self.height))
		self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, self.mtx, (self.width, self.height), 5)    

	def createWindow(self):
		with dpg.window(tag="Status"):
			with dpg.group(horizontal=True):
				self.xyzplot.createGUI(-1, -1)

	def processFace(self, K, u0, v0, u1, v1):
		# K is camera matrix
		# (u0,v0) is top-left of face bounding box
		# (u1,v1) is bottom-right of face bounding box
		# Returns (x,y,z) in mm of face center in camera coordinates

		# Validate K is a 3x3 matrix
		K = np.array(K)
		if K.shape != (3, 3):
			return (0, 0, 0)

		fx = K[0, 0]
		fy = K[1, 1]
		cx = K[0, 2]
		cy = K[1, 2]

		# Compute horizontal angles to left and right edges of bounding box
		angle_left  = math.atan2(u0 - cx, fx)
		angle_right = math.atan2(u1 - cx, fx)

		# Depth from known face width and angular spread
		denom = math.sin(angle_right) - math.sin(angle_left)
		if abs(denom) < 1e-8:
			return (0, 0, 0)
		z_mm = FACE_WIDTH_MM / denom

		# Center pixel of bounding box
		u_c = (u0 + u1) / 2.0
		v_c = (v0 + v1) / 2.0

		# Back-project center pixel to camera coordinates
		x_mm =  (z_mm / fx) * (u_c - cx)
		y_mm = -(z_mm / fy) * (v_c - cy)

		return (float(x_mm), float(y_mm), float(z_mm))

	def run(self):
		dpg.create_context()
		dpg.create_viewport()
		self.createWindow()
		dpg.setup_dearpygui()
		dpg.show_viewport()
		dpg.set_primary_window("Status", True)
		frameno = 0

		while dpg.is_dearpygui_running():
			# Read frame from camera
			ret, frame = self.video_capture.read()
			if not ret:
				print("Failed to grab frame")
				break
			# Undistort frame
			frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
			# Detect faces with YOLO
			results = self.model(frame, classes=[0], verbose=False)
			
			if len(results[0].boxes) > 0:
				box = results[0].boxes[0]
				if box.conf[0] > 0.5:
					x1, y1, x2, y2 = map(int, box.xyxy[0])
					cv2.rectangle(frame, (x1, y1), (x2, y2),(255,0,0), 2)
					u0 = x1
					v0 = y1
					u1 = x2
					v1 = y2
				(xmm, ymm, zmm) = self.processFace(self.mtx, u0, v0, u1, v1)
				self.xyzplot.addDataVector(frameno, (xmm, ymm, zmm))
				cv2.putText(frame, f"x={xmm:.0f}mm y={ymm:.0f}mm z={zmm:.0f}mm", (u0, v0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
				frameno = frameno + 1
			cv2.imshow('Video', frame)

			dpg.render_dearpygui_frame()
		video_capture.release()
		cv2.destroyAllWindows()
		dpg.destroy_context()


if __name__ == "__main__":
	gui = FaceGui()
	gui.run()