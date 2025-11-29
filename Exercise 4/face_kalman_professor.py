import dearpygui.dearpygui as dpg
import sys
import select
import time
import math
import sys
import cv2
import os
import pickle
import numpy as np

# Changed from linkalman_solution to linkalman to use our implementation
from linkalman import LinKalman

from DataPlot import DataPlot

# FIX: Use absolute path
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'calibration_data.pkl')
FACE_WIDTH_MM = 140  # Average width of a human face in mm
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# FIX: Add toggle for calibration to solve visibility issue
USE_CALIBRATION = False

class FaceGui:
	def __init__(self):
		# Create all data plots to hold 1000 points before scrolling
		self.xyzplot = DataPlot(("x", "y", "z", "x_kalman", "y_kalman", "z_kalman"), 1000)

		# Initialize face detector
		cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
		self.faceCascade = cv2.CascadeClassifier(cascPathface)
		self.video_capture = cv2.VideoCapture(0)
		self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		print(f"Video resolution: {self.width}x{self.height}")

		# Load calibration data and initialize undistortion maps
		if USE_CALIBRATION:
			with open(CALIBRATION_FILE, 'rb') as f:
				calibration_data = pickle.load(f)

			mtx = calibration_data['camera_matrix']
			dist = calibration_data['distortion_coefficients']
			print(f"Camera matrix:\n{mtx}")
			print(f"Distortion coefficients:\n{dist}")
			self.K, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1, (self.width, self.height))
			self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, self.K, (self.width, self.height), 5)
		else:
			# Approximate K for standard webcam if no calibration
			self.K = np.array([[self.width, 0, self.width/2],
							   [0, self.width, self.height/2],
							   [0, 0, 1]])
			self.mapx = None
			self.mapy = None

	def createWindow(self):
		with dpg.window(tag="Status"):
			with dpg.group(horizontal=True):
				self.xyzplot.createGUI(-1, -1)

	def processFace(self, K, u0, v0, u1, v1):
		# K is camera matrix
		# (u0,v0) is top-left of face bounding box
		# (u1,v1) is bottom-right of face bounding box
		# Returns (x,y,z) in mm of face center in camera coordinates

		f0 = K[0,0]  # Focal length in pixels
		f1 = K[1,1]
		cu = K[0,2]  # Principal point in pixels
		cv = K[1,2]
		# To centered coordinates
		u0 = u0 - cu
		u1 = u1 - cu
		v0 = v0 - cv
		v1 = v1 - cv
		# Calculate angles and derive distance
		angle1 = math.atan2(u0, f0)
		angle2 = math.atan2(u1, f0)
		z = FACE_WIDTH_MM / (math.sin(angle2) - math.sin(angle1))
		# Calculate x,y in mm
		u = (u0 + u1) / 2
		v = (v0 + v1) / 2
		x = (z/f0) * u
		y = -(z/f1) * v # y axis is inverted in image coordinates
		return (x, y, z)

	def project3DFaceto2D(self, x, y, z):
		if z == 0:
			return 0, 0, 0, 0
		u0 = ((x-FACE_WIDTH_MM/2)/z)*self.K[0][0] + self.K[0][2]
		v0 = -((y-FACE_WIDTH_MM/2)/z)*self.K[1][1] + self.K[1][2]
		u1 = ((x+FACE_WIDTH_MM/2)/z)*self.K[0][0] + self.K[0][2]
		v1 = -((y+FACE_WIDTH_MM/2)/z)*self.K[1][1] + self.K[1][2]
		return int(u0), int(v0), int(u1), int(v1)
		
	def draw3DFace(self, frame, x, y, z, color):
		try:
			u0, v0, u1, v1 = self.project3DFaceto2D(x, y, z)
		except OverflowError:
			return
		cv2.rectangle(frame, (u0, v0), (u1, v1), color, 2)
		
	def createKalman(self):
		# Professor's parameters
		x = np.array([0, 0, 0])
		A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		# Note: R has higher noise for Z (10) than X,Y (1)
		R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 10]])
		Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.2]])
		self.kalman = LinKalman(x, A, H, R, Q)
		
	def stepKalman(self, frame, xmm, ymm, zmm):
		xhat, Phat = self.kalman.predictState()
		self.draw3DFace(frame, xhat[0], xhat[1], xhat[2], (0, 255, 0))
		z = np.array([xmm, ymm, zmm])
		x, P = self.kalman.update(z)
		self.draw3DFace(frame, x[0], x[1], x[2], (0, 0, 255))
		return x[0], x[1], x[2]
	
	def largest(self, faces):
		largest = faces[0]
		for face in faces[1:]:
			if largest[2]*largest[3] < face[2]*face[3]:
				largest = face
		return largest
		
	def run(self):
		dpg.create_context()
		dpg.create_viewport()
		self.createWindow()
		dpg.setup_dearpygui()
		dpg.show_viewport()
		dpg.set_primary_window("Status", True)
		frameno = 0
		
		self.createKalman()

		while dpg.is_dearpygui_running():
			# Read frame from camera
			ret, frame = self.video_capture.read()
			if not ret:
				print("Failed to grab frame")
				break

			# Resize frame to target resolution
			frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

			# Undistort frame
			if USE_CALIBRATION:
				frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

			# Convert to greyscale and detect faces
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
			
			if len(faces) > 0:
				(x, y, w, h) = self.largest(faces)
				xmm, ymm, zmm = self.processFace(self.K, x, y, x+w, y+h)
				self.draw3DFace(frame, xmm, ymm, zmm, (255, 0, 0))
				x_k, y_k, z_k = self.stepKalman(frame, xmm, ymm, zmm)

				self.xyzplot.addDataVector(frameno, (xmm, ymm, zmm, x_k, y_k, z_k))
				cv2.putText(frame, f"x={xmm:.0f}mm y={ymm:.0f}mm z={zmm:.0f}mm", (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
				frameno = frameno + 1
			cv2.imshow('Video', frame)

			dpg.render_dearpygui_frame()
		self.video_capture.release()
		cv2.destroyAllWindows()
		dpg.destroy_context()


if __name__ == "__main__":
	gui = FaceGui()
	gui.run()
