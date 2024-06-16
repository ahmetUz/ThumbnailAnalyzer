import torch
import cv2
import numpy as np
import os

def get_depth_map(self):
	# download the model
	model_type = "DPT_Large"
	midas = torch.hub.load("intel-isl/MiDaS", model_type)

	# load the model on the device
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	midas.to(device).eval()

	# load transforms for the model
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

	transform = midas_transforms.dpt_transform

	img = cv2.imread(self.thumbnail_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	input_batch = transform(img).to(device)
	
	with torch.no_grad():
		prediction = midas(input_batch)
		prediction = torch.nn.functional.interpolate(
			prediction.unsqueeze(1),
			size=img.shape[:2],
			mode="bicubic",
			align_corners=False,
		).squeeze()
	
	self.depth_map = prediction.cpu().numpy()

def segment_planes(self, num_planes=3):
	# Normalize the depth map
	depth_map_normalized = (self.depth_map - np.min(self.depth_map)) / (np.max(self.depth_map) - np.min(self.depth_map))

	# Create thresholds to segment plans
	thresholds = np.linspace(0, 1, num_planes + 1)

	self.segmented_planes = np.digitize(depth_map_normalized, bins=thresholds) - 1

def save_segmented_planes(self):
	img = cv2.imread(self.thumbnail_path)
	dir = os.path.dirname(self.thumbnail_path)
	
	for plane in np.unique(self.segmented_planes):
		plane_mask = self.segmented_planes == plane
		segmented_image = np.zeros_like(img)
		segmented_image[plane_mask] = img[plane_mask]

		output_filename = f"{dir}/plane_{plane}.png"
		cv2.imwrite(output_filename, segmented_image)
		print(f"Saved {output_filename}")

def extract_planes(self):
	self.get_depth_map()
	self.segment_planes()
	self.save_segmented_planes()