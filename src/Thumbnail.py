class Thumbnail:
	def __init__(self, youtube_url):
		self.youtube_url = youtube_url
		self.thumbnail_path = None
		self.depth_map = None
		self.segmented_planes = None
		self.segmented_planes_path = []

	def fetch_thumbnail(self):
		pass

	def get_depth_map(self):
		pass

	def segment_planes(self):
		pass

	def save_segmented_planes(self):
		pass

	def extract_planes(self):
		pass

from thumbnail_methods.fetch_thumbnail import fetch_thumbnail
from thumbnail_methods.extract_planes import extract_planes, get_depth_map, segment_planes, save_segmented_planes

Thumbnail.fetch_thumbnail = fetch_thumbnail
Thumbnail.get_depth_map = get_depth_map
Thumbnail.segment_planes = segment_planes
Thumbnail.save_segmented_planes = save_segmented_planes
Thumbnail.extract_planes = extract_planes