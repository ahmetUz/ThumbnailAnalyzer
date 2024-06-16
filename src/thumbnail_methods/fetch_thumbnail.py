import urllib.parse
import requests
import os

def fetch_thumbnail(self, output_folder='../thumbnails'):
	url_data = urllib.parse.urlparse(self.youtube_url)
	query = urllib.parse.parse_qs(url_data.query)
	video_id = query.get('v')

	if not video_id:
		if "youtu.be" in self.youtube_url:
			video_id = url_data.path.lstrip('/')
		else:
			raise ValueError("Invalid YouTube URL")
	else:
		video_id = video_id[0]

	thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
	response = requests.get(thumbnail_url)
	if response.status_code == 200:
		os.makedirs(f"{output_folder}/{video_id}", exist_ok=True)
		output_path = os.path.join(output_folder, f"{video_id}/thumbnail.jpg")
		with open(output_path, "wb") as file:
			file.write(response.content)
		self.thumbnail_path = output_path
		print(f"Thumbnail saved as {output_path}")
	else:
		ValueError("Invalid YouTube URL")
