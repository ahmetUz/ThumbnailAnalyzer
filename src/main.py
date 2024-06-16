import argparse
from Thumbnail import Thumbnail

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process a YouTube thumbnail.')
	parser.add_argument('youtube_url', type=str, help='URL of the YouTube video')
	args = parser.parse_args()

	thumbnail = Thumbnail(args.youtube_url)
	thumbnail.fetch_thumbnail()
	thumbnail.extract_planes()
