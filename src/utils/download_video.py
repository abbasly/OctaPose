import sys
import os
from yt_dlp import YoutubeDL

def download_video(url, out_dir="../data/videos"):
    os.makedirs(out_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo+bestaudio',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(out_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'noplaylist': True,
        'progress_hooks': [hook],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        print_metadata(info)

def hook(d):
    if d['status'] == 'finished':
        print(f"✅ Download complete: {d['filename']}")

def print_metadata(info):
    print("\n📄 Video Metadata:")
    print(f"Title     : {info.get('title')}")
    print(f"Uploader  : {info.get('uploader')}")
    print(f"Duration  : {info.get('duration')} seconds")
    print(f"Upload date : {info.get('upload_date')}")
    print(f"View count : {info.get('view_count')}")
    print(f"Filepath  : {info.get('_filename')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Usage: python download_ufc.py <youtube_url> [output_dir]")
        sys.exit(1)

    url = sys.argv[1]

    download_video(url)
