#!/bin/bash

# Check for URL argument
if [ -z "$1" ]; then
    echo "❌ Please provide a YouTube URL."
    echo "Usage: ./yt_qt_download.sh https://www.youtube.com/watch?v=XXXXX"
    exit 1
fi

URL=$1

echo "📥 Downloading video from: $URL"

yt-dlp \
    -f 'bv*[vcodec~="avc1"]+ba[acodec~="mp4a"]/b[ext=mp4]' \
    --merge-output-format mp4 \
    --remux-video mp4 \
    --postprocessor-args ffmpeg:"-movflags +faststart" \
    "$URL"

echo "✅ Done! Video is QuickTime compatible."
