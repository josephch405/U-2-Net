ffmpeg -i data/test_videos/v0.mp4 -q:v 5 data/workbench/img%04d.jpg -hide_banner
python u2net_test_video.py
ffmpeg -i img%04d.jpg -vf fps=30 -pix_fmt yuv420p output.mp4