# data/

Sample clips and intermediate artifacts go here. Everything in this directory
is gitignored except this README.

## Suggested first clip

Pull a short clip from **SoccerNet-Tracking** (https://soccernet.org). The
dataset requires accepting terms; the official Python helper is on PyPI
(`SoccerNet`). For v1 we just need ~30 seconds.

```bash
# Example, after registering and getting a password from the SoccerNet team:
pip install SoccerNet
python -c "from SoccerNet.Downloader import SoccerNetDownloader as S; \
           d = S(LocalDirectory='./data/soccernet'); \
           d.password = 'YOUR_PASSWORD'; \
           d.downloadDataTask(task='tracking', split=['test'])"
```

Then trim a 30s segment with ffmpeg:

```bash
ffmpeg -i data/soccernet/.../some_clip.mkv -t 30 -c copy data/clip30.mp4
```

## Alternate sources

- SoccerTrack (top-down/tactical multi-camera) — github.com/AtomScott/SoccerTrack
- Roboflow Universe — short, easy, less rigorous
- SkillCorner Broadcast Tracking — academic broadcast benchmark
