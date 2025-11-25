# Objaverse Rendering

Scripts to perform distributed rendering of Objaverse objects in Blender across many GPUs and processes.

### System requirements

We have only tested the rendering scripts on Ubuntu machines that have NVIDIA GPUs.

If you run into any issues, please open an issue! :)

### Installation

1. Install Blender

```bash
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz
rm blender-3.2.2-linux-x64.tar.xz
```

2. Update certificates for Blender to download URLs

```bash
# this is needed to download urls in blender
# https://github.com/python-poetry/poetry/issues/5117#issuecomment-1058747106
sudo update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs
```

3. Install Python dependencies

```bash
pip install -r requirements.txt
```

4. (Optional) If you are running rendering on a headless machine, you will need to start an xserver. To do this, run:

```bash
sudo apt-get install xserver-xorg
sudo python3 scripts/start_xserver.py start
```

### Rendering

1. Filter the objets:
   ```
   python3 filter_objaverse.py --sample_size 15000 --output_file ../filtered_15k.json
   ```

2. Download the objects:

```
python3 download_objaverse.py --filtered_file ../filtered_15k.json --output_name mvd_15k --batch_size 500
```
3. Run:
   ```
   ./progress_tracker.sh
   ```


