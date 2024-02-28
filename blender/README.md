# Rendering Scenes by Blender

We provide the scene visualization script by [Blender](https://www.blender.org/) in `src/utils/blender_script.py` and `src/utils/visualize.py`.
They are used to provide ground-truth images to compute FID/KID/... scores and draw demonstrations in the paper.

Download the Blender software from the [official website](https://www.blender.org/download/lts/3-3/), and put the uncompressed folder here.
Or you can modify the `_blender_binary_path()` in `utils/visualize.py` to the path of your Blender software.
You can also provide the path to the Blender software by setting the environment variable `BLENDER_PATH`.

The version used in this project is `3.3.1`.

```bash
cd blender
wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz
tar -xvf blender-3.3.1-linux-x64.tar.xz
rm blender-3.3.1-linux-x64.tar.xz
```
