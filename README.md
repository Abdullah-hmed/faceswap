# Faceswap

A command line tool to perform face swaps on images and videos.

## Requirements

- Docker

## Installation

#### Step 1: Install Requirements

Make sure you have:

* **[Docker](https://docs.docker.com/get-docker/)** installed
* A **CUDA-capable NVIDIA GPU**
* **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)** installed (to enable GPU access inside containers)

Verify it works:

```bash
docker --version
nvidia-smi
```

#### Step 2: Clone the Repository

```bash
git clone https://github.com/Abdullah-hmed/faceswap.git
cd faceswap
```

#### Step 3: Build the Docker Image

```bash
docker build -t faceswap-app .
```

#### Step 4: Run the Tool

A simple face swap can be done by running the command below in any directory that contains your image and video:

```bash
docker run --rm --gpus all -v "${PWD}:/input" faceswap-app main.py -f /input/image.jpeg -m /input/video.mp4 -o /input/output.mp4
```
 > **Note:** When using Docker, all input/output file paths must be relative to the /input directory, since your current folder is mounted there inside the container


## Usage

#### 📷 Example 1: Image + Video Face Swap

**Linux/macOS (bash):**

```bash
docker run --rm --gpus all -v "${PWD}:/input" faceswap-app main.py \
  -f /input/image.jpeg \
  -m /input/video.mp4 \
  -o /input/output.mp4
```

**Windows PowerShell:**

```powershell
docker run --rm --gpus all -v "${PWD}:/input" faceswap-app main.py `
  -f /input/image.jpeg `
  -m /input/video.mp4 `
  -o /input/output.mp4
```

> **Note:** All file paths must be relative to `/input` because your current directory is mounted into the container at `/input`.

---

#### 🎯 Example 2: Choose a Specific Face to Swap

**Linux/macOS (bash):**

```bash
docker run --rm -it --gpus all -v "${PWD}:/input" faceswap-app main.py \
  -f /input/image.jpg \
  -m /input/video.mp4 \
  -o /input/output.mp4 \
  --choose-face
```

**Windows PowerShell:**

```powershell
docker run --rm -it --gpus all -v "${PWD}:/input" faceswap-app main.py `
  -f /input/image.jpg `
  -m /input/video.mp4 `
  -o /input/output.mp4 `
  --choose-face
```

> **Note:** The `-it` flag is essential when using `--choose-face`, as it enables interactive prompts for face selection.

---

#### 🎥 Example 3: Run Live Webcam Face Swap

**Linux/macOS (bash):**

```bash
docker run --rm -it --gpus all -v "${PWD}/faces:/app/faces" -p 3000:3000 faceswap-app \
  webcam/webcam.py
```

**Windows PowerShell:**

```powershell
docker run --rm -it --gpus all -v "${PWD}\faces:/app/faces" -p 3000:3000 faceswap-app `
  webcam/webcam.py
```

> **Note:** Mount a folder containing your face images to `/app/faces`. The Flask server will be available at [http://localhost:3000](http://localhost:3000).

## License

[MIT](LICENSE)