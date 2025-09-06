# Cytologick - Research Tool for Pap Smear Analysis

## What is Cytologick?

Cytologick is a Python-based research tool for analyzing Pap smear slides using artificial intelligence. This software is designed for research and educational purposes to explore automated detection of cellular abnormalities like LSIL, HSIL, and ASCUS in cytological samples.

PyTorch is the recommended framework for training and local inference in this repository. The PyTorch pipeline includes mixed precision (AMP), a cosine LR scheduler, tqdm progress bars, and per‑epoch checkpoints.

## What it does

- Loads MRXS slide files
- Trains U-Net segmentation models on annotated cell data
- Supports both TensorFlow and PyTorch frameworks (PyTorch recommended)
- Provides Qt desktop interface for viewing slides
- Runs inference on slide regions
- Works with ASAP annotation files
- Detects LSIL, HSIL, ASCUS, and ASCH cell patterns  

![Cytologick Example](./assets/example.jpg)

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) package manager
- [OpenSlide](https://openslide.org) for reading medical slide files

### Step 1: Create Python Environment

```bash
conda create -n cytologick python=3.10
conda activate cytologick
```

### Step 2: Install OpenSlide

**Windows:**

```bash
# Download from https://openslide.org/download/
# Extract to a folder and note the path
```

**Linux/macOS:**

```bash
# Follow OS-specific instructions at https://openslide.org/download/
```

### Step 3: Install ASAP (Optional - for training only)

Download and install [ASAP](https://computationalpathologygroup.github.io/ASAP/) if you plan to create training datasets from annotated slides.

### Step 4: Install Dependencies

## Option A: PyTorch Only (Recommended)

```bash
pip install -r requirements-pytorch.txt
conda install openslide-python
```

## Option B: Both Frameworks (Full Support)

```bash
pip install -r requirements.txt
conda install openslide-python
```

## Option C: TensorFlow Only (Optional)

```bash
pip install -r requirements-tensorflow.txt
conda install openslide-python
```

**Notes:**

- Both frameworks can be installed together
- NumPy version constraint (`<2.0.0`) required for TensorFlow compatibility
- Framework selection done via `config.yaml` (PyTorch recommended)

### Step 5: Download Web Dependencies (Optional: for run_web.py)

Download and place these files in `__web/static/`:

- [jquery.js](https://jquery.com)
- [openseadragon.js](https://openseadragon.github.io)
- openseadragon-scalebar.js

### Step 6: Configure Settings

Edit `config.py` or create `config.yaml` to set your paths:

**Required Settings:**

```python
OPENSLIDE_PATH = "C:/path/to/openslide/bin"  # Windows
SLIDE_DIR = "./current"  # Folder containing your MRXS files
```

**Optional Settings:**

```python
FRAMEWORK = "pytorch"  # Choose "pytorch" (recommended) or "tensorflow"
IP_EXPOSED = "127.0.0.1"  # Web interface IP
UNET_PRED_MODE = "remote"  # Use cloud models
```

**Or use YAML config (config.yaml) — recommended:**

```yaml
neural_network:
  framework: pytorch  # Options: 'pytorch' (recommended), 'tensorflow'
general:
  openslide_path: C:/path/to/openslide/bin
gui:
  slide_dir: ./current

### Minimal `config.yaml`

Copy-paste and edit paths to get started quickly (PyTorch recommended):

```yaml
general:
  openslide_path: /path/to/openslide/bin
  hdd_slides: /path/to/your/slides          # Folder with .mrxs files
  temp_folder: temp

neural_network:
  framework: pytorch
  dataset_folder: dataset
  masks_folder: masks
  images_folder: rois
  classes: 3
  labels:               # Map your annotation labels to class index 2 (foreground)
    LSIL: 2
    HSIL: 2
    ASCUS: 2

gui:
  slide_dir: ./current

dataset:
  exclude_duplicates: false
  broaden_individual_rect: 1000
```

## What You Need

- **Slide Files**: MRXS format Pap smear slides
- **Python Environment**: Python 3.10.6 or later with Conda
- **[OpenSlide](https://openslide.org/)**: For reading medical slide files
- **[ASAP](https://computationalpathologygroup.github.io/ASAP/)**: For slide annotation (optional, for training)
- **Internet Connection**: For cloud-based AI analysis (recommended)

## Quick Start

### 1. Run Desktop Application (Recommended)

```bash
# Activate your environment
conda activate cytologick

# Run the application
python run.py
```

Opens a desktop research application for experimental slide analysis.

### 2. Web Interface (Experimental)

```bash
python run_web.py
```

⚠️ **Note**: Web interface is experimental and under development.

## Getting Started

### 1. Prepare Your Files

- Place your MRXS slide files in the `current/` folder
- Make sure OpenSlide path is configured in `config.py`

### 2. Start Analysis

```bash
# Activate your environment
conda activate cytologick

# Run desktop application (recommended)
python run.py
```

### 3. Using the Interface

1. **Load Slide**: Select your MRXS file from the file browser
2. **Run Analysis**: Click analyze to run experimental detection
3. **View Results**: Areas of interest will be highlighted with annotations
4. **Export Results**: Save analysis results for research purposes

### Advanced: Training Your Own Models (PyTorch)

#### Step 1: Prepare Your Annotated Data

1. **Annotate slides** in [ASAP](https://computationalpathologygroup.github.io/ASAP/)
   - Draw rectangles around abnormal cells
   - Label each rectangle (LSIL, HSIL, ASCUS, etc.)
   - Save annotation files as XML

2. **Organize your files:**

   ``` bash
   your_data_folder/
   ├── slide1.mrxs
   ├── slide1.xml
   ├── slide2.mrxs
   ├── slide2.xml
   └── ...
   ```

#### Step 2: Configure Training Paths

Edit `config.py` to point to your data:

```python
HDD_SLIDES = "/path/to/your_data_folder"
DATASET_FOLDER = "dataset"
```

#### Step 3: Choose Your Framework

Set your preferred framework in `config.yaml`:

```yaml
neural_network:
  framework: pytorch  # or "tensorflow"
```

#### Step 4: Run Training Pipeline (PyTorch)

```bash
# Extract annotations from ASAP XML files
python get_xmls.py

# Create training dataset from extracted annotations
python get_dataset.py

# Train new model (uses configured framework; PyTorch recommended)
python model_new.py

# Continue training existing model
python model_train.py
```

**Under the hood:**

- PyTorch: `clogic.ai_pytorch` + `segmentation_models_pytorch` (AMP, scheduler, tqdm)
- TensorFlow: `clogic.ai` + `segmentation_models` (legacy)

#### Outputs and Checkpoints (PyTorch)

- Per-epoch checkpoints: `{model_path}_epochNNN.pth`
- Rolling last checkpoint: `{model_path}_last.pth`
- Best by IoU: `{model_path}_best.pth`
- Final at end: `{model_path}_final.pth`

By default, `model_new.py` saves to files prefixed by `_new` in the project root (e.g., `_new_best.pth`). Pass a folder in `model_path` to organize runs (e.g., `models/run1/cytounet`).

#### Dataset Structure (generated by get_dataset.py)

- Images: `dataset/rois/` (or `${DATASET_FOLDER}/${IMAGES_FOLDER}` from config)
- Masks: `dataset/masks/` (or `${DATASET_FOLDER}/${MASKS_FOLDER}` from config)
- Tile sizes: 128×128 by default for training (see `config.IMAGE_SHAPE`)

#### Step 5: Use Your Trained Model

After training completes, copy the trained model to use locally:

**For PyTorch models (recommended):**

```bash
# Create folder if needed
mkdir -p _main

# Copy your best or final weights into _main
cp _new_best.pth  _main/model_best.pth   # or
cp _new_final.pth _main/model_final.pth  # or
cp _new_last.pth  _main/model.pth
```

The desktop app (`run.py`) looks for PyTorch weights in `_main/` under these names: `model.pth`, `model_best.pth`, `model_final.pth`.

**For TensorFlow models:**

```bash
cp -r trained_model_output/ _main/
```

Your custom model will now be used automatically when you run `python run.py`.

## Getting AI Models

### Option 1: Local Models (Recommended)

For best performance and offline analysis:

1. **Create model folder:**

   ```bash
   mkdir _main
   ```

2. **Place your trained model:**
   - PyTorch: Copy `.pth` weights to `_main/` as `model.pth`, `model_best.pth`, or `model_final.pth`
   - TensorFlow: Copy SavedModel directory to `_main/`
   - The application will automatically detect and load local models

3. **Model requirements:**
   - **PyTorch**: State dict (.pth files) in `_main/` folder (recommended)
   - **TensorFlow**: SavedModel format in `_main/` folder  
   - Input shape: (128, 128, 3)
   - Output: Segmentation mask for cell classification

**Local model characteristics:**

- Runs offline (no internet required)
- Inference speed depends on your hardware
- Data processed locally
- Model weights loaded into memory on startup

### Option 2: Remote Models

Models running on remote servers via TensorFlow Serving. Requires internet connection.

**To use custom cloud models:**

1. Edit `clogic/inference.py`
2. Modify the `apply_remote` function parameters:
   - `endpoint_url`: Your TensorFlow Serving endpoint
   - `model_name`: Your model's name on the server

Remote models send data over network to TensorFlow Serving endpoints.

## Understanding Annotations

### ASAP Annotation Format

Cytologick uses ASAP annotation format for training data. See `annotation_example.xml` for a sample annotation file.

**Key elements:**

- **Rectangles**: Define regions of interest around abnormal cells
- **Coordinates**: Precise pixel locations for each annotation
- **Labels**: Cell type classifications (LSIL, HSIL, ASCUS, etc.)

**Creating annotations:**

1. Open slide in [ASAP](https://computationalpathologygroup.github.io/ASAP/)
2. Draw rectangles around abnormal cells
3. Label each rectangle with appropriate cell type
4. Save as XML annotation file

## Cell Types (Research Focus)

Cytologick research focuses on these cervical cell patterns:

- **LSIL** - Low-grade Squamous Intraepithelial Lesion
- **HSIL** - High-grade Squamous Intraepithelial Lesion  
- **ASCUS** - Atypical Squamous Cells of Undetermined Significance
- **ASCH** - Atypical Squamous Cells, Cannot Exclude HSIL

## Keywords

Pap smear research, cytology AI, medical image analysis research, LSIL detection research, HSIL detection research, ASCUS detection research, digital pathology research, deep learning, U-Net segmentation, TensorFlow, PyTorch, computer vision research

## Contributing

If you're interested in contributing to Cytologick, please read our contributing guidelines. We welcome issues and pull requests!
