# Cytologick - Automated Pap Smear Analysis Software

## What is Cytologick?

Cytologick is a Python-based medical software that automatically analyzes Pap smear slides to detect abnormal cells using artificial intelligence. It helps medical professionals identify cellular abnormalities like LSIL, HSIL, and ASCUS by providing visual annotations and detailed reports.

## Features

✅ **Automated Cell Detection** - AI-powered identification of abnormal cervical cells  
✅ **MRXS Format Support** - Compatible with digital pathology slide formats  
✅ **ASAP Integration** - Works with ASAP annotation files for training  
✅ **Dual Interface** - Desktop application and web interface options  
✅ **Local & Cloud AI** - Support for both offline and cloud-based models  
✅ **Custom Training** - Train your own models on annotated datasets  
✅ **Medical Accuracy** - Detects LSIL, HSIL, ASCUS, and ASCH abnormalities  

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
```bash
# Install Python packages
pip install -r requirements.txt

# Install additional packages via conda
conda install tensorflow
conda install openslide-python
```

### Step 5: Download Web Dependencies
Download and place these files in `__web/static/`:
- [jquery.js](https://jquery.com)
- [openseadragon.js](https://openseadragon.github.io)
- openseadragon-scalebar.js

### Step 6: Configure Settings
Edit `config.py` or create `config.ini` to set your paths:

**Required Settings:**
```python
OPENSLIDE_PATH = "C:/path/to/openslide/bin"  # Windows
SLIDE_DIR = "./current"  # Folder containing your MRXS files
```

**Optional Settings:**
```python
IP_EXPOSED = "127.0.0.1"  # Web interface IP
UNET_PRED_MODE = "remote"  # Use cloud models
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
Opens a desktop application that can use both local and cloud-based AI models.

### 2. Web Interface (Under Construction)
```bash
python run_web.py
```
⚠️ **Note**: Web interface is currently under development. Use desktop application for stable analysis.

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
2. **Run Analysis**: Click analyze to detect abnormal cells
3. **View Results**: Abnormal areas will be highlighted with annotations
4. **Export Report**: Save results for medical review

### Advanced: Training Your Own Models

#### Step 1: Prepare Your Annotated Data
1. **Annotate slides** in [ASAP](https://computationalpathologygroup.github.io/ASAP/)
   - Draw rectangles around abnormal cells
   - Label each rectangle (LSIL, HSIL, ASCUS, etc.)
   - Save annotation files as XML

2. **Organize your files:**
   ```
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

#### Step 3: Run Training Pipeline
```bash
# Extract annotations from ASAP XML files
python get_xmls.py

# Create training dataset from extracted annotations
python get_dataset.py

# Define neural network architecture
python model_new.py

# Train the model (this will take several hours)
python model_train.py
```

#### Step 4: Use Your Trained Model
After training completes, copy the trained model to use locally:
```bash
# Copy trained model to local folder
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
   - Copy your TensorFlow model files to the `_main/` folder
   - The application will automatically detect and load local models

3. **Model requirements:**
   - TensorFlow SavedModel format
   - Input shape: (128, 128, 3)
   - Output: Segmentation mask for cell classification

**Benefits of local models:**
- No internet connection required
- Faster inference (no network latency)
- Full privacy - data stays on your machine
- More reliable for clinical use

### Option 2: Cloud-Based Models
Remote models hosted on **TensorFlow Serving** infrastructure. Requires internet connection.

**To use custom cloud models:**
1. Edit `clogic/inference.py`
2. Modify the `apply_remote` function parameters:
   - `endpoint_url`: Your TensorFlow Serving endpoint
   - `model_name`: Your model's name on the server

**Note:** Remote models use TensorFlow Serving backend for scalable inference.

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

## Cell Types Detected

Cytologick can identify these cervical cell abnormalities:
- **LSIL** - Low-grade Squamous Intraepithelial Lesion
- **HSIL** - High-grade Squamous Intraepithelial Lesion  
- **ASCUS** - Atypical Squamous Cells of Undetermined Significance
- **ASCH** - Atypical Squamous Cells, Cannot Exclude HSIL

## Keywords

Pap smear analysis, cervical cancer screening, cytology AI, medical image analysis, LSIL detection, HSIL detection, ASCUS detection, pathology automation, deep learning healthcare, digital pathology, medical diagnosis, EfficientNet U-Net, TensorFlow medical imaging

## Contributing

If you're interested in contributing to Cytologick, please read our contributing guidelines. We welcome issues and pull requests!
