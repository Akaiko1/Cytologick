# Cytologick C++ GUI

Standalone C++ version of the Cytologick desktop application for AI-powered Pap smear analysis.

## Features

- Qt6-based desktop GUI
- ONNX Runtime for model inference (CPU/GPU)
- OpenSlide for whole-slide image reading (MRXS only, more formats coming soon)
- Direct inference mode with sliding window
- Confidence threshold adjustment
- Detection visualization with overlays

---

## Quick Start

```bash
# 1. Export model to ONNX (requires Python environment)
python scripts/export_onnx.py

# 2. Install dependencies and build (see platform-specific instructions below)

# 3. Run
./build/Cytologick
```

---

## Running Pre-built Binary

Download the latest release from [GitHub Releases](https://github.com/Akaiko1/Cytologick/releases).

Each release includes:

- `Cytologick` executable (or `Cytologick.exe` on Windows)
- `model.onnx` — pre-trained ONNX model (separate download)
- `config.yaml` — default configuration
- Required DLLs (Windows only)

### Setup

1. Extract the archive to any folder
2. Create `_main` folder and place the model inside:

   ```text
   Cytologick/
   ├── Cytologick.exe
   ├── config.yaml
   └── _main/
       └── model.onnx    <-- place model here
   ```

3. Edit `config.yaml` to set your slide directory:

   ```yaml
   general:
     slide_dir: /path/to/your/slides    # Folder with .mrxs files
     hdd_slides: /path/to/more/slides   # Optional: additional slides folder
   ```

4. Install runtime dependencies (macOS/Linux only — Windows has all DLLs bundled)

### macOS (Apple Silicon)

```bash
# Install runtime dependencies via Homebrew
brew install qt@6 opencv openslide yaml-cpp onnxruntime

# Run the application
./Cytologick
```

### Linux (Ubuntu 22.04+)

```bash
# Install runtime dependencies
sudo apt-get install -y \
    libqt6core6 libqt6gui6 libqt6widgets6 \
    libopencv-core406 libopencv-imgproc406 libopencv-imgcodecs406 \
    libopenslide0 \
    libyaml-cpp0.8

# ONNX Runtime - download and install manually
ONNX_VERSION=1.16.3
wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz
tar -xzf onnxruntime-linux-x64-${ONNX_VERSION}.tgz
sudo mv onnxruntime-linux-x64-${ONNX_VERSION} /opt/onnxruntime
echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Run the application
./Cytologick
```

### Linux (Fedora/RHEL)

```bash
# Install runtime dependencies
sudo dnf install -y \
    qt6-qtbase \
    opencv \
    openslide \
    yaml-cpp

# ONNX Runtime - same as Ubuntu (download manually)

# Run the application
./Cytologick
```

### Linux (Arch)

```bash
# Install runtime dependencies
sudo pacman -S qt6-base opencv openslide yaml-cpp

# ONNX Runtime from AUR
yay -S onnxruntime

# Run the application
./Cytologick
```

### Windows

Download the release archive from [GitHub Releases](https://github.com/Akaiko1/Cytologick/releases) — all required DLLs are already included.

```powershell
# Extract and run
Expand-Archive Cytologick-windows-x64.zip -DestinationPath Cytologick
cd Cytologick
.\Cytologick.exe
```

---

## Build Instructions

### macOS (Apple Silicon & Intel)

#### Prerequisites

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake qt@6 opencv openslide yaml-cpp onnxruntime

# For Apple Silicon, ensure Rosetta 2 is NOT used (native ARM build)
```

#### Build

```bash
cd cpp

# Create build directory
mkdir -p build && cd build

# Configure with CMake
# Qt6 path varies by architecture:
#   Apple Silicon: /opt/homebrew/opt/qt@6
#   Intel Mac: /usr/local/opt/qt@6

# Apple Silicon (M1/M2/M3):
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qt@6;/opt/homebrew" \
    -DOPENSLIDE_INCLUDE_DIR=/opt/homebrew/include \
    -DOPENSLIDE_LIBRARY=/opt/homebrew/lib/libopenslide.dylib

# Intel Mac:
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/usr/local/opt/qt@6;/usr/local" \
    -DOPENSLIDE_INCLUDE_DIR=/usr/local/include \
    -DOPENSLIDE_LIBRARY=/usr/local/lib/libopenslide.dylib

# Build
cmake --build . -j$(sysctl -n hw.ncpu)

# Run
./Cytologick
```

#### Create macOS App Bundle (Optional)

```bash
# After building, create .app bundle
mkdir -p Cytologick.app/Contents/MacOS
mkdir -p Cytologick.app/Contents/Frameworks
cp Cytologick Cytologick.app/Contents/MacOS/

# Create Info.plist
cat > Cytologick.app/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Cytologick</string>
    <key>CFBundleIdentifier</key>
    <string>com.cytologick.app</string>
    <key>CFBundleName</key>
    <string>Cytologick</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF

# Deploy Qt frameworks (requires macdeployqt)
/opt/homebrew/opt/qt@6/bin/macdeployqt Cytologick.app
```

---

### Linux (Ubuntu 22.04 / Debian 12)

#### Prerequisites

```bash
# Update package list
sudo apt-get update

# Install build tools
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git

# Install Qt6
sudo apt-get install -y \
    qt6-base-dev \
    qt6-base-dev-tools \
    libqt6core6 \
    libqt6gui6 \
    libqt6widgets6

# Install OpenCV
sudo apt-get install -y \
    libopencv-dev

# Install OpenSlide
sudo apt-get install -y \
    libopenslide-dev \
    libopenslide0

# Install yaml-cpp
sudo apt-get install -y \
    libyaml-cpp-dev

# Install ONNX Runtime (download pre-built)
ONNX_VERSION=1.16.3
wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz
tar -xzf onnxruntime-linux-x64-${ONNX_VERSION}.tgz
sudo mv onnxruntime-linux-x64-${ONNX_VERSION} /opt/onnxruntime

# Add to library path
echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Build

```bash
cd cpp
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Donnxruntime_INCLUDE_DIR=/opt/onnxruntime/include \
    -Donnxruntime_LIB=/opt/onnxruntime/lib/libonnxruntime.so

cmake --build . -j$(nproc)

# Run
./Cytologick
```

#### Create AppImage (Optional)

```bash
# Install linuxdeploy
wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
wget https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-x86_64.AppImage
chmod +x linuxdeploy*.AppImage

# Create AppDir structure
mkdir -p AppDir/usr/bin
mkdir -p AppDir/usr/share/applications
mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps

cp Cytologick AppDir/usr/bin/

# Create .desktop file
cat > AppDir/usr/share/applications/cytologick.desktop << 'EOF'
[Desktop Entry]
Name=Cytologick
Exec=Cytologick
Icon=cytologick
Type=Application
Categories=Science;Medical;
EOF

# Build AppImage
./linuxdeploy-x86_64.AppImage --appdir AppDir --plugin qt --output appimage
```

---

### Windows 10/11

#### Prerequisites

1. **Visual Studio 2022** - Install with "Desktop development with C++" workload
   - Download from: https://visualstudio.microsoft.com/

2. **CMake 3.20+** - Usually included with VS, or download from https://cmake.org/

3. **vcpkg** - Package manager for C++ dependencies

```powershell
# Open PowerShell as Administrator

# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg

# Bootstrap
.\bootstrap-vcpkg.bat

# Integrate with Visual Studio
.\vcpkg integrate install

# Install dependencies (this takes a while)
.\vcpkg install qtbase:x64-windows opencv4:x64-windows yaml-cpp:x64-windows

# ONNX Runtime - download manually (not available in vcpkg for CPU-only)
# See next section
```

4. **OpenSlide** - Download Windows binaries

```powershell
# Download OpenSlide
Invoke-WebRequest -Uri "https://github.com/openslide/openslide-winbuild/releases/download/v20231011/openslide-win64-20231011.zip" -OutFile openslide.zip
Expand-Archive openslide.zip -DestinationPath C:\openslide
```

5. **ONNX Runtime** - Download pre-built binaries

```powershell
# Download ONNX Runtime (CPU)
$ONNX_VERSION = "1.16.3"
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v$ONNX_VERSION/onnxruntime-win-x64-$ONNX_VERSION.zip" -OutFile onnxruntime.zip
Expand-Archive onnxruntime.zip -DestinationPath C:\onnxruntime
```

#### Build

```powershell
cd cpp
mkdir build
cd build

# Configure with CMake
$ONNX_VERSION = "1.16.3"
cmake .. `
    -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake `
    -DVCPKG_MANIFEST_MODE=OFF `
    -DOPENSLIDE_INCLUDE_DIR=C:\openslide\openslide-win64-20231011\include `
    -DOPENSLIDE_LIBRARY=C:\openslide\openslide-win64-20231011\lib\libopenslide.lib `
    -Donnxruntime_INCLUDE_DIR=C:\onnxruntime\onnxruntime-win-x64-$ONNX_VERSION\include `
    -Donnxruntime_LIB=C:\onnxruntime\onnxruntime-win-x64-$ONNX_VERSION\lib\onnxruntime.lib

# Build Release
cmake --build . --config Release

# The executable will be at: build\Release\Cytologick.exe
```

#### Create Standalone Distribution

```powershell
# Create distribution folder
mkdir dist
copy Release\Cytologick.exe dist\

# Copy OpenSlide DLLs
copy C:\openslide\openslide-win64-20231011\bin\*.dll dist\

# Deploy Qt DLLs
C:\vcpkg\installed\x64-windows\tools\Qt6\bin\windeployqt.exe dist\Cytologick.exe

# Copy ONNX Runtime DLL
$ONNX_VERSION = "1.16.3"
copy C:\onnxruntime\onnxruntime-win-x64-$ONNX_VERSION\lib\onnxruntime.dll dist\

# Copy model
mkdir dist\_main
copy ..\..\_main\model.onnx dist\_main\
```

---

## Exporting the Model

Before running the C++ application, you must export your PyTorch model to ONNX format.

### Requirements

```bash
# Activate your Python environment with PyTorch
conda activate cytologick  # or your environment name

# Ensure required packages are installed
pip install torch onnx segmentation-models-pytorch
```

### Export

```bash
cd cpp/scripts

# Auto-detect model in _main folder
python export_onnx.py

# Or specify paths explicitly
python export_onnx.py --input ../../_main/_new_best.pth --output ../../_main/model.onnx

# Custom settings
python export_onnx.py \
    --input ../../_main/_new_best.pth \
    --output ../../_main/model.onnx \
    --classes 3 \
    --size 128
```

The exported `model.onnx` file should be placed in the `_main/` directory.

---

## Usage

1. **Export model** to ONNX format (see above)

2. **Place model** in `_main/` folder (e.g., `_main/model.onnx`)

3. **Run the application:**
   ```bash
   ./Cytologick          # Linux/macOS
   Cytologick.exe        # Windows
   ```

4. **Select a slide** from the menu dialog

5. **Choose zoom level** (lower level = higher resolution, more memory)

6. **Drag to select** a region of interest on the slide

7. **Click "Analyze"** in the preview window

8. **Adjust confidence** threshold with the slider

---

## Configuration

Create `config.yaml` in the same directory as the executable:

```yaml
general:
  slide_dir: /path/to/slides
  hdd_slides: /path/to/more/slides
  openslide_path: C:/openslide/bin  # Windows only

neural_network:
  image_shape: [128, 128]
  image_chunk: [256, 256]
  classes: 3
  batch_size: 16

gui:
  default_threshold: 0.6

model:
  path: /path/to/model.onnx  # Optional, auto-detected from _main/
```

---

## Project Structure

```text
cpp/
├── CMakeLists.txt          # Build configuration
├── vcpkg.json              # vcpkg dependencies (Windows)
├── README.md               # This file
├── src/
│   ├── main.cpp            # Application entry point
│   ├── mainwindow.h/cpp    # Main viewer window
│   ├── menuwindow.h/cpp    # Slide selection dialog
│   ├── previewwindow.h/cpp # Analysis preview window
│   ├── inference.h/cpp     # ONNX inference engine
│   ├── graphics.h/cpp      # Visualization utilities
│   ├── slidereader.h/cpp   # OpenSlide wrapper
│   └── config.h/cpp        # Configuration loading
├── resources/
│   └── cytologick.qrc      # Qt resources
└── scripts/
    └── export_onnx.py      # Model export script
```

---

## Troubleshooting

### "No model found"

- Export your model to ONNX format: `python scripts/export_onnx.py`
- Place the `.onnx` file in the `_main/` directory
- Supported names: `model.onnx`, `new_best.onnx`, `model_best.onnx`

### "CUDA not available"

The application automatically falls back to CPU inference. For GPU support:

- **Windows:** Use `onnxruntime-gpu` from vcpkg, install CUDA Toolkit
- **Linux:** Download ONNX Runtime GPU build, install CUDA drivers
- **macOS:** GPU support via CoreML (automatic on Apple Silicon)

### "Failed to open slide"

- Verify OpenSlide is installed: `brew info openslide` / `apt show libopenslide0`
- **Windows:** Ensure OpenSlide DLLs are in PATH or same folder as exe
- Check slide format is supported (currently MRXS only)

### CMake can't find Qt6

```bash
# macOS - add Qt to CMAKE_PREFIX_PATH
cmake .. -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/qt@6

# Linux - install qt6-base-dev
sudo apt install qt6-base-dev

# Windows - use vcpkg toolchain file
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```

### CMake can't find ONNX Runtime

```bash
# Linux - set include and lib paths explicitly
cmake .. \
    -Donnxruntime_INCLUDE_DIR=/opt/onnxruntime/include \
    -Donnxruntime_LIB=/opt/onnxruntime/lib/libonnxruntime.so

# Windows - same approach
cmake .. `
    -Donnxruntime_INCLUDE_DIR=C:\onnxruntime\onnxruntime-win-x64-1.16.3\include `
    -Donnxruntime_LIB=C:\onnxruntime\onnxruntime-win-x64-1.16.3\lib\onnxruntime.lib
```

### Linker errors on macOS

```bash
# Ensure all Homebrew packages are linked
brew link --force qt@6
brew link --force opencv
brew link --force openslide
```

---

## License

Same as the main Cytologick project.
