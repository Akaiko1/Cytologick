#include "previewwindow.h"
#include "mainwindow.h"
#include "graphics.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPainter>
#include <QMessageBox>
#include <QApplication>
#include <QtConcurrent>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

namespace cytologick {

// ============================================================================
// PreviewDisplayLabel implementation
// ============================================================================

PreviewDisplayLabel::PreviewDisplayLabel(QWidget* parent)
    : QLabel(parent)
{
}

void PreviewDisplayLabel::setImages(const QPixmap& original, const QPixmap& overlay) {
    bool sizeChanged = m_original.isNull() || m_original.size() != original.size();
    m_original = original;
    m_overlay = overlay;
    if (sizeChanged) {
        resize(original.size());
    }
    update();
}

void PreviewDisplayLabel::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);

    // Draw original image
    if (!m_original.isNull()) {
        painter.drawPixmap(rect(), m_original);
    }

    // Draw overlay on top
    if (!m_overlay.isNull()) {
        painter.drawPixmap(rect(), m_overlay);
    }
}

// ============================================================================
// PreviewWindow implementation
// ============================================================================

PreviewWindow::PreviewWindow(MainWindow* parent, const QPixmap& pixmap, const cv::Mat& sourceImage)
    : QDialog(parent)
    , m_mainWindow(parent)
    , m_originalImage(pixmap)
    , m_sourceImage(sourceImage.clone())
{
    setupUi();
    setMaximumSize(1600, 900);
    resize(m_originalImage.width() + 220, m_originalImage.height());
}

void PreviewWindow::setupUi() {
    setWindowTitle("Preview");

    QHBoxLayout* mainLayout = new QHBoxLayout(this);

    // Display area - custom label with paint override
    m_displayLabel = new PreviewDisplayLabel(this);
    m_displayLabel->setImages(m_originalImage);
    m_displayLabel->setMinimumSize(100, 100);

    // Control panel
    QWidget* controlWidget = new QWidget(this);
    controlWidget->setMaximumWidth(200);
    QVBoxLayout* controlLayout = new QVBoxLayout(controlWidget);

    // Model status
    QLabel* modelStatus = new QLabel(this);
    if (m_mainWindow->getInferenceEngine().isModelLoaded()) {
        modelStatus->setText("Model: Loaded");
        modelStatus->setStyleSheet("color: #2ecc71;");
    } else {
        modelStatus->setText("Model: Not loaded");
        modelStatus->setStyleSheet("color: #e74c3c;");
    }

    // Inference mode
    m_directButton = new QRadioButton("Local: Fast", this);
    m_smoothButton = new QRadioButton("Local: Comprehensive", this);
    m_regionButton = new QRadioButton("Local: Region", this);
    const auto& cfg = m_mainWindow->getConfig();
    if (cfg.predMode == "smooth") {
        m_useSmooth = true;
        m_useRegion = false;
        m_smoothButton->setChecked(true);
    } else if (cfg.predMode == "region") {
        m_useSmooth = false;
        m_useRegion = true;
        m_regionButton->setChecked(true);
    } else {
        m_useSmooth = false;
        m_useRegion = false;
        m_directButton->setChecked(true);
    }
    connect(m_directButton, &QRadioButton::toggled, this, &PreviewWindow::onModeToggled);
    connect(m_smoothButton, &QRadioButton::toggled, this, &PreviewWindow::onModeToggled);
    connect(m_regionButton, &QRadioButton::toggled, this, &PreviewWindow::onModeToggled);

    // Confidence threshold
    m_confLabel = new QLabel(this);
    updateConfidenceLabel();

    m_confSlider = new QSlider(Qt::Horizontal, this);
    m_confSlider->setMinimum(30);
    m_confSlider->setMaximum(100);
    m_confSlider->setSingleStep(1);
    m_confSlider->setValue(static_cast<int>(m_confThreshold * 100));
    connect(m_confSlider, &QSlider::valueChanged, this, &PreviewWindow::onConfidenceChanged);

    // Results info
    m_infoLabel = new QLabel("Analysis\nResults", this);
    m_infoLabel->setWordWrap(true);

    // Analyze button
    m_analyzeButton = new QPushButton("Analyze", this);
    m_analyzeButton->setEnabled(m_mainWindow->getInferenceEngine().isModelLoaded());
    connect(m_analyzeButton, &QPushButton::clicked, this, &PreviewWindow::onAnalyzeClicked);

    // Assemble control panel
    controlLayout->addWidget(modelStatus);
    controlLayout->addSpacing(10);
    controlLayout->addWidget(m_directButton);
    controlLayout->addWidget(m_smoothButton);
    controlLayout->addWidget(m_regionButton);
    controlLayout->addSpacing(5);
    controlLayout->addWidget(m_confLabel);
    controlLayout->addWidget(m_confSlider);
    controlLayout->addSpacing(10);
    controlLayout->addWidget(m_infoLabel);
    controlLayout->addStretch();
    controlLayout->addWidget(m_analyzeButton);

    // Assemble main layout
    mainLayout->addWidget(m_displayLabel);
    mainLayout->addWidget(controlWidget);
}

void PreviewWindow::updateConfidenceLabel() {
    m_confLabel->setText(QString("Min confidence: %1%").arg(static_cast<int>(m_confThreshold * 100)));
}

void PreviewWindow::onConfidenceChanged(int value) {
    m_confThreshold = static_cast<float>(value) / 100.0f;
    updateConfidenceLabel();
    if (!m_cachedPathologyMap.empty()) {
        scheduleRender();
    }
}

void PreviewWindow::onAnalyzeClicked() {
    auto& inference = m_mainWindow->getInferenceEngine();
    auto& config = m_mainWindow->getConfig();

    if (!inference.isModelLoaded()) {
        QMessageBox::warning(this, "Error", "No model loaded");
        return;
    }

    m_analyzeButton->setEnabled(false);
    m_analyzeButton->setText("Analyzing...");
    QApplication::processEvents();
    QApplication::setOverrideCursor(Qt::WaitCursor);

    std::cout << "Running inference on image: " << m_sourceImage.cols << "x" << m_sourceImage.rows << std::endl;

    // Run inference
    cv::Mat pathologyMap;
    m_cachedRegionBboxes.clear();
    if (m_useSmooth) {
        pathologyMap = inference.runInferenceSmooth(m_sourceImage, config);
    } else if (m_useRegion) {
        auto result = inference.runInferenceRegion(m_sourceImage, config);
        pathologyMap = result.first;
        m_cachedRegionBboxes.reserve(result.second.size());
        for (const auto& bbox : result.second) {
            m_cachedRegionBboxes.push_back({bbox.x, bbox.y, bbox.width, bbox.height, bbox.maxProbability});
        }
    } else {
        pathologyMap = inference.runInference(m_sourceImage, config);
    }

    if (pathologyMap.empty()) {
        QApplication::restoreOverrideCursor();
        m_analyzeButton->setEnabled(true);
        m_analyzeButton->setText("Analyze");
        QMessageBox::warning(this, "Error",
            QString("Inference failed:\n%1").arg(
                QString::fromStdString(inference.getLastError())
            )
        );
        return;
    }

    std::cout << "Pathology map size: " << pathologyMap.cols << "x" << pathologyMap.rows
              << " channels: " << pathologyMap.channels() << std::endl;

    m_cachedPathologyMap = pathologyMap.clone();
    renderOverlayFromCache(true);

    QApplication::restoreOverrideCursor();
    m_analyzeButton->setEnabled(true);
    m_analyzeButton->setText("Analyze");
}

void PreviewWindow::onModeToggled(bool checked) {
    if (!checked) {
        return;
    }
    m_useSmooth = (sender() == m_smoothButton);
    m_useRegion = (sender() == m_regionButton);
    if (m_useSmooth) {
        m_mainWindow->getConfig().predMode = "smooth";
    } else if (m_useRegion) {
        m_mainWindow->getConfig().predMode = "region";
    } else {
        m_mainWindow->getConfig().predMode = "direct";
    }
    m_cachedPathologyMap.release();
    m_cachedRegionBboxes.clear();
    m_displayLabel->setImages(m_originalImage);
    m_infoLabel->setText("Analysis\nResults");
}

void PreviewWindow::renderOverlayFromCache(bool fullAnalysis) {
    if (m_cachedPathologyMap.empty()) {
        return;
    }

    cv::Mat overlay;
    if (fullAnalysis) {
        // Full analysis with stats (after Analyze button click)
        auto [fullOverlay, stats] = processDensePathologyMap(m_cachedPathologyMap, m_confThreshold);
        overlay = fullOverlay;
        if (!m_cachedRegionBboxes.empty()) {
            drawRegionBboxes(overlay, m_cachedRegionBboxes);
        }

        std::cout << "Detections: " << stats.totalDetections << " (atypical: " << stats.atypicalCells
                  << ", normal: " << stats.normalCells << ")" << std::endl;

        cv::imwrite("gui_map.png", overlay);
        m_infoLabel->setText(QString::fromStdString(formatDetectionStats(stats)));
    } else {
        // Fast rendering for slider updates (no stats recalculation)
        overlay = renderOverlayFast(m_cachedPathologyMap, m_confThreshold);
        if (!overlay.empty() && !m_cachedRegionBboxes.empty()) {
            drawRegionBboxes(overlay, m_cachedRegionBboxes);
        }
    }

    if (overlay.empty()) {
        return;
    }

    // Convert overlay to QPixmap
    // OpenCV uses BGRA, Qt expects RGBA for Format_RGBA8888
    cv::Mat rgbaOverlay;
    cv::cvtColor(overlay, rgbaOverlay, cv::COLOR_BGRA2RGBA);

    QImage overlayImage(rgbaOverlay.data, rgbaOverlay.cols, rgbaOverlay.rows,
                        rgbaOverlay.step, QImage::Format_RGBA8888);
    QPixmap overlayPixmap = QPixmap::fromImage(overlayImage.copy());

    // Update display with overlay
    m_displayLabel->setImages(m_originalImage, overlayPixmap);
}

void PreviewWindow::scheduleRender() {
    // Store current threshold for async render
    m_pendingThreshold = m_confThreshold;

    // If render is already in progress, mark as pending and return
    if (m_renderWatcher && m_renderWatcher->isRunning()) {
        m_renderPending = true;
        return;
    }

    // Create watcher if needed
    if (!m_renderWatcher) {
        m_renderWatcher = new QFutureWatcher<cv::Mat>(this);
        connect(m_renderWatcher, &QFutureWatcher<cv::Mat>::finished,
                this, &PreviewWindow::onRenderFinished);
    }

    // Capture variables for lambda
    cv::Mat pathologyMap = m_cachedPathologyMap.clone();
    float threshold = m_pendingThreshold;
    std::vector<RegionBbox> regionBboxes = m_cachedRegionBboxes;

    // Run rendering in background thread
    QFuture<cv::Mat> future = QtConcurrent::run([pathologyMap, threshold, regionBboxes]() {
        cv::Mat overlay = renderOverlayFast(pathologyMap, threshold);
        if (!overlay.empty() && !regionBboxes.empty()) {
            drawRegionBboxes(overlay, regionBboxes);
        }
        return overlay;
    });

    m_renderWatcher->setFuture(future);
}

void PreviewWindow::onRenderFinished() {
    cv::Mat overlay = m_renderWatcher->result();

    if (!overlay.empty()) {
        // Convert overlay to QPixmap
        cv::Mat rgbaOverlay;
        cv::cvtColor(overlay, rgbaOverlay, cv::COLOR_BGRA2RGBA);

        QImage overlayImage(rgbaOverlay.data, rgbaOverlay.cols, rgbaOverlay.rows,
                            rgbaOverlay.step, QImage::Format_RGBA8888);
        QPixmap overlayPixmap = QPixmap::fromImage(overlayImage.copy());

        m_displayLabel->setImages(m_originalImage, overlayPixmap);
    }

    // If there's a pending render with a different threshold, start it
    if (m_renderPending) {
        m_renderPending = false;
        scheduleRender();
    }
}

} // namespace cytologick
