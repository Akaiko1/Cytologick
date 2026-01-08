#include "previewwindow.h"
#include "mainwindow.h"
#include "graphics.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPainter>
#include <QMessageBox>
#include <QApplication>
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
    m_original = original;
    m_overlay = overlay;
    resize(original.size());
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
    setMaximumSize(2000, 1200);
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

    // Confidence threshold
    m_confLabel = new QLabel(this);
    updateConfidenceLabel();

    m_confSlider = new QSlider(Qt::Horizontal, this);
    m_confSlider->setMinimum(0);
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
    cv::Mat pathologyMap = inference.runInference(m_sourceImage, config);

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

    // Process results into visualization
    auto [overlay, stats] = processDensePathologyMap(pathologyMap, m_confThreshold);

    std::cout << "Detections: " << stats.totalDetections << " (atypical: " << stats.atypicalCells
              << ", normal: " << stats.normalCells << ")" << std::endl;

    // Save overlay for debugging
    cv::imwrite("gui_map.png", overlay);

    // Convert overlay to QPixmap
    // OpenCV uses BGRA, Qt expects RGBA for Format_RGBA8888
    cv::Mat rgbaOverlay;
    cv::cvtColor(overlay, rgbaOverlay, cv::COLOR_BGRA2RGBA);

    QImage overlayImage(rgbaOverlay.data, rgbaOverlay.cols, rgbaOverlay.rows,
                        rgbaOverlay.step, QImage::Format_RGBA8888);
    QPixmap overlayPixmap = QPixmap::fromImage(overlayImage.copy());

    // Update display with overlay
    m_displayLabel->setImages(m_originalImage, overlayPixmap);

    // Update info label
    m_infoLabel->setText(QString::fromStdString(formatDetectionStats(stats)));

    QApplication::restoreOverrideCursor();
    m_analyzeButton->setEnabled(true);
    m_analyzeButton->setText("Analyze");
}

} // namespace cytologick
