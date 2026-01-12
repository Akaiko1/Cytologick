#pragma once

#include <QDialog>
#include <QLabel>
#include <QRadioButton>
#include <QSlider>
#include <QPushButton>
#include <QPixmap>
#include <QFuture>
#include <QFutureWatcher>
#include <opencv2/core.hpp>
#include <vector>

#include "graphics.h"

namespace cytologick {

class MainWindow;

/**
 * Custom QLabel for preview display with overlay support
 */
class PreviewDisplayLabel : public QLabel {
    Q_OBJECT

public:
    explicit PreviewDisplayLabel(QWidget* parent = nullptr);

    void setImages(const QPixmap& original, const QPixmap& overlay = QPixmap());

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QPixmap m_original;
    QPixmap m_overlay;
};

/**
 * Analysis preview window showing inference results
 * Port of Python Preview class
 */
class PreviewWindow : public QDialog {
    Q_OBJECT

public:
    /**
     * Create preview window
     * @param parent Parent MainWindow (provides model access)
     * @param pixmap Image to display
     * @param sourceImage OpenCV Mat of source for inference
     */
    PreviewWindow(MainWindow* parent, const QPixmap& pixmap, const cv::Mat& sourceImage);
    ~PreviewWindow() override = default;

private slots:
    void onConfidenceChanged(int value);
    void onAnalyzeClicked();
    void onModeToggled(bool checked);
    void onRenderFinished();

private:
    void setupUi();
    void updateConfidenceLabel();
    void renderOverlayFromCache(bool fullAnalysis = false);
    void scheduleRender();

    MainWindow* m_mainWindow;

    // UI elements
    PreviewDisplayLabel* m_displayLabel = nullptr;
    QLabel* m_infoLabel = nullptr;
    QLabel* m_confLabel = nullptr;
    QRadioButton* m_directButton = nullptr;
    QRadioButton* m_smoothButton = nullptr;
    QRadioButton* m_regionButton = nullptr;
    QSlider* m_confSlider = nullptr;
    QPushButton* m_analyzeButton = nullptr;

    // Images
    QPixmap m_originalImage;
    cv::Mat m_sourceImage;
    cv::Mat m_cachedPathologyMap;
    std::vector<RegionBbox> m_cachedRegionBboxes;

    // Async rendering
    QFutureWatcher<cv::Mat>* m_renderWatcher = nullptr;
    float m_pendingThreshold = 0.0f;
    bool m_renderPending = false;

    // Settings
    float m_confThreshold = 0.6f;
    bool m_useSmooth = false;
    bool m_useRegion = false;
};

} // namespace cytologick
