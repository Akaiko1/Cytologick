#pragma once

#include <QDialog>
#include <QLabel>
#include <QRadioButton>
#include <QSlider>
#include <QPushButton>
#include <QAbstractButton>
#include <QPixmap>
#include <QRect>
#include <QSize>
#include <QFuture>
#include <QFutureWatcher>
#include <opencv2/core.hpp>
#include <vector>

#include "graphics.h"
#include "annotator/annotation_types.h"

namespace cytologick {

class MainWindow;

class ToggleSwitch : public QAbstractButton {
    Q_OBJECT

public:
    explicit ToggleSwitch(QWidget* parent = nullptr);
    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event) override;
};

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
    PreviewWindow(MainWindow* parent,
                  const QPixmap& pixmap,
                  const cv::Mat& sourceImage,
                  const std::vector<Annotation>& referenceAnnotations,
                  const QRect& sourceRectLevel0);
    ~PreviewWindow() override = default;

private slots:
    void onConfidenceChanged(int value);
    void onAnalyzeClicked();
    void onModeToggled(bool checked);
    void onRenderFinished();
    void onShowMarkupToggled(bool checked);

private:
    void setupUi();
    void updateConfidenceLabel();
    void renderOverlayFromCache(bool fullAnalysis = false);
    void scheduleRender();
    void rebuildMarkupOverlay();
    void updateDisplayedOverlay();

    MainWindow* m_mainWindow;

    // UI elements
    PreviewDisplayLabel* m_displayLabel = nullptr;
    QLabel* m_infoLabel = nullptr;
    QLabel* m_confLabel = nullptr;
    QRadioButton* m_directButton = nullptr;
    QRadioButton* m_smoothButton = nullptr;
    QRadioButton* m_regionButton = nullptr;
    ToggleSwitch* m_showMarkupToggle = nullptr;
    QSlider* m_confSlider = nullptr;
    QPushButton* m_analyzeButton = nullptr;

    // Images
    QPixmap m_originalImage;
    cv::Mat m_sourceImage;
    cv::Mat m_cachedPathologyMap;
    std::vector<RegionBbox> m_cachedRegionBboxes;
    std::vector<Annotation> m_referenceAnnotations;
    QRect m_sourceRectLevel0;
    QPixmap m_inferenceOverlayPixmap;
    QPixmap m_markupOverlayPixmap;

    // Async rendering
    QFutureWatcher<cv::Mat>* m_renderWatcher = nullptr;
    float m_pendingThreshold = 0.0f;
    bool m_renderPending = false;

    // Settings
    float m_confThreshold = 0.6f;
    bool m_useSmooth = false;
    bool m_useRegion = false;
    bool m_showMarkup = false;
};

} // namespace cytologick
