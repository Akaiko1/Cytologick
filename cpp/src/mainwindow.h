#pragma once

#include "config.h"
#include "slidereader.h"
#include "inference.h"

#include <QMainWindow>
#include <QScrollArea>
#include <QLabel>
#include <QPixmap>
#include <QPoint>
#include <QImage>
#include <QRect>
#include <QRectF>
#include <QWidget>
#include <QEvent>
#include <memory>

namespace cytologick {

class MenuWindow;
class PreviewWindow;
class MainWindow;

class OverviewMapWidget : public QWidget {
    Q_OBJECT

public:
    explicit OverviewMapWidget(QWidget* parent = nullptr);

    void setOverview(const QImage& image, double downsampleToLevel0, const QSize& level0Size);
    void setViewportLevel0(const QRectF& viewportLevel0);
    void clear();

signals:
    void jumpRequestedLevel0(QPointF centerLevel0);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QRect imageTargetRect() const;
    QPointF widgetToLevel0(const QPoint& pos) const;

    QImage m_overview;
    QSize m_level0Size;
    QRectF m_viewportLevel0;
    double m_overviewDownsample = 1.0;
    bool m_mouseDown = false;
};

/**
 * Custom QLabel for slide image display with selection rectangle
 */
class ImageLabel : public QLabel {
    Q_OBJECT

public:
    explicit ImageLabel(MainWindow* mainWindow, QWidget* parent = nullptr);

    void setSlideImage(const QPixmap& pixmap);
    void setVirtualCanvas(const QSize& size, int levelIndex, double downsample);
    void invalidateRegionCache();
    void setSelectionRect(const QPoint& start, const QPoint& end, bool visible);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QRect computeCacheRect(const QRect& targetRect) const;

    MainWindow* m_mainWindow;
    QPixmap m_slideImage;
    QPoint m_selStart;
    QPoint m_selEnd;
    bool m_showSelection = false;

    bool m_useVirtualCanvas = false;
    int m_levelIndex = 0;
    double m_downsample = 1.0;
    QImage m_cachedRegionImage;
    QRect m_cachedRegionRect;
};

/**
 * Main application window for slide viewing and region selection
 * Port of Python Viewer class
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    // Accessors for child windows
    Config& getConfig() { return m_config; }
    SlideReader& getSlideReader() { return m_slideReader; }
    InferenceEngine& getInferenceEngine() { return m_inference; }

    /**
     * Set the current slide image to display
     * @param pixmap Slide image at current zoom level
     * @param scaleFactor Scaling coefficient from level 0
     */
    void setSlideImage(const QPixmap& pixmap, double scaleFactor);
    void loadSlideLevel(int levelIndex);
    void onSlideOpened(const std::filesystem::path& path);

    /**
     * Get current scaling coefficient
     */
    double getScaleFactor() const { return m_scaleFactor; }

    // Called by ImageLabel when selection completes
    void onSelectionComplete(const QPoint& pressPos, const QPoint& releasePos);
    void updateStatusBar(const QString& message);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private slots:
    void showSlideMenu();
    void onSelectWorkingDirectory();
    void onMapJumpRequested(QPointF centerLevel0);

private:
    void setupUi();
    void loadModel();
    void showPreview();
    void applyWorkingDirectory(const std::filesystem::path& dir, bool persist);
    void rebuildOverviewMap();
    void updateOverviewViewport();
    void centerViewOnLevel0(const QPointF& centerLevel0);
    void positionOverviewMapOverlay();

    // Configuration and model
    Config m_config;
    SlideReader m_slideReader;
    InferenceEngine m_inference;

    // UI elements
    QScrollArea* m_scrollArea = nullptr;
    ImageLabel* m_imageLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    OverviewMapWidget* m_overviewMap = nullptr;

    // Child windows
    std::unique_ptr<MenuWindow> m_menuWindow;

    // Selection state
    QPoint m_pressPos;
    QPoint m_releasePos;
    double m_scaleFactor = 1.0;
    int m_currentLevel = 0;
    bool m_levelLoaded = false;
    std::filesystem::path m_slidePath;
};

} // namespace cytologick
