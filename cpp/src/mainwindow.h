#pragma once

#include "config.h"
#include "slidereader.h"
#include "inference.h"

#include <QMainWindow>
#include <QScrollArea>
#include <QLabel>
#include <QPixmap>
#include <QPoint>
#include <memory>

namespace cytologick {

class MenuWindow;
class PreviewWindow;
class MainWindow;

/**
 * Custom QLabel for slide image display with selection rectangle
 */
class ImageLabel : public QLabel {
    Q_OBJECT

public:
    explicit ImageLabel(MainWindow* mainWindow, QWidget* parent = nullptr);

    void setSlideImage(const QPixmap& pixmap);
    void setSelectionRect(const QPoint& start, const QPoint& end, bool visible);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    MainWindow* m_mainWindow;
    QPixmap m_slideImage;
    QPoint m_selStart;
    QPoint m_selEnd;
    bool m_showSelection = false;
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
    void setSlideImage(const QPixmap& pixmap, int scaleFactor);

    /**
     * Get current scaling coefficient
     */
    int getScaleFactor() const { return m_scaleFactor; }

    // Called by ImageLabel when selection completes
    void onSelectionComplete(const QPoint& pressPos, const QPoint& releasePos);
    void updateStatusBar(const QString& message);

private slots:
    void showSlideMenu();
    void onSelectWorkingDirectory();

private:
    void setupUi();
    void loadModel();
    void showPreview();
    void applyWorkingDirectory(const std::filesystem::path& dir, bool persist);

    // Configuration and model
    Config m_config;
    SlideReader m_slideReader;
    InferenceEngine m_inference;

    // UI elements
    QScrollArea* m_scrollArea = nullptr;
    ImageLabel* m_imageLabel = nullptr;
    QLabel* m_statusLabel = nullptr;

    // Child windows
    std::unique_ptr<MenuWindow> m_menuWindow;

    // Selection state
    QPoint m_pressPos;
    QPoint m_releasePos;
    int m_scaleFactor = 1;
};

} // namespace cytologick
