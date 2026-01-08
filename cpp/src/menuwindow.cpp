#include "menuwindow.h"
#include "mainwindow.h"
#include "slidereader.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <QMessageBox>
#include <QApplication>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <set>

namespace fs = std::filesystem;

namespace cytologick {

MenuWindow::MenuWindow(MainWindow* parent)
    : QDialog(parent)
    , m_mainWindow(parent)
{
    setupUi();
    findSlides();
}

void MenuWindow::setupUi() {
    setWindowTitle("Select Slide");
    setMinimumWidth(400);

    QVBoxLayout* layout = new QVBoxLayout(this);

    // Slide selection
    QLabel* slideLabel = new QLabel("Slide:", this);
    m_slideCombo = new QComboBox(this);
    m_slideCombo->setMinimumWidth(350);

    // Zoom level selection
    QLabel* levelLabel = new QLabel("Zoom Level:", this);
    m_levelCombo = new QComboBox(this);
    m_levelCombo->setMinimumWidth(200);

    layout->addWidget(slideLabel);
    layout->addWidget(m_slideCombo);
    layout->addWidget(levelLabel);
    layout->addWidget(m_levelCombo);

    // Connect signals
    connect(m_slideCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MenuWindow::onSlideSelected);
    connect(m_slideCombo, QOverload<int>::of(&QComboBox::activated),
            this, &MenuWindow::onSlideSelected);
    connect(m_levelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MenuWindow::onLevelSelected);
}

void MenuWindow::findSlides() {
    auto& config = m_mainWindow->getConfig();

    // Collect slides from HDD_SLIDES and SLIDE_DIR (matching Python gui.py)
    std::set<fs::path> uniqueSlides;

    // Search in HDD_SLIDES first
    if (!config.hddSlides.empty() && fs::exists(config.hddSlides)) {
        auto slides = SlideReader::findSlides(config.hddSlides, true);
        uniqueSlides.insert(slides.begin(), slides.end());
    }

    // Search in SLIDE_DIR
    if (!config.slideDir.empty() && fs::exists(config.slideDir)) {
        auto slides = SlideReader::findSlides(config.slideDir, true);
        uniqueSlides.insert(slides.begin(), slides.end());
    }

    m_slideList.assign(uniqueSlides.begin(), uniqueSlides.end());
    std::sort(m_slideList.begin(), m_slideList.end());

    // Populate combo box
    m_slideCombo->clear();
    for (const auto& path : m_slideList) {
        m_slideCombo->addItem(QString::fromStdString(path.string()));
    }

    if (m_slideList.empty()) {
        m_slideCombo->addItem("No slides found");
        m_slideCombo->setEnabled(false);
    }
}

void MenuWindow::onSlideSelected(int index) {
    if (index < 0 || index >= static_cast<int>(m_slideList.size())) {
        return;
    }

    const auto& slidePath = m_slideList[index];
    auto& slideReader = m_mainWindow->getSlideReader();

    if (!slideReader.open(slidePath)) {
        QMessageBox::warning(this, "Error",
            QString("Failed to open slide:\n%1\n\nError: %2")
                .arg(QString::fromStdString(slidePath.string()))
                .arg(QString::fromStdString(slideReader.getLastError()))
        );
        return;
    }

    updateLevelList();
}

void MenuWindow::updateLevelList() {
    auto& slideReader = m_mainWindow->getSlideReader();

    m_levelDimensions = slideReader.getAllLevelDimensions();

    m_levelCombo->clear();

    // Limit to first 4 levels to prevent memory issues (same as Python)
    int maxLevels = std::min(4, static_cast<int>(m_levelDimensions.size()));

    for (int i = 0; i < maxLevels; ++i) {
        auto [w, h] = m_levelDimensions[i];
        QString text = QString("Level %1 (%2x%3)").arg(i).arg(w).arg(h);
        m_levelCombo->addItem(text);
    }
}

void MenuWindow::onLevelSelected(int index) {
    if (index < 0 || index >= static_cast<int>(m_levelDimensions.size())) {
        return;
    }

    auto& slideReader = m_mainWindow->getSlideReader();

    // Calculate position from the end (higher index = more zoom)
    int position = static_cast<int>(m_levelDimensions.size()) - index - 1;

    // Clamp position
    position = std::max(0, std::min(position, static_cast<int>(m_levelDimensions.size()) - 1));

    // Calculate scaling coefficient
    int scaleFactor = 1;
    if (m_levelDimensions[0].first > 0 && m_levelDimensions[position].first > 0) {
        scaleFactor = static_cast<int>(
            m_levelDimensions[0].first / m_levelDimensions[position].first
        );
    }

    // Read slide at selected level
    auto [width, height] = m_levelDimensions[position];

    QApplication::setOverrideCursor(Qt::WaitCursor);
    cv::Mat region = slideReader.readRegionRGB(0, 0, position, width, height);
    QApplication::restoreOverrideCursor();

    if (region.empty()) {
        QMessageBox::warning(this, "Error", "Failed to read slide region");
        return;
    }

    // Convert to QPixmap
    cv::Mat rgbSwapped;
    cv::cvtColor(region, rgbSwapped, cv::COLOR_RGB2BGR);
    cv::imwrite("gui_temp.bmp", rgbSwapped);

    QImage qImage(region.data, region.cols, region.rows, region.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qImage.copy());

    m_mainWindow->setSlideImage(pixmap, scaleFactor);
}

} // namespace cytologick
