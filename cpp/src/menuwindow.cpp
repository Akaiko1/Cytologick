#include "menuwindow.h"
#include "mainwindow.h"
#include "slidereader.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QApplication>
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
    connect(m_slideCombo, QOverload<int>::of(&QComboBox::activated),
            this, &MenuWindow::onSlideSelected);
    connect(m_levelCombo, QOverload<int>::of(&QComboBox::activated),
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

    m_mainWindow->onSlideOpened(slidePath);
    updateLevelList();
}

void MenuWindow::updateLevelList() {
    auto& slideReader = m_mainWindow->getSlideReader();

    m_levelDimensions = slideReader.getAllLevelDimensions();

    m_levelCombo->clear();

    // Show all levels, from lowest-resolution (fast) to highest-resolution (level 0).
    for (int level = static_cast<int>(m_levelDimensions.size()) - 1; level >= 0; --level) {
        auto [w, h] = m_levelDimensions[level];
        const double ds = slideReader.getLevelDownsample(level);
        QString text = QString("Level %1 (%2x%3) ds=%4")
                           .arg(level)
                           .arg(w)
                           .arg(h)
                           .arg(ds, 0, 'f', 3);
        m_levelCombo->addItem(text, level);
    }

    // Auto-load the first (lowest-resolution) level after opening a slide.
    if (m_levelCombo->count() > 0) {
        onLevelSelected(0);
    }
}

void MenuWindow::onLevelSelected(int index) {
    if (index < 0 || index >= m_levelCombo->count()) {
        return;
    }

    bool ok = false;
    const int level = m_levelCombo->itemData(index).toInt(&ok);
    if (!ok) return;

    QApplication::setOverrideCursor(Qt::WaitCursor);
    m_mainWindow->loadSlideLevel(level);
    QApplication::restoreOverrideCursor();
}

} // namespace cytologick
