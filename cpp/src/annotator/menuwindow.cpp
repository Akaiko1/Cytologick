#include "menuwindow.h"
#include "window.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QMessageBox>
#include <QApplication>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <set>

namespace fs = std::filesystem;

namespace cytologick {

AnnotatorMenuWindow::AnnotatorMenuWindow(AnnotatorWindow* parent)
    : QDialog(parent)
    , m_mainWindow(parent)
{
    setupUi();
    startScanSlides();
}

void AnnotatorMenuWindow::setupUi() {
    setWindowTitle("Select Slide");
    setMinimumWidth(500);
    setStyleSheet(
        "QDialog { background: #f8fafc; color: #0f172a; }"
        "QGroupBox {"
        "  border: 1px solid #d8dee6;"
        "  border-radius: 8px;"
        "  margin-top: 10px;"
        "  padding-top: 8px;"
        "  background: #ffffff;"
        "  font-weight: 600;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px;"
        "}"
        "QLabel { color: #0f172a; }"
        "QComboBox, QPushButton { font-size: 12px; }"
        "QComboBox {"
        "  border: 1px solid #cbd5e1;"
        "  border-radius: 6px;"
        "  padding: 5px 6px;"
        "  background: #ffffff;"
        "  color: #0f172a;"
        "}"
        "QComboBox QAbstractItemView {"
        "  background: #ffffff;"
        "  color: #0f172a;"
        "  selection-background-color: #dbeafe;"
        "  selection-color: #0f172a;"
        "}"
        "QPushButton {"
        "  border: 1px solid #cbd5e1;"
        "  border-radius: 6px;"
        "  padding: 6px 10px;"
        "  background: #f8fafc;"
        "}"
        "QPushButton:hover { background: #eef2ff; }"
    );

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(10);

    m_status = new QLabel("Scanning for slides...", this);
    m_status->setStyleSheet("background: #e2e8f0; color: #0f172a; border-radius: 6px; padding: 6px 8px;");
    layout->addWidget(m_status);

    QGroupBox* sourceBox = new QGroupBox("Slide Source", this);
    QFormLayout* sourceLayout = new QFormLayout(sourceBox);
    sourceLayout->setContentsMargins(8, 8, 8, 8);
    sourceLayout->setSpacing(8);

    m_slideCombo = new QComboBox(sourceBox);
    m_slideCombo->setMinimumWidth(440);
    sourceLayout->addRow("Slide:", m_slideCombo);

    m_levelCombo = new QComboBox(sourceBox);
    m_levelCombo->setMinimumWidth(260);
    sourceLayout->addRow("Zoom Level:", m_levelCombo);
    layout->addWidget(sourceBox);

    QPushButton* closeBtn = new QPushButton("Close", this);
    closeBtn->setMinimumWidth(90);
    connect(closeBtn, &QPushButton::clicked, this, &QDialog::hide);
    QHBoxLayout* actionsLayout = new QHBoxLayout();
    actionsLayout->addStretch(1);
    actionsLayout->addWidget(closeBtn);
    layout->addLayout(actionsLayout);

    // Only react to explicit user choice. Using currentIndexChanged causes
    // expensive slide open / preview load during initial population, which
    // looks like the app "hangs" on startup for large slide directories.
    connect(m_slideCombo, QOverload<int>::of(&QComboBox::activated),
            this, &AnnotatorMenuWindow::onSlideSelected);
    connect(m_levelCombo, QOverload<int>::of(&QComboBox::activated),
            this, &AnnotatorMenuWindow::onLevelSelected);
}

void AnnotatorMenuWindow::startScanSlides() {
    if (!m_mainWindow) return;

    m_slideCombo->clear();
    m_slideCombo->addItem("Scanning...");
    m_slideCombo->setEnabled(false);
    m_levelCombo->clear();
    m_levelCombo->setEnabled(false);
    if (m_status) m_status->setText("Scanning for slides...");

    // Copy paths for worker thread.
    const auto cfg = m_mainWindow->getConfig();
    const fs::path hdd = cfg.hddSlides;
    const fs::path dir = cfg.slideDir;

    auto future = QtConcurrent::run([hdd, dir]() -> std::vector<fs::path> {
        std::set<fs::path> uniqueSlides;
        if (!hdd.empty() && fs::exists(hdd)) {
            auto slides = SlideReader::findSlides(hdd, true);
            uniqueSlides.insert(slides.begin(), slides.end());
        }
        if (!dir.empty() && fs::exists(dir)) {
            auto slides = SlideReader::findSlides(dir, true);
            uniqueSlides.insert(slides.begin(), slides.end());
        }
        std::vector<fs::path> out(uniqueSlides.begin(), uniqueSlides.end());
        std::sort(out.begin(), out.end());
        return out;
    });

    connect(&m_scanWatcher, &QFutureWatcher<std::vector<fs::path>>::finished,
            this, &AnnotatorMenuWindow::onScanFinished);
    m_scanWatcher.setFuture(future);
}

void AnnotatorMenuWindow::onScanFinished() {
    m_slideList = m_scanWatcher.result();

    m_slideCombo->clear();
    for (const auto& p : m_slideList) {
        m_slideCombo->addItem(QString::fromStdString(p.string()));
    }

    const bool hasSlides = !m_slideList.empty();
    m_slideCombo->setEnabled(hasSlides);
    m_levelCombo->setEnabled(hasSlides);

    if (m_status) {
        m_status->setText(hasSlides ? QString("Found %1 slide(s).").arg(m_slideList.size())
                                    : "No slides found (check config.yaml paths).");
    }
}

void AnnotatorMenuWindow::onSlideSelected(int index) {
    if (!m_mainWindow) return;
    if (index < 0 || index >= static_cast<int>(m_slideList.size())) return;

    const auto& slidePath = m_slideList[index];
    auto& reader = m_mainWindow->getSlideReader();

    if (!reader.open(slidePath)) {
        QMessageBox::warning(this, "Error",
            QString("Failed to open slide:\n%1\n\nError: %2")
                .arg(QString::fromStdString(slidePath.string()))
                .arg(QString::fromStdString(reader.getLastError()))
        );
        return;
    }

    m_mainWindow->onSlideOpened(slidePath);
    updateLevelList();
}

void AnnotatorMenuWindow::updateLevelList() {
    if (!m_mainWindow) return;
    auto& reader = m_mainWindow->getSlideReader();
    m_levelDimensions = reader.getAllLevelDimensions();

    m_levelCombo->clear();
    // Show all available pyramid levels. Order from lowest-resolution (fastest)
    // to highest-resolution (slow/large).
    for (int level = static_cast<int>(m_levelDimensions.size()) - 1; level >= 0; --level) {
        auto [w, h] = m_levelDimensions[level];
        const double ds = reader.getLevelDownsample(level);
        const QString text = QString("Level %1 (%2x%3) ds=%4")
            .arg(level)
            .arg(w)
            .arg(h)
            .arg(ds, 0, 'f', 3);
        m_levelCombo->addItem(text, level);
    }

    // Auto-load the lowest-resolution level after the user explicitly chose a slide.
    if (m_levelCombo->count() > 0) {
        const int level = m_levelCombo->itemData(0).toInt();
        QApplication::setOverrideCursor(Qt::WaitCursor);
        m_mainWindow->loadSlideLevel(level);
        QApplication::restoreOverrideCursor();
    }
}

void AnnotatorMenuWindow::onLevelSelected(int index) {
    if (!m_mainWindow) return;
    if (index < 0 || index >= m_levelCombo->count()) return;
    bool ok = false;
    const int level = m_levelCombo->itemData(index).toInt(&ok);
    if (!ok) return;

    QApplication::setOverrideCursor(Qt::WaitCursor);
    m_mainWindow->loadSlideLevel(level);
    QApplication::restoreOverrideCursor();
}

} // namespace cytologick
