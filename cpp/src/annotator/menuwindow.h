#pragma once

#include <QDialog>
#include <QComboBox>
#include <QLabel>
#include <QFutureWatcher>
#include <vector>
#include <filesystem>

namespace cytologick {

class AnnotatorWindow;

class AnnotatorMenuWindow : public QDialog {
    Q_OBJECT

public:
    explicit AnnotatorMenuWindow(AnnotatorWindow* parent);
    ~AnnotatorMenuWindow() override = default;

private slots:
    void onSlideSelected(int index);
    void onLevelSelected(int index);
    void onScanFinished();

private:
    void setupUi();
    void startScanSlides();
    void updateLevelList();

    AnnotatorWindow* m_mainWindow;
    QComboBox* m_slideCombo = nullptr;
    QComboBox* m_levelCombo = nullptr;
    QLabel* m_status = nullptr;

    std::vector<std::filesystem::path> m_slideList;
    std::vector<std::pair<int64_t, int64_t>> m_levelDimensions;

    QFutureWatcher<std::vector<std::filesystem::path>> m_scanWatcher;
};

} // namespace cytologick
