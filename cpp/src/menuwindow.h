#pragma once

#include <QDialog>
#include <QComboBox>
#include <vector>
#include <filesystem>

namespace cytologick {

class MainWindow;

/**
 * Slide selection menu dialog
 * Port of Python Menu class
 */
class MenuWindow : public QDialog {
    Q_OBJECT

public:
    explicit MenuWindow(MainWindow* parent);
    ~MenuWindow() override = default;

private slots:
    void onSlideSelected(int index);
    void onLevelSelected(int index);

private:
    void setupUi();
    void findSlides();
    void updateLevelList();

    MainWindow* m_mainWindow;

    QComboBox* m_slideCombo = nullptr;
    QComboBox* m_levelCombo = nullptr;

    std::vector<std::filesystem::path> m_slideList;
    std::vector<std::pair<int64_t, int64_t>> m_levelDimensions;
};

} // namespace cytologick
