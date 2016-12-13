#ifndef GPU_C_H
#define GPU_C_H

#include <QMainWindow>
#include <QImage>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QFileDialog>

namespace Ui {
class GPU_C;
}

class GPU_C : public QMainWindow
{
    Q_OBJECT

public:
    explicit GPU_C(QWidget *parent = 0);
    ~GPU_C();

private:
    Ui::GPU_C *ui;
    QPixmap image;
    QImage *imageObject;
    QGraphicsScene *scene;

private slots:
    void on_OpenImage_clicked();
    void on_Grey_clicked();
    void on_Blur_clicked();
    void on_Corner_clicked();
    void on_CloseButton_clicked();
};

#endif // GPU_C_H
