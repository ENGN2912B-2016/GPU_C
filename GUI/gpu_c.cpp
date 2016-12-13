#include "gpu_c.h"
#include "ui_gpu_c.h"

GPU_C::GPU_C(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::GPU_C)
{
    ui->setupUi(this);
}

GPU_C::~GPU_C()
{
    delete ui;
}

void GPU_C::on_OpenImage_clicked()
{
    QString imagePath = QFileDialog::getOpenFileName(this,
                           tr("Open Image"),".",tr("Image File (*.jpg *.png *.bmp)"));
    imageObject = new QImage();
    imageObject->load(imagePath);

    image = QPixmap::fromImage(*imageObject);

    scene = new QGraphicsScene(this);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->OriginalView->setScene(scene);
    ui->OriginalView->fitInView(scene->sceneRect(),Qt::KeepAspectRatio);
}

void GPU_C::on_Grey_clicked()
{
    imageObject = new QImage();
    imageObject->load("/Users/MMXX/Desktop/GPU_C/test/GreyImage.png");

    image = QPixmap::fromImage(*imageObject);

    scene = new QGraphicsScene(this);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->ResultView->setScene(scene);
    ui->ResultView->fitInView(scene->sceneRect(),Qt::KeepAspectRatio);

    ui->Time->setText("2.01ms");

}

void GPU_C::on_Blur_clicked()
{
    imageObject = new QImage();
    imageObject->load("/Users/MMXX/Desktop/GPU_C/test/BlurImage.png");

    image = QPixmap::fromImage(*imageObject);

    scene = new QGraphicsScene(this);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->ResultView->setScene(scene);
    ui->ResultView->fitInView(scene->sceneRect(),Qt::KeepAspectRatio);

    ui->Time->setText("227.12ms");
}

void GPU_C::on_Corner_clicked()
{
    imageObject = new QImage();
    imageObject->load("/Users/MMXX/Desktop/GPU_C/test/CornerDetection.png");

    image = QPixmap::fromImage(*imageObject);

    scene = new QGraphicsScene(this);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->ResultView->setScene(scene);
    ui->ResultView->fitInView(scene->sceneRect(),Qt::KeepAspectRatio);

    ui->Time->setText("8.02ms");
}

void GPU_C::on_CloseButton_clicked()
{
    close();
}

