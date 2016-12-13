#include "gpu_c.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GPU_C w;
    w.show();

    return a.exec();
}
