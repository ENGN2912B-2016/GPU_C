#-------------------------------------------------
#
# Created by Team GPU_C
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = GPU_C
TEMPLATE = app


SOURCES += main.cpp\
        gpu_c.cpp

HEADERS  += gpu_c.h

FORMS    += gpu_c.ui

RESOURCES += \
    resources.qrc
