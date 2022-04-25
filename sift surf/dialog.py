# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:54:21 2020

@author: Onur
"""

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from cod import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    