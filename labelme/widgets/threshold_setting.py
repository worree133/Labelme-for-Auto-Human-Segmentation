import re

from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.logger import logger

import labelme.utils


QT5 = QT_VERSION[0] == "5"


# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


class thresholdDialog(QtWidgets.QDialog):

    def __init__(self):
        super().__init__()


        self.setWindowTitle('labelme')
        self.resize(400, 200)
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(3)
        self.setLayout(layout)
        self.slider = QtWidgets.QSlider(self)
        self.slider.setGeometry(50, 50, 300, 50)
        self.slider.setRange(0, 255)
        self.slider.setValue(127)
        self.slider.setOrientation(1)
        self.slider.valueChanged.connect(self.showMsg)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(40, 150, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Close).setText("Enter")
        self.buttonBox.setObjectName("buttonBox")
        self.buttonBox.rejected.connect(self.reject)  # type: ignore
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(20,20,100,30)
        self.label.setText('Threshold:127' )

    def showMsg(self):
        self.label.setText('Threshold:' + str(self.slider.value()))







