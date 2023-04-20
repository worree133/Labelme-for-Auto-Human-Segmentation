import sys
import os
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, QPoint
from qtpy.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget, QShortcut, QAction, QToolBar
from qtpy.QtGui import QPixmap, QPainter, QPen, QColor, QPainterPath, QKeySequence
from labelme import __appname__

class LayerItem(QtWidgets.QGraphicsRectItem):
    HumanDrawState, BackgroundDrawState, HumanFloodfillState, BackgroundFloodfillState = range(4)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_state = LayerItem.HumanFloodfillState
        self.save = []
        self.m_line_draw = QtCore.QLineF()
        self.m_pixmap = QtGui.QPixmap()
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.setOpacity(0.5)

    def reset(self,image):
        self.image = image
        r = self.parentItem().pixmap().rect()
        self.setRect(QtCore.QRectF(r))
        self.m_pixmap = QtGui.QPixmap(image)




    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.save()
        painter.drawPixmap(QtCore.QPoint(), self.m_pixmap)
        painter.restore()



    def mousePressEvent(self, event):
        if self.current_state == LayerItem.HumanFloodfillState:
            self.save.append(self.m_pixmap.copy())
            self.fill(event.pos(),QtGui.QColor(125,3,67,255))
        elif self.current_state == LayerItem.BackgroundFloodfillState:
            self.save.append(self.m_pixmap.copy())
            self.fill(event.pos(),QtGui.QColor(109,83,104,255))
        elif self.current_state == LayerItem.HumanDrawState or LayerItem.BackgroundDrawState:
            self.m_line_draw.setP1(event.pos())
            self.m_line_draw.setP2(event.pos())
        super().mousePressEvent(event)
        event.accept()



    def fill(self, pos, color):

        x = pos.x()
        y = pos.y()

        # Get our target color from origin.

        image = self.m_pixmap.toImage()

        w, h = image.width(), image.height()

        # Get our target color from origin.
        target_color = image.pixel(x,y)
        have_seen = set()
        queue = [(x, y)]


        def get_cardinal_points(have_seen, center_pos):
            points = []
            cx, cy = center_pos
            for x, y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                xx, yy = cx + x, cy + y
                if (xx >= 0 and xx < w and
                        yy >= 0 and yy < h and
                        (xx, yy) not in have_seen):
                    points.append((xx, yy))
                    have_seen.add((xx, yy))

            return points

        # Now perform the search and fill.
        p = QPainter(self.m_pixmap)
        p.setPen(color)

        while queue:
            x, y = queue.pop()
            if abs(image.pixel(x, y) - target_color) < 1000000:
                p.drawPoint(QPoint(x, y))
                # Prepend to the queue
                queue[0:0] = get_cardinal_points(have_seen, (x, y))
                # or append,
                # queue.extend(get_cardinal_points(have_seen, (x, y))

        self.update()

    def mouseMoveEvent(self, event):

        if self.current_state == LayerItem.HumanDrawState or self.current_state == LayerItem.BackgroundDrawState:

            self.m_line_draw.setP2(event.pos())
            self._draw_line(
                self.m_line_draw, QtGui.QPen(self.pen_color, self.pen_thickness)
            )
            self.m_line_draw.setP1(event.pos())
        super().mouseMoveEvent(event)

    def _draw_line(self, line, pen):
        painter = QtGui.QPainter(self.m_pixmap)
        painter.setPen(pen)
        painter.drawLine(line)
        painter.end()
        self.update()

    def undo(self):

        if len(self.save)>0:
            self.m_pixmap = self.save.pop()
            self.update()

    @property
    def pen_thickness(self):
        return self._pen_thickness

    @pen_thickness.setter
    def pen_thickness(self, thickness):
        self._pen_thickness = thickness

    @property
    def pen_color(self):
        return self._pen_color

    @pen_color.setter
    def pen_color(self, color):
        self._pen_color = color

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, state):
        self._current_state = state


class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.background_item = QtWidgets.QGraphicsPixmapItem()
        self.foreground_item = LayerItem(self.background_item)
        self.scene().addItem(self.background_item)

    def set_image(self, image1, image2):

        self.scene().setSceneRect(
            QtCore.QRectF(QtCore.QPointF(), QtCore.QSizeF(image1.size()))
        )
        self.background_item.setPixmap(image2)
        self.foreground_item.reset(image1)
        self.fitInView(self.background_item, QtCore.Qt.KeepAspectRatio)
        self.centerOn(self.background_item)


    def wheelEvent(self, event):

        if event.modifiers() & Qt.ControlModifier:
            factor = 1.1
            if event.angleDelta().y() < 0:
                factor = 0.9
            view_pos = event.pos()
            scene_pos = self.mapToScene(view_pos)
            self.centerOn(scene_pos)
            self.scale(factor, factor)
            delta = self.mapToScene(view_pos) - self.mapToScene(self.viewport().rect().center())
            self.centerOn(scene_pos - delta)
        else:
            bar = self.verticalScrollBar()
            units = -0.05*event.angleDelta().y()
            value = bar.value() + bar.singleStep() * units
            bar.setValue(int(value))



class segDrawing(QtWidgets.QMainWindow):
    
    def __init__(self, filename=None, image=None, points=None, ori_size=None, parent=None):
        super().__init__(parent)
        self.basedir = None
        self.image = QtGui.QPixmap(image)
        self.filename = filename
        self.points = points
        self.ori_size = ori_size
        menu = self.menuBar().addMenu(self.tr("File"))
        self.open_dir = menu.addAction(self.tr("Open directory..."))
        self.open_dir.triggered.connect(self.openDirDialog)
        human_group = QtWidgets.QGroupBox(self.tr("Human"))
        background_group = QtWidgets.QGroupBox(self.tr("Background"))
        self.close_button = QtWidgets.QPushButton('Submit', clicked=self.submit)

        #color = QtCore.Qt.gray
        self.human_pen_slider = QtWidgets.QSlider(
            QtCore.Qt.Horizontal,
            minimum=3,
            maximum=51,
            value=5,
            focusPolicy=QtCore.Qt.StrongFocus,
            tickPosition=QtWidgets.QSlider.TicksBothSides,
            tickInterval=1,
            singleStep=1,
            valueChanged=self.onThicknessChanged,
        )

        self.background_pen_slider = QtWidgets.QSlider(
            QtCore.Qt.Horizontal,
            minimum=3,
            maximum=51,
            value=5,
            focusPolicy=QtCore.Qt.StrongFocus,
            tickPosition=QtWidgets.QSlider.TicksBothSides,
            tickInterval=1,
            singleStep=1,
            valueChanged=self.onThicknessChanged,
        )

        self.pendraw_human_checkbox = QtWidgets.QCheckBox(
            self.tr("Pen drawing"))

        self.floodfill_human_checkbox = QtWidgets.QCheckBox(
            self.tr("Floodfill"))
        self.floodfill_human_checkbox.setChecked(True)
        self.pendraw_background_checkbox = QtWidgets.QCheckBox(
            self.tr("Pen drawing"))
        self.floodfill_background_checkbox = QtWidgets.QCheckBox(
            self.tr("Floodfill"))

        self.pendraw_human_checkbox.clicked.connect(
            lambda checked: (checked, self.human_pendraw(), self.floodfill_human_checkbox.setChecked(
                False), self.floodfill_background_checkbox.setChecked(
                False), self.pendraw_background_checkbox.setChecked(False)))

        self.floodfill_human_checkbox.clicked.connect(
            lambda checked: (checked, self.humanfill(), self.floodfill_background_checkbox.setChecked(False),
            self.pendraw_background_checkbox.setChecked(False), self.pendraw_human_checkbox.setChecked(False)))

        self.pendraw_background_checkbox.clicked.connect(
            lambda checked: (checked, self.background_pendraw(), self.floodfill_background_checkbox.setChecked(
                False), self.pendraw_human_checkbox.setChecked(
                False), self.floodfill_human_checkbox.setChecked(False)))

        self.floodfill_background_checkbox.clicked.connect(
            lambda checked: (checked, self.backgroundfill(), self.pendraw_background_checkbox.setChecked(
                False), self.pendraw_human_checkbox.setChecked(
                False), self.floodfill_human_checkbox.setChecked(False)))


        self.view = GraphicsView()
        self.view.foreground_item.pen_thickness = self.human_pen_slider.value()
        #self.undo = QAction(self.tr("UndoSeg"),self)
        #self.undo.setShortcut(QKeySequence.Undo)
        #self.undo.setShortcutContext(Qt.ApplicationShortcut)
        #self.toolbar.addAction(self.undo)
        #self.shortcut_undo = QShortcut(QKeySequence(self.tr('Ctrl+Z',"UndoSeg")), parent)
        self.shortcut_undo = QShortcut(QKeySequence.Undo, self)
        self.shortcut_undo.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_undo.activated.connect(self.view.foreground_item.undo)
        #self.shortcut_undo.installEventFilter(self)
        #self.undo.activated.connect(self.view.foreground_item.undo)
        #self.undo.activated.connect(self.test)
        self.OpenFile()


        # layouts
        human_lay = QtWidgets.QFormLayout(human_group)
        human_lay.addWidget(self.pendraw_human_checkbox)
        human_lay.addRow(self.tr("Pen thickness"), self.human_pen_slider)
        human_lay.addWidget(self.floodfill_human_checkbox)

        background_lay = QtWidgets.QFormLayout(background_group)
        background_lay.addWidget(self.pendraw_background_checkbox)
        background_lay.addRow(self.tr("Pen thickness"), self.background_pen_slider)
        background_lay.addWidget(self.floodfill_background_checkbox)


        vlay = QtWidgets.QVBoxLayout()
        vlay.addWidget(human_group)
        vlay.addWidget(background_group)
        vlay.addWidget(self.close_button)
        vlay.addStretch()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        lay = QtWidgets.QHBoxLayout(central_widget)
        lay.addLayout(vlay, stretch=0)
        lay.addWidget(self.view, stretch=1)

        self.resize(1200, 600)

    def checkimgsize(self,seg,image):

        if seg.size()!=self.ori_size:
            seg = seg.scaled(self.ori_size)

        return seg

    def openDirDialog(self, _value=False, dirpath=None):

        defaultOpenDirPath = dirpath if dirpath else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.basedir = targetDirPath
        self.filename = os.path.basename(self.filename)
        file, ext = os.path.splitext(self.filename)

        self.filename = os.path.join(self.basedir,file)
        self.OpenFile()


    def OpenFile(self):

        if self.filename:

            pixmap = QtGui.QPixmap(self.filename)
            pixmap = self.checkimgsize(pixmap, self.image)
            x1,y1 = self.points[0]
            x2,y2 = self.points[1]
            pixmap = pixmap.copy(x1, y1, abs(x2 - x1), abs(y2 - y1))

            if pixmap.isNull():
                QtWidgets.QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % self.filename
                )
                return
            self.view.set_image(pixmap,self.image)


    @QtCore.Slot()
    def submit(self):
        self.close()


    def humanfill(self):
        self.view.foreground_item.current_state = (
            LayerItem.HumanFloodfillState
        )


    def human_pendraw(self):
        self.view.foreground_item.current_state = (
            LayerItem.HumanDrawState
        )
        self.view.foreground_item.pen_color = QtGui.QColor(125, 3, 67, 255)



    def backgroundfill(self):
        self.view.foreground_item.current_state = (
            LayerItem.BackgroundFloodfillState
        )


    def background_pendraw(self):
        self.view.foreground_item.current_state = (
            LayerItem.BackgroundDrawState
        )
        self.view.foreground_item.pen_color = QtGui.QColor(109, 83, 104, 255)

    @QtCore.Slot(int)
    def onThicknessChanged(self, value):
        self.view.foreground_item.pen_thickness = value



