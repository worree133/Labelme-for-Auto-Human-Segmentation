import sys
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, QPoint
from qtpy.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget
from qtpy.QtGui import QPixmap, QPainter, QPen, QColor, QPainterPath


class LayerItem(QtWidgets.QGraphicsRectItem):
    DrawState, EraseState = range(2)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_state = LayerItem.DrawState
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))

        self.m_line_eraser = QtCore.QLineF()
        self.m_line_draw = QtCore.QLineF()
        self.m_pixmap = QtGui.QPixmap()

    def reset(self):
        r = self.parentItem().pixmap().rect()
        self.setRect(QtCore.QRectF(r))
        self.m_pixmap = QtGui.QPixmap(self.parentItem().pixmap().size())
        self.m_pixmap.fill(QtCore.Qt.transparent)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.save()
        painter.drawPixmap(QtCore.QPoint(), self.m_pixmap)
        painter.restore()



    def mousePressEvent(self, event):
        if self.current_state == LayerItem.EraseState:
            self._clear(event.pos().toPoint())
        elif event.button() == QtCore.Qt.RightButton:
            self.fill(event.pos())
        elif self.current_state == LayerItem.DrawState:
            self.m_line_draw.setP1(event.pos())
            self.m_line_draw.setP2(event.pos())
        super().mousePressEvent(event)
        event.accept()



    def fill(self, pos):

        x = pos.x()
        y = pos.y()

        # Get our target color from origin.

        image = self.m_pixmap.toImage()

        w, h = image.width(), image.height()

        # Get our target color from origin.
        target_color = image.pixel(x, y)

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
        p.setPen(QtGui.QColor(255,255,255,60))

        while queue:
            x, y = queue.pop()
            if image.pixel(x, y) == target_color:
                p.drawPoint(QPoint(x, y))
                # Prepend to the queue
                queue[0:0] = get_cardinal_points(have_seen, (x, y))
                # or append,
                # queue.extend(get_cardinal_points(have_seen, (x, y))

        self.update()

    def mouseMoveEvent(self, event):
        if self.current_state == LayerItem.EraseState:
            self._clear(event.pos().toPoint())
        elif self.current_state == LayerItem.DrawState:
            self.m_line_draw.setP2(event.pos())
            self._draw_line(
                self.m_line_draw, QtGui.QPen(self.pen_color, self.pen_thickness)
            )
            self.m_line_draw.setP1(event.pos())
        super().mouseMoveEvent(event)

    def _draw_line(self, line, pen):
        painter = QtGui.QPainter(self.m_pixmap)
        painter.setOpacity(0.2)
        painter.setPen(pen)
        painter.drawLine(line)
        painter.end()
        self.update()

    def _clear(self, pos):
        painter = QtGui.QPainter(self.m_pixmap)
        r = QtCore.QRect(QtCore.QPoint(), 10 * QtCore.QSize())
        r.moveCenter(pos)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.eraseRect(r)
        painter.end()
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

    def set_image(self, image):
        self.scene().setSceneRect(
            QtCore.QRectF(QtCore.QPointF(), QtCore.QSizeF(image.size()))
        )
        self.background_item.setPixmap(image)
        self.foreground_item.reset()
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



class trimapDrawing (QtWidgets.QMainWindow):
    def __init__(self, filename=None, parent=None):
        super().__init__(parent)

        pen_group = QtWidgets.QGroupBox(self.tr("Pen settings"))
        eraser_group = QtWidgets.QGroupBox(self.tr("Eraser"))
        self.close_button = QtWidgets.QPushButton('Submit', clicked=self.submit)
        color = QtGui.QColor(128,128,128,40)
        #color = QtCore.Qt.gray
        self.pen_slider = QtWidgets.QSlider(
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

        self.eraser_checkbox = QtWidgets.QCheckBox(
            self.tr("Eraser"), stateChanged=self.onStateChanged
        )

        self.view = GraphicsView()
        self.view.foreground_item.pen_thickness = self.pen_slider.value()
        self.view.foreground_item.pen_color = color

        if filename:
            pixmap = QtGui.QPixmap(filename)
            if pixmap.isNull():
                QtWidgets.QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % filename
                )
                return
            self.view.set_image(pixmap)

        # layouts
        pen_lay = QtWidgets.QFormLayout(pen_group)
        pen_lay.addRow(self.tr("Pen thickness"), self.pen_slider)

        eraser_lay = QtWidgets.QVBoxLayout(eraser_group)
        eraser_lay.addWidget(self.eraser_checkbox)

        vlay = QtWidgets.QVBoxLayout()
        vlay.addWidget(pen_group)
        vlay.addWidget(eraser_group)
        vlay.addWidget(self.close_button)
        vlay.addStretch()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        lay = QtWidgets.QHBoxLayout(central_widget)
        lay.addLayout(vlay, stretch=0)
        lay.addWidget(self.view, stretch=1)

        self.resize(1200, 600)


    @QtCore.Slot()
    def submit(self):
        self.close()



    @QtCore.Slot(int)
    def onStateChanged(self, state):
        self.view.foreground_item.current_state = (
            LayerItem.EraseState
            if state == QtCore.Qt.Checked
            else LayerItem.DrawState
        )

    @QtCore.Slot(int)
    def onThicknessChanged(self, value):
        self.view.foreground_item.pen_thickness = value



