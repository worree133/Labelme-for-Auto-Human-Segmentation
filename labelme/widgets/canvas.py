import os.path
import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtGui import QColor
from labelme.label_file import LabelFile
from labelme import QT5
from labelme.shape import Shape
import labelme.utils
from labelme.models.vitae.ViTAE import *
from labelme.convertmask.utils.methods.getmultishapes import *
from labelme.widgets.threshold_setting import thresholdDialog
from qtpy.QtWidgets import *
# TODO(unknown):
# - [maybe] Find optimal epsilon value.
from pathlib import Path
model_path = Path(__file__).parents[1]
CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self,*args, **kwargs ):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(
                    self.double_click
                )
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": True,
                "linestrip": False,
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.lastPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        self.points = None
        self.filename = None
        self.indexrecord = []
        self.count = 0
        self.draw = False
        self.pointsave = None
        self.checksignal = True
        self.orishape = []
        self.groupid = 0
        self.shapehistory = []
        self.addedpoint = []
        self.mattingdict = {}
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

    def getFilename(self, filename):
        self.filename=filename
        self.update()

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.

        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return

        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):

        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair

        else:
            # EDIT -> CREATE

            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None



    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.prevMovePoint = pos
        self.restoreCursor()

        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon","linestrip"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode in ["rectangle", "point", "line"]:
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"


            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [
                    s.copy() for s in self.selectedShapes
                ]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True
        return shape

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes


    def fill(self, pos):
        print(123)
        x = pos.x()
        y = pos.y()

        # Get our target color from origin.

        image = self.pixmap.toImage()

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
        p.setPen(QtGui.QColor(125,3,67,255))

        while queue:
            x, y = queue.pop()
            if image.pixel(x, y) == target_color:
                p.drawPoint(QPoint(x, y))
                # Prepend to the queue
                queue[0:0] = get_cardinal_points(have_seen, (x, y))
                # or append,
                # queue.extend(get_cardinal_points(have_seen, (x, y))
        print('ok')
        self.update()

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():

                if self.createMode == "linestrip":

                    for shape in self.shapes:

                        index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)

                        if index_edge is not None and shape.canAddPoint():
                            p=shape.points.copy()
                            self.orishape.append(p)
                            self.prevhVertex = self.hVertex
                            self.hVertex = None
                            point = self.prevMovePoint
                            self.addedpoint.append(point)
                            shape.insertPoint(index_edge, point)
                            shape.highlightVertex(index_edge, shape.MOVE_VERTEX)

                        index = shape.nearestVertex(pos, self.epsilon / self.scale)
                        shape.highlightVertex(index, shape.MOVE_VERTEX)
             
                        if index is not None:
                 
                            self.changedShape = shape
                            self.shapepoints = shape.points

                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":

                        self.fill(ev.pos())
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "point", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)

                    if self.createMode == "circle":
                        self.current.shape_type = "circle"
                    self.line.points = [pos, pos]
                    self.setHiding()
                    self.drawingPolygon.emit(True)
                    self.update()
            elif self.editing():

                if self.selectedEdge():
                    self.addPointToEdge()

                elif (
                    self.selectedVertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)

                self.prevPoint = pos
                self.repaint()


        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None
                and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos

    def mouseReleaseEvent(self, ev):

        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if (
                not menu.exec_(self.mapToGlobal(ev.pos()))
                and self.selectedShapesCopy
            ):
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )


        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (
                self.shapesBackups[-1][index].points
                != self.shapes[index].points
            ):
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False


    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()

        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (
            self.double_click == "close"
            and self.canCloseShape()
            and len(self.current) > 3
        ):
            self.current.popPoint()
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)

        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape]
                            )
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)

                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def setthreshold(self,img,th=127):
        pix=img.copy()
        pix[pix > th] = 255
        pix[pix != 255] = 0
        return pix

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
        p = QtGui.QPainter(self.m_pixmap)
        p.setPen(QtCore.Qt.white)

        while queue:
            x, y = queue.pop()
            if image.pixel(x, y) == target_color:
                p.drawPoint(QtCore.QPoint(x, y))
                # Prepend to the queue
                queue[0:0] = get_cardinal_points(have_seen, (x, y))
                # or append,
                # queue.extend(get_cardinal_points(have_seen, (x, y))

        self.update()

    def checkpoints(self, pt1, pt2):
        if pt1.x() > pt2.x():

            if pt1.y() > pt2.y():
                x1, y1 = pt2.x(), pt2.y()
                x2, y2 = pt1.x(), pt1.y()
            else:
                x1, y1 = pt2.x(), pt1.y()
                x2 ,y2 = pt1.x(), pt2.y()


        elif pt1.x() < pt2.x():

            if pt1.y() > pt2.y():
                x1, y1 = pt1.x(), pt2.y()
                x2, y2 = pt2.x(), pt1.y()
            else:
                x1, y1 = pt1.x(), pt1.y()
                x2, y2 = pt2.x(), pt2.y()

        return x1, y1, x2, y2

    def getRectcropP3M(self, pixmap, pt1, pt2):

        x1, y1, x2, y2 = self.checkpoints(pt1,pt2)
        self.pointsave = [[x1,y1],[x2,y2]]
        mattingsave = pixmap.copy(x1, y1, abs(x2 - x1), abs(y2 - y1))
        mattingsave = loadVitae(mattingsave)
        self.mattingdict[self.groupid] = mattingsave
        self.lastsave = self.setthreshold(mattingsave)
        lastshape, self.groupid = getMultiShapes(self.lastsave,self.filename,x1,y1,x2,y2,mattingsave,self.groupid)

        self.shapehistory.append(lastshape.copy())
        #painter.drawPixmap(x1, y1, x2 - x1, y2 - y1,pix)

    def getRectcrop(self, pixmap, pt1, pt2):

        x1, y1, x2, y2 = self.checkpoints(pt1,pt2)

        self.pointsave = [[x1,y1],[x2,y2]]
        self.cropimage = pixmap.copy(x1, y1, abs(x2 - x1), abs(y2 - y1))
        self.cropimage = self.cropimage.toImage()

    def setpoint(self, pt1, pt2):
        x3, y3 = self.pointsave[0]
        x4, y4 = self.pointsave[1]
        x1, y1, x2, y2 =self.checkpoints(pt1,pt2)

        if x1>x4 or x3>x2 or y1>y4 or y3>y2:
            #raise ValueError("Wrong Area Selected. ")
            return self.undoLastLine()
        x1 = max(x1,x3)
        y1 = max(y1,y3)
        x2 = min(x2,x4)
        y2 = min(y2,y4)

        return x1, y1, x2, y2, x3, y3



    def setboundary(self):

        if self.createMode == "circle":

            for shape in self.shapes:

                def showboundary():



                    if self.count == 0:

                        self.pix = self.tunesave.copy()
                        self.count += 1

                    else:

                        self.pix = self.tunesave.copy()

                    self.pix = self.tunesave.copy()
                    self.pix[self.pix > self.thdlg.slider.value()] = 255
                    self.pix[self.pix != 255] = 0

                    if len(self.pix) == 0:
                        self.pix[self.pix > self.thdlg.slider.value()] = 255
                        self.pix[self.pix != 255] = 0

                    region = process(self.pix.astype(np.uint8))

                    shape = Shape(shape_type='polygon')

                    shape.points = []

                    x, y = self.newpt[0]

                    for subregion in region:

                        for i in range(0, subregion.shape[0]):
                            a = list(map(add, subregion[i][0].tolist(), [x, y]))
                            shape.points.append(QtCore.QPointF(a[0], a[1]))
                            # self.shapes.append(QtCore.QPointF(a[0], a[1]))
                    self.shapes = [shape]

                    self.repaint()

                def confirmboundary():
                    if self.count == 0:
                        return self.undoLastLine()

                    x1, y1 = self.newpt[0]
                    x2, y2 = self.newpt[1]
                    x3, y3 = self.pointsave[0]

                    self.count = 0
                    #self.lastsave[int(y1-y3):int(y2-y3),int(x1-x3):int(x2-x3)] = self.pix

                    lastshape = getNewShapes(self.pix, group, rectpts, modeltype, self.filename, x1, y1, x2, y2)

                    self.shapehistory.append(lastshape)

                self.thdlg = thresholdDialog()

                rectpts, group, modeltype = self.Checks()

                if self.checksignal == False:
                    return
                else:
                    self.thdlg.slider.valueChanged.connect(showboundary)
                    self.thdlg.buttonBox.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(showboundary)
                    self.thdlg.buttonBox.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(confirmboundary)
                    self.thdlg.exec_()
                break


    def correctShapes(self,correctedpoints):

  
        for pt in self.addedpoint:
            self.indexrecord.append(self.changedShape.nearestVertex(pt, self.epsilon / self.scale))

        a, b = self.indexrecord[0],self.indexrecord[-1]

        c = len(self.shapepoints)

        orishape = []
        nshape = []
        ori = []


        for shape in self.orishape[0]:
            ori.append([shape.x(), shape.y()])
        for shape in self.shapepoints:
            orishape.append([shape.x(), shape.y()])
        for shape in correctedpoints:
            nshape.append([shape.x(), shape.y()])


        if b>a:

            if (b-a) >  (c-b+a):
                nshape.reverse()
                nshape = orishape[a + 1:b] + nshape

            else:
                nshape = orishape[:a] + nshape + orishape[b + 1:]


        else:

            if (a-b) > (c-a+b):
                nshape = orishape[b+1:a] + nshape

            else:
                nshape.reverse()
                nshape = orishape[:b] + nshape + orishape[a + 1:]


        shapesave = outputNewshape(self.filename, nshape, ori)


        self.shapehistory.append(shapesave.copy())



    def Checks(self):

        x1, y1, x2, y2 = self.checkpoints(self.shapes[-1].points[0], self.shapes[-1].points[1])

        rect, groupid, modeltype = findcorrespondingrect(self.filename, x1, y1, x2, y2)

        if rect == None:
            self.checksignal = False
            return

        x3,y3 = rect[0]
        x4,y4 = rect[1]

        x1 = max(x1, x3)
        y1 = max(y1, y3)
        x2 = min(x2, x4)
        y2 = min(y2, y4)

        if modeltype == 'vitae':

            if groupid in self.mattingdict:

                self.tunesave =  self.mattingdict[groupid][int(y1 - y3):int(y2 - y3), int(x1 - x3):int(x2 - x3)]

            else:
                mattingsave = self.pixmap.copy(x3, y3, abs(x3 - x4), abs(y3 - y4))
                mattingsave = loadVitae(mattingsave)
                self.tunesave = mattingsave[int(y1 - y3):int(y2 - y3), int(x1 - x3):int(x2 - x3)]

        elif modeltype == 'matteformer':

            self.tunesave =  self.mattingdict[groupid][int(y1 - y3):int(y2 - y3), int(x1 - x3):int(x2 - x3)]

        self.newpt = [(x1, y1), (x2, y2)]
        self.pointsave =  rect
        self.checksignal = True

        return rect, groupid, modeltype
            # raise ValueError("Wrong Area Selected. ")



    def paintEvent(self, event):

        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)



        p = self._painter

        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        # draw crosshair
        if (
            self._crosshair[self._createMode]
            and self.drawing()
            and self.prevMovePoint
            and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 0, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y()),
                self.width() - 1,
                int(self.prevMovePoint.y()),
            )
            p.drawLine(
                int(self.prevMovePoint.x()),
                0,
                int(self.prevMovePoint.x()),
                self.height() - 1,
            )

        Shape.scale = self.scale

        for shape in self.shapes:

            if (shape.selected or not self._hideBackround) and self.isVisible(
                shape
            ):


                shape.fill = shape.selected or shape == self.hShape
                self.points=shape.paint(p)

                #shape.points = [QtCore.QPointF(1, 1), QtCore.QPointF(500, 500), QtCore.QPointF(900, 100)]
                if shape.shape_type == "rectangle":
                    self.getRectcropP3M(self.pixmap, *self.points)

                elif shape.shape_type in ["point","line"]:
                    self.getRectcrop(self.pixmap, *self.points)
                    self.shapes.remove(shape)

                elif shape.shape_type == "linestrip":
                    self.correctShapes(self.points)
                    self.indexrecord = []
                    self.orishape = []
                    self.addedpoint = []
                #if shape.shape_type == "circle":
                    #self.tuneboundary(*self.points)

        if self.current:
            self.current.paint(p)
            self.line.paint(p)

        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (
            self.fillDrawing()
            and self.createMode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)


        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.setboundary()
        #self.setsegcrop()
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(
                self.selectedShapes, self.prevPoint + offset
            )
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if (
                    self.shapesBackups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):

        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]

        else:
            self.current = None
            self.drawingPolygon.emit(False)

        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        self.m_pixmap = QtGui.QPixmap(self.pixmap.size())
        self.m_pixmap.fill(QtCore.Qt.transparent)
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(self.pixmap)
        #lay = QVBoxLayout(self)
        #lay.addWidget(self.label)
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()

    def undoshape(self):

        (ImgPath, imgname) = os.path.split(self.filename)
        saveJsonPath = ImgPath + os.sep + imgname[:-4] + '.json'

        # with open(savePath, 'w') as f:
        # f.write(str(lastshapes))


        if len(self.shapehistory) > 0:
            with open(saveJsonPath, 'r') as f:
                obj = json.load(f)
                obj['shapes'] = []
                obj['shapes'] = self.shapehistory[-1]
                self.shapehistory.pop()

            j = json.dumps(obj, sort_keys=True, indent=4)

            with open(saveJsonPath, 'w') as f:
                f.write(j)
            return True
        else:
            return False

    def resetdata(self):
        self.shapehistory = []
        self.groupid = 0
        self.mattingdict = {}

