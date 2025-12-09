# gui_app.py
import os

# Force software OpenGL to reduce DLL/init conflicts on some Windows systems
os.environ['QT_OPENGL'] = 'software'

# Try to preload torch before Qt initializes (best-effort; won't abort GUI on failure)
_preloaded_torch = None
try:
    import importlib
    _preloaded_torch = importlib.import_module('torch')
    print(f"Preloaded torch successful: {_preloaded_torch.__version__}")
except Exception as e:
    print(f"Preloading torch failed (will try later at inference): {e}")

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSlider, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QTransform, QCursor
from PyQt5.QtCore import Qt, QPoint
from PIL import Image
import io
import numpy as np
from inference import InferenceEngine


class DraggableLabel(QLabel):
    """
    QLabel 支持拖拽平移（通过控制所属 QScrollArea 的滚动条）。
    这个类不会修改 pixmap 本身，只在鼠标拖动时改变父 scroll area 的滚动位置。
    """
    def __init__(self, parent_scroll: QScrollArea = None):
        super().__init__()
        self.parent_scroll = parent_scroll
        self.setMouseTracking(True)
        self._dragging = False
        self._last_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.parent_scroll is not None:
            self._dragging = True
            self._last_pos = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self.parent_scroll is not None:
            # 计算相对移动，然后移动滚动条（与鼠标移动方向一致的惯性体验）
            delta = event.pos() - self._last_pos
            hbar = self.parent_scroll.horizontalScrollBar()
            vbar = self.parent_scroll.verticalScrollBar()
            # 减去 delta 使得拖拽动作看起来像抓住画布移动
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            self._last_pos = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # 将滚轮事件传给父控件（ScrollArea）以便在有滚动条时正常滚动。
        if self.parent_scroll is not None:
            QApplication.sendEvent(self.parent_scroll, event)
        else:
            super().wheelEvent(event)


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cat vs Dog Classifier')
        # 使用可拖拽的 QLabel
        self.img_label = DraggableLabel(None)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(400, 300)

        # 不把 QLabel 设置为 scaledContents（我们用 pixmap 缩放/变换）
        self.img_label.setScaledContents(False)

        # 把 label 放到 QScrollArea 里以支持平移（通过滚动条）
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.img_label)
        self.scroll.setWidgetResizable(False)  # 我们会在每次设置 pixmap 时手动设置 label 大小
        self.img_label.parent_scroll = self.scroll  # 让 label 知道它的 scroll

        self.open_btn = QPushButton('Open Image')
        self.open_btn.clicked.connect(self.open_image)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_display)

        # 缩放百分比标签
        self.zoom_label = QLabel('100%')
        self.zoom_slider.valueChanged.connect(self.update_zoom_label)

        self.rotate_left_btn = QPushButton('Rotate Left')
        self.rotate_left_btn.clicked.connect(self.rotate_left)
        self.rotate_right_btn = QPushButton('Rotate Right')
        self.rotate_right_btn.clicked.connect(self.rotate_right)
        self.infer_btn = QPushButton('Run Inference')
        self.infer_btn.clicked.connect(self.run_inference)
        self.result_label = QLabel('Result: -')

        controls = QHBoxLayout()
        controls.addWidget(self.open_btn)
        controls.addWidget(self.rotate_left_btn)
        controls.addWidget(self.rotate_right_btn)
        controls.addWidget(self.infer_btn)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel('Zoom:'))
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_label)

        layout = QVBoxLayout()
        # 将 scroll area（包含 label）放到界面上
        layout.addWidget(self.scroll)
        layout.addLayout(zoom_layout)
        layout.addLayout(controls)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        # state
        self.pil_image = None
        self.qpix_original = None  # 原始 QPixmap（由 pil2pixmap 产生）
        self.angle = 0
        self.engine = None

        # 平移/缩放状态（如果需要可以扩展）
        # self.offset = QPoint(0, 0)  # 不再需要：使用 scrollbar 管理

        # 安全阈值（可按内存大小调整）
        self.MAX_BASE = 3000   # 打开图片时如果边长大于此进行下采样（避免超大内存）
        self.MAX_SIDE = 4000   # transformed pixmap 单边最大像素，超出时做降尺度保护

    def update_zoom_label(self, value):
        self.zoom_label.setText(f'{value}%')

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open image', '',
            'Images (*.png *.jpg *.jpeg *.bmp *.gif)'
        )
        if not path:
            return

        try:
            img = Image.open(path).convert('RGB')

            # 如果图片非常大，先按比例缩小到 MAX_BASE 以防后续变换内存爆炸
            if max(img.width, img.height) > self.MAX_BASE:
                scale = self.MAX_BASE / max(img.width, img.height)
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), Image.BILINEAR)

            self.pil_image = img
            self.angle = 0
            # 生成并保存原始 QPixmap（用于后续 QTransform 变换）
            self.qpix_original = self.pil2pixmap(self.pil_image)
            self.update_display()
            self.result_label.setText('Result: -')
            # 将滚动条归位（图片打开时居中或回到左上角）
            self.scroll.horizontalScrollBar().setValue(0)
            self.scroll.verticalScrollBar().setValue(0)
        except Exception as e:
            self.result_label.setText(f'Error loading image: {str(e)}')

    def pil2pixmap(self, img: Image.Image):
        """
        将 PIL Image 转换为 QPixmap（稳健版）：
        - 优先使用 QImage(data, w, h, bytesPerLine, format) 并调用 copy()，
          避免 Python 字节缓冲被回收导致颜色通道错位（黑白/扭曲）。
        - 回退到 PNG 字节流方法以兼容所有情况。
        """
        try:
            mode = img.mode
            w, h = img.size
            if mode == "RGB":
                data = img.tobytes("raw", "RGB")
                bytes_per_line = 3 * w
                qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
                # copy() 确保像素数据被拷贝到 Qt 管理的内存，避免与 Python 缓冲绑定
                return QPixmap.fromImage(qimg.copy())
            elif mode == "RGBA":
                data = img.tobytes("raw", "RGBA")
                bytes_per_line = 4 * w
                qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGBA8888)
                return QPixmap.fromImage(qimg.copy())
            else:
                # 转为 RGB 再尝试
                img_rgb = img.convert("RGB")
                return self.pil2pixmap(img_rgb)
        except Exception as e:
            # 回退到 PNG-bytes 方法（兼容性最强）
            try:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.getvalue(), "PNG")
                return pixmap
            except Exception as e2:
                print(f"pil2pixmap failed: {e}; fallback failed: {e2}")
                return QPixmap()

    def update_display(self):
        """
        保持你原来的 QTransform 缩放/旋转语义，但在极端大尺寸时做保护，
        避免生成超大 QPixmap 导致 OOM/崩溃。
        同时为支持平移，我们会把 QLabel 的大小设为 pixmap 的大小，
        让 QScrollArea 管理滚动条，用户可通过拖拽改变视图。
        """
        if self.pil_image is None or self.qpix_original is None:
            return

        try:
            scale = self.zoom_slider.value() / 100.0

            # 构造变换（先旋转再缩放，保留你的逻辑）
            transform = QTransform()
            transform.rotate(self.angle)
            transform.scale(scale, scale)

            # 应用变换到原始 pixmap
            transformed_pixmap = self.qpix_original.transformed(
                transform,
                Qt.SmoothTransformation
            )

            # 保护：若变换后过大，按比例降尺度，但不改变用户缩放意图，只有在极端时触发
            tw = transformed_pixmap.width()
            th = transformed_pixmap.height()
            if max(tw, th) > self.MAX_SIDE:
                factor = self.MAX_SIDE / max(tw, th)
                new_w = max(1, int(tw * factor))
                new_h = max(1, int(th * factor))
                transformed_pixmap = transformed_pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 显示到 label，并设置 label 固定大小（这样 scroll area 可以显示滚动条）
            self.img_label.setPixmap(transformed_pixmap)
            self.img_label.setFixedSize(transformed_pixmap.size())

            # 如果图片小于 scroll viewport，居中显示（QLabel 的 alignment 负责）
            # 这里我们把 scroll 的 widgetResizable 设为 False，label 的大小由 pixmap 决定

        except Exception as e:
            print(f"Error in update_display: {e}")
            self.img_label.setText(f"Display Error: {str(e)}")

    def rotate_left(self):
        self.angle = (self.angle - 90) % 360
        # 旋转后把 scroll 滚回到左上角以避免视图混乱（也可以保留当前位置）
        self.update_display()
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)

    def rotate_right(self):
        self.angle = (self.angle + 90) % 360
        self.update_display()
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)

    def run_inference(self):
        if self.pil_image is None:
            self.result_label.setText('Result: Open an image first')
            return

        # 确保 UI 更新
        QApplication.processEvents()

        if self.engine is None:
            try:
                self.result_label.setText('Preparing inference engine...')
                QApplication.processEvents()
                self.engine = InferenceEngine()
                self.result_label.setText('Engine ready, running inference...')
                QApplication.processEvents()
            except Exception as e:
                self.result_label.setText(f'Error preparing engine: {e}')
                return

        try:
            # 对旋转后的 PIL 图像做推理（保持原始 self.pil_image 不变）
            test_img = self.pil_image.rotate(self.angle, expand=True)
            res = self.engine.predict(test_img)

            if isinstance(res, dict) and res.get('error'):
                self.result_label.setText(f"Error: {res.get('message')}")
            else:
                self.result_label.setText(
                    f"Result: {res['label']} ({res['prob']:.2%})"
                )
        except Exception as e:
            self.result_label.setText(f'Inference error: {str(e)}')

    def resizeEvent(self, event):
        """窗口大小改变时更新显示（保留你的行为）"""
        super().resizeEvent(event)
        if hasattr(self, 'pil_image') and self.pil_image is not None:
            self.update_display()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    w = ImageViewer()
    w.resize(900, 600)
    w.show()

    sys.exit(app.exec_())
