#!/usr/bin/env python
"""
ColorChecker Analysis Application

This script provides a Tkinter UI for loading images and performing a DeltaE analysis
on a ColorChecker chart. It allows auto-detection and manual selection of the
ColorChecker pattern and then computes DeltaE differences between the extracted patches
and a reference.
"""

import datetime
import logging
import os
import platform
import re
import threading
import time
from tkinter import filedialog
from tkinter import ttk
import tkinter as tk

from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
from colormath.color_objects import sRGBColor
import cv2
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

# Use TkAgg backend for matplotlib.
matplotlib.use('TkAgg')

# Constants for saving figures
SAVED_FIG_DPI = 300
SAVED_FIG_WIDTH_INCHES = 8
SAVED_FIG_HEIGHT_INCHES = 6


# =============================================================================
# Analysis Functions
# =============================================================================
class DeltaEAnalysis:
    """
    Contains methods to perform DeltaE analysis on a ColorChecker image.
    """

    REF_MACBETH_LAB = np.array(
        [
            [37.99, 13.56, 14.06],  # Dark Skin
            [65.71, 18.13, 17.81],  # Light Skin
            [49.93, -4.88, -21.93],  # Blue Sky
            [43.14, -13.10, 21.91],  # Foliage
            [55.11, 8.84, -25.40],  # Blue Flower
            [70.72, -33.40, -0.20],  # Bluish Green
            [62.66, 36.07, 57.10],  # Orange
            [40.02, 10.41, -45.96],  # Purplish Blue
            [51.12, 48.24, 16.25],  # Moderate Red
            [30.33, 22.98, -21.59],  # Purple
            [72.53, -23.71, 57.26],  # Yellow Green
            [71.94, 19.36, 67.85],  # Orange Yellow
            [28.78, 14.18, -50.30],  # Blue
            [55.26, -38.34, 31.37],  # Green
            [42.10, 53.38, 28.19],  # Red
            [81.73, 4.04, 79.82],  # Yellow
            [51.94, 49.99, -14.57],  # Magenta
            [49.04, -28.63, -28.64],  # Cyan
            [96.54, -0.48, 1.23],  # White (N9.5)
            [81.26, -0.53, 0.03],  # Neutral 8 (N8)
            [66.77, -0.73, -0.52],  # Neutral 6.5 (N6.5)
            [50.87, -0.13, -0.27],  # Neutral 5 (N5)
            [35.66, -0.46, -0.48],  # Neutral 3.5 (N3.5)
            [20.46, -0.08, -0.31],  # Black (N2)
        ]
    )

    @staticmethod
    def convert_to_lab(image, assume_rgb=False):
        """
        Convert an image to Lab color space using colormath.
        If assume_rgb is False (default), assume image is in BGR and convert it to RGB.
        If assume_rgb is True, assume the image is already in RGB.
        The conversion is done pixel-by-pixel (slow).
        """
        if assume_rgb:
            # If image values are > 1, assume 8-bit, so scale to [0,1]
            image_rgb = image if image.max() <= 1 else image / 255.0
        else:
            # If BGR, reverse channels. Also scale if necessary.
            image_rgb = image[..., ::-1] if image.max() <= 1 else image[..., ::-1] / 255.0

        h, w, _ = image.shape
        image_lab = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                rgb = sRGBColor(*image_rgb[i, j])
                lab = convert_color(rgb, LabColor)
                image_lab[i, j] = [lab.lab_l, lab.lab_a, lab.lab_b]
        return image_lab

    @staticmethod
    def lab_to_rgb(lab):
        """
        Convert a Lab color (as an iterable of three values) to an RGB color.

        :param lab: Iterable of three Lab values.
        :return: Corresponding RGB color as an 8-bit integer numpy array.
        """
        lab_color = LabColor(*lab)
        ref_color = convert_color(lab_color, sRGBColor)
        # Clip the RGB values to [0,1] and convert to 8-bit integer
        rgb_normalized = np.clip([ref_color.rgb_r, ref_color.rgb_g, ref_color.rgb_b], 0, 1)
        return (rgb_normalized * 255).astype(np.uint8)

    @staticmethod
    def unskew_image(image, points):
        """
        Apply a perspective transform to extract and unskew the ColorChecker chart.

        :param image: Source image in BGR format.
        :param points: A list or array of 4 points (x, y) defining the chart.
        :return: Unskewed image.
        :raises ValueError: If the shape of points is not (4, 2).
        """
        points = np.array(points, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError(f'Invalid points shape: {points.shape}, expected (4,2)')
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        width = int(max(x_max - x_min, 1))
        height = int(max(y_max - y_min, 1))
        dest_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(points, dest_points)
        return cv2.warpPerspective(image, matrix, (width, height))

    @staticmethod
    def compute_delta_e(patch_labs):
        """
        Compute DeltaE (CIE2000) values for 24 patches and return both
        the list of individual DeltaE values and the average.

        :param patch_labs: Array of LabColor objects (length 24).
        :return: A tuple (list of deltaE values, average deltaE).
        """
        delta_e_values = [
            float(delta_e_cie2000(patch_labs[i], LabColor(*DeltaEAnalysis.REF_MACBETH_LAB[i])))
            for i in range(24)
        ]
        return delta_e_values, np.mean(delta_e_values)

    @staticmethod
    def run(image, points, left_ax=None, result_text_var=None, file_path=None):
        """
        Compute DeltaE analysis by unskewing the image, extracting patch center colors,
        and displaying both the processed image (with patch overlays) and a histogram
        of DeltaE values.

        :param image: Source image in BGR format.
        :param points: Four vertices of the ColorChecker pattern.
        :param left_ax: Matplotlib Axes for displaying the unskewed image.
        :param result_text_var: Tkinter StringVar for displaying the mean DeltaE.
        :param file_path: Path to the image file.
        :return: A tuple containing:
            - list of deltaE values for each patch,
            - mean DeltaE,
            - list of reference colors (normalized to [0,1]) used for histogram bars.
        """
        # Unskew the image using provided vertices
        unskewed_image = DeltaEAnalysis.unskew_image(image, points)
        if left_ax is not None:
            left_ax.clear()
            left_ax.imshow(cv2.cvtColor(unskewed_image, cv2.COLOR_BGR2RGB))
            left_ax.axis('off')
            left_ax.autoscale(False)
            left_ax.set_title(f'{os.path.basename(file_path)}')

        h, w, _ = unskewed_image.shape
        patch_h, patch_w = h // 4, w // 6
        center_lab_values = []
        ref_colors = []  # Colors (normalized to [0,1]) for histogram bars

        # Loop over the 4x6 grid of patches
        for i in range(4):
            for j in range(6):
                idx = i * 6 + j  # 0-based index
                patch_number = idx + 1  # 1-based index
                y, x = i * patch_h, j * patch_w

                # Compute reference color in RGB (normalized)
                ref_color_rgb = (
                    DeltaEAnalysis.lab_to_rgb(DeltaEAnalysis.REF_MACBETH_LAB[idx]) / 255.0
                )
                ref_colors.append(ref_color_rgb)

                # Draw a red rectangle at the center for sampling
                center_rect_w = int(patch_w * 0.33)
                center_rect_h = int(patch_h * 0.33)
                center_x = int(x + (patch_w - center_rect_w) / 2)
                center_y = int(y + patch_h * 0.4)

                if left_ax is not None:
                    # Draw reference patch rectangle (top 30% of patch)
                    ref_rect = plt.Rectangle(
                        (x, y),
                        patch_w,
                        patch_h * 0.3,
                        facecolor=ref_color_rgb,
                        edgecolor='green',
                        linewidth=2,
                        alpha=1,
                        fill=True,
                    )
                    left_ax.add_patch(ref_rect)

                    center_rect = plt.Rectangle(
                        (center_x, center_y),
                        center_rect_w,
                        center_rect_h,
                        edgecolor='red',
                        facecolor='none',
                        linewidth=2,
                    )
                    left_ax.add_patch(center_rect)

                    # Add patch index number centered in the red rectangle
                    left_ax.text(
                        center_x + center_rect_w / 2,
                        center_y + center_rect_h / 2,
                        str(patch_number),
                        color='red',
                        fontsize=10,
                        horizontalalignment='center',
                        verticalalignment='center',
                    )

                # Extract center patch and compute mean Lab values
                center_patch = unskewed_image[
                    center_y : center_y + center_rect_h, center_x : center_x + center_rect_w
                ]
                center_patch_lab = DeltaEAnalysis.convert_to_lab(center_patch)
                mean_lab = np.mean(center_patch_lab.reshape(-1, 3), axis=0)
                center_lab_values.append(LabColor(*mean_lab))

        # Compute DeltaE values and average DeltaE
        delta_e_values, mean_delta_e = DeltaEAnalysis.compute_delta_e(np.array(center_lab_values))
        result_text_var.set(f'{mean_delta_e}')

        if left_ax is not None:
            left_ax.figure.canvas.draw()

        return delta_e_values, mean_delta_e, ref_colors


# =============================================================================
# UI Helper Classes
# =============================================================================
class TextHandler(logging.Handler):
    """
    A logging handler that writes log messages to a Tkinter Text widget.
    """

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)

        self.text_widget.after(0, append)


class ToolBar(tk.Frame):
    """
    Toolbar for the application that provides buttons for zooming,
    auto-detecting the ColorChecker, manual selection, and DeltaE computation.
    """

    def __init__(
        self,
        master_frame,
        ax,
        fig_text,
        app,
        **kwargs,
    ):
        super().__init__(master_frame, **kwargs)
        self.ax = ax
        self.fig_text = fig_text
        self.save_ax_limits()
        self.fig_text_content = ''
        self.operation_ongoing = False
        self.app = app

        self.zoom_selector = None
        self.zoom_active = False
        self.manual_selection_active = False

        # Toggle logs/status button
        self.toggle_logger_btn = tk.Button(self, text='Show logs', command=self.toggle_logger)
        self.toggle_logger_btn.pack(side=tk.LEFT, padx=2)

        # Zoom and reset buttons
        self.zoom_btn = tk.Button(self, text='Zoom', command=self.toggle_zoom_selector)
        self.zoom_btn.pack(side=tk.LEFT, padx=2)
        self.reset_btn = tk.Button(self, text='Reset zoom', command=self.reset_zoom)
        self.reset_btn.pack(side=tk.LEFT, padx=2)

        # Auto–detect, manual selection, and compute DeltaE buttons
        self.auto_detect_btn = tk.Button(self, text='Auto detect', command=self.auto_detect)
        self.auto_detect_btn.pack(side=tk.LEFT, padx=2)
        self.manual_selection_btn = tk.Button(
            self, text='Manual selection', command=self.toggle_manual_selection
        )
        self.manual_selection_btn.pack(side=tk.LEFT, padx=2)
        self.compute_deltae_btn = tk.Button(
            self, text='Compute DeltaE', command=self.compute_deltae
        )
        self.compute_deltae_btn.pack(side=tk.LEFT, padx=2)

        # Connect key press events for zoom/manual cancellation
        self.cid_keypress = ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def save_ax_limits(self):
        """Save the initial axes limits for later resetting."""
        self.init_xlim = self.ax.get_xlim()
        self.init_ylim = self.ax.get_ylim()

    def toggle_logger(self):
        """Toggle the visibility of the log frame using grid."""
        if self.app.logger_frame.winfo_ismapped():
            self.app.logger_frame.grid_remove()
            self.toggle_logger_btn.config(text='Show logs')
        else:
            self.app.logger_frame.grid()
            self.toggle_logger_btn.config(text='Hide logs')

    def reset_zoom(self):
        """Reset the axes to their original limits."""
        self.ax.set_xlim(self.init_xlim)
        self.ax.set_ylim(self.init_ylim)
        self.ax.figure.canvas.draw()
        self.zoom_active = False
        if self.zoom_selector is not None:
            self.zoom_selector.set_active(False)
            self.zoom_selector = None

    def toggle_zoom_selector(self):
        """Activate or deactivate the zoom (RectangleSelector) mode."""
        if self.zoom_selector is None:
            if not self.operation_ongoing:
                self.ax.figure.canvas.get_tk_widget().config(cursor='plus')
                self.zoom_selector = RectangleSelector(
                    self.ax,
                    self.on_select,
                    useblit=True,
                    button=[1],
                    minspanx=5,
                    minspany=5,
                    spancoords='pixels',
                )
                self.zoom_btn.config(relief='sunken')
                self.starting_an_operation('Click and drag to select a region to zoom into.')
                self.ax.figure.canvas.draw()
                self.ax.figure.canvas.get_tk_widget().focus_set()
        else:
            self.disable_zoom_selector()

    def disable_zoom_selector(self):
        """Disable the zoom selector and restore state."""
        self.zoom_selector.set_active(False)
        self.zoom_selector = None
        self.ax.figure.canvas.get_tk_widget().config(cursor='arrow')
        self.zoom_active = True
        self.zoom_btn.config(relief='raised')
        self.end_ongoing_operation()

    def on_select(self, eclick, erelease):
        """
        Callback when a zoom region is selected.
        Sets the new axes limits.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        self.ax.set_xlim(min(x1, x2), max(x1, x2))
        # Adjust y–limits based on original orientation
        if self.init_ylim[0] > self.init_ylim[1]:
            self.ax.set_ylim(max(y1, y2), min(y1, y2))
        else:
            self.ax.set_ylim(min(y1, y2), max(y1, y2))
        self.ax.figure.canvas.draw()
        self.disable_zoom_selector()

    def auto_detect(self):
        """Trigger the auto–detect callback if not busy."""
        if not self.operation_ongoing:
            self.starting_an_operation('Auto–detecting ColorChecker pattern...')
            self.app.auto_detect_callback()

    def toggle_manual_selection(self):
        """Toggle manual selection mode for the ColorChecker."""
        if not self.manual_selection_active:
            if not self.operation_ongoing:
                self.app.manual_selection_callback(True)
                self.manual_selection_active = True
                self.manual_selection_btn.config(relief='sunken')
                self.starting_an_operation(
                    'Draw a polygon with 4 vertices enclosing the ColorChecker pattern using left mouse button or press "esc" to cancel.'
                )
                self.ax.figure.canvas.get_tk_widget().focus_set()
        else:
            self.disable_manual_selection()

    def disable_manual_selection(self):
        """Disable manual selection mode."""
        self.app.manual_selection_callback(False)
        self.manual_selection_active = False
        self.manual_selection_btn.config(relief='raised')
        self.end_ongoing_operation()

    def compute_deltae(self):
        """Trigger the DeltaE computation if not busy."""
        if not self.operation_ongoing:
            self.app.compute_deltae_callback()

    def on_key_press(self, event):
        """
        Listen for key presses (e.g., escape) to cancel zoom or manual selection.
        """
        if event.key == 'escape':
            if self.zoom_selector is not None:
                self.disable_zoom_selector()
            if self.manual_selection_active:
                self.disable_manual_selection()

    def starting_an_operation(self, temp_fig_text):
        """
        Set the operation flag and disable toolbar buttons.

        :param temp_fig_text: Temporary status message to display.
        """
        self.operation_ongoing = True
        self.auto_detect_btn.config(state='disabled')
        self.manual_selection_btn.config(state='disabled')
        self.compute_deltae_btn.config(state='disabled')
        self.toggle_logger_btn.config(state='disabled')
        self.zoom_btn.config(state='disabled')
        self.reset_btn.config(state='disabled')
        self.fig_text_content = self.fig_text.get_text()
        self.fig_text.set_text(temp_fig_text)
        self.ax.figure.canvas.draw()

    def end_ongoing_operation(self, fig_text_override=None):
        """
        Clear the operation flag and re–enable toolbar buttons.

        :param fig_text_override: Optional text to set instead of the previous status message.
        """
        self.app.root.update_idletasks()
        self.operation_ongoing = False
        if fig_text_override is not None:
            self.fig_text.set_text(fig_text_override)
        else:
            self.fig_text.set_text(self.fig_text_content)
        self.ax.figure.canvas.draw()
        self.auto_detect_btn.config(state='normal')
        self.manual_selection_btn.config(state='normal')
        if len(self.app.colorchecker_selector.vertices) == 4:
            self.compute_deltae_btn.config(state='normal')
        self.toggle_logger_btn.config(state='normal')
        self.zoom_btn.config(state='normal')
        self.reset_btn.config(state='normal')


class ColorCheckerSelector:
    """
    Provides interactive selection and adjustment of the ColorChecker pattern's vertices.
    """

    def __init__(self, ax, fig_text, app):
        self.ax = ax
        self.fig_text = fig_text
        self.app = app
        self.vertices = []
        self.markers = []
        self.selected_idx = None
        self.polygon = None
        self.polygon_selector = None
        self.magnifier_rect = None
        self.magnifier_ax = None

        # Connect matplotlib events
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_keypress = ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Parameters for fine–tuning vertex positions and zoom magnifier
        self.latest_key = None
        self.last_executed_key = None
        self.last_update_time = 0
        self.key_cooldown = 0.5  # seconds between key updates
        self.update_step = 2  # pixels to move per key press
        self.zoom_size = 100  # size of magnifier inset in pixels

    def update_image(self):
        """
        Update the image shown in the axes and prepare the magnifier.

        :param image: New image (BGR) to display.
        """
        self.app.toolbar.save_ax_limits()
        self.markers = []
        self.polygon = None
        if self.magnifier_ax is None:
            self.magnifier_ax = inset_axes(
                self.ax,
                width=self.zoom_size / self.ax.figure.dpi,
                height=self.zoom_size / self.ax.figure.dpi,
                loc='upper right',
                bbox_to_anchor=(1.1, 1),
                bbox_transform=self.ax.transAxes,
                borderpad=0,
            )
        self.draw_polygon()

    def draw_polygon(self):
        """
        Draw the polygon defined by the current vertices.
        """
        if len(self.vertices) == 4:
            # Ensure each vertex is drawn/updated.
            for idx, point in enumerate(self.vertices):
                self.update_point(idx, point[0], point[1], draw_polygon=False)
            if self.polygon is None:
                self.polygon = patches.Polygon(
                    np.array(self.vertices), closed=True, edgecolor='cyan', linewidth=2, fill=False
                )
                self.ax.add_patch(self.polygon)
            else:
                self.polygon.set_xy(self.vertices)
            # Adjust axes limits to the polygon with some margin.
            margin = self.app.loaded_image.shape[1] / 15
            x_min, y_min = np.min(self.vertices, axis=0) - margin
            x_max, y_max = np.max(self.vertices, axis=0) + margin
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = (
                min(self.app.loaded_image.shape[1], x_max),
                min(self.app.loaded_image.shape[0], y_max),
            )
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_max, y_min)
            self.fig_text.set_text(
                'ColorChecker pattern vertices are set. Adjust them using left mouse button or WAXD keys.'
            )
            self.app.toolbar.compute_deltae_btn.config(state='normal')
        else:
            self.fig_text.set_text(
                'Click on "Auto detect" or "Manual selection" to set the ColorChecker pattern vertices.'
            )
            self.app.toolbar.compute_deltae_btn.config(state='disabled')
        self.selected_idx = None
        self.clear_magnifier_rect()
        self.ax.figure.canvas.draw()

    def sort_points_clockwise(self, pts):
        """
        Sort a set of 2D points in clockwise order.

        :param pts: Iterable of points.
        :return: Numpy array of sorted points.
        """
        center = np.mean(pts, axis=0)
        sorted_pts = sorted(pts, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        return np.array(sorted_pts)

    def auto_detect(self):
        """
        Auto–detect the ColorChecker pattern using OpenCV's MCC detector.
        """
        detector = cv2.mcc.CCheckerDetector_create()
        fig_text_override = 'Auto–detecting ColorChecker pattern failed. Try manual selection.'
        if not detector.process(self.app.loaded_image, cv2.mcc.MCC24, 1):
            self.app.logger.info('ColorChecker not detected.')
        else:
            checkers = detector.getListColorChecker()
            if not checkers:
                self.app.logger.info('ColorChecker not detected.')
            else:
                checker = checkers[0]
                bounding_box = checker.getBox()
                vertices = self.sort_points_clockwise(
                    np.array(bounding_box, dtype=np.float32)
                ).tolist()
                self.app.logger.info('Auto detection successful.')
                if vertices:
                    self.vertices = np.array(vertices)
                    fig_text_override = 'ColorChecker pattern vertices are set. Adjust them using left mouse button or WAXD keys.'
                    self.draw_polygon()
                    if self.polygon_selector is not None:
                        self.polygon_selector.set_visible(False)
                        self.polygon_selector.disconnect_events()
                        self.polygon_selector = None

        self.app.root.update_idletasks()
        self.app.toolbar.end_ongoing_operation(fig_text_override)
        self.ax.figure.canvas.draw_idle()

    def control_manual_selection(self, enable=True):
        """
        Enable or disable manual polygon selection mode.

        :param enable: True to start manual selection, False to cancel.
        """
        if enable:
            if self.polygon is not None:
                self.polygon.remove()
                self.polygon = None
            self.selected_idx = None
            if self.magnifier_rect:
                self.magnifier_rect.remove()
                self.magnifier_rect = None
            self.polygon_selector = PolygonSelector(self.ax, self.on_select)
        else:
            if self.polygon_selector is not None:
                self.polygon_selector.set_visible(False)
                self.polygon_selector.disconnect_events()
                self.polygon_selector = None
                self.draw_polygon()
        self.ax.figure.canvas.draw()

    def on_select(self, verts):
        """
        Callback when a polygon is drawn in manual selection mode.
        Expects exactly 4 vertices.

        :param verts: List of vertices from the polygon selector.
        """
        points = np.array(verts, dtype=np.float32)
        if len(points) != 4:
            self.app.logger.info(f'Only {len(points)} vertices selected! Draw exactly 4 vertices.')
            # Restart the polygon selector.
            self.polygon_selector.set_visible(False)
            self.polygon_selector.disconnect_events()
            self.polygon_selector = PolygonSelector(self.ax, self.on_select)
            self.ax.figure.canvas.draw()
            return
        self.polygon_selector.set_visible(False)
        self.polygon_selector.disconnect_events()
        self.polygon_selector = None
        self.vertices = points
        self.app.toolbar.disable_manual_selection()
        self.draw_polygon()

    def on_press(self, event):
        """
        Mouse press callback for selecting a vertex to adjust.

        :param event: Matplotlib event.
        """
        if event.inaxes is None or event.inaxes != self.ax:
            return
        # If in manual selection mode, do not change vertices.
        if len(self.vertices) != 4 or self.app.toolbar.manual_selection_active:
            return
        distances = np.linalg.norm(self.vertices - np.array([event.xdata, event.ydata]), axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 100:
            self.selected_idx = closest_idx
            self.update_magnifier(
                self.vertices[self.selected_idx][0], self.vertices[self.selected_idx][1]
            )
            self.ax.figure.canvas.draw()
        else:
            self.selected_idx = None

    def on_motion(self, event):
        """
        Mouse motion callback to update vertex position or magnifier view.

        :param event: Matplotlib event.
        """
        if event.inaxes is None or event.inaxes != self.ax:
            return
        if self.app.toolbar.manual_selection_active:
            self.update_magnifier(event.xdata, event.ydata, add_magnifier_rect=False)
        elif self.selected_idx is not None and event.button == 1:
            self.update_point(self.selected_idx, event.xdata, event.ydata)

    def on_key_press(self, event):
        """
        Keyboard callback for fine–tuning the selected vertex.
        Uses 'a', 'd', 'w', and 'x' for left, right, up, and down.

        :param event: Matplotlib key press event.
        """
        if event.inaxes is None or event.inaxes != self.ax:
            return
        if self.app.toolbar.manual_selection_active:
            return
        if event.key == 'escape':
            self.selected_idx = None
            self.clear_magnifier_rect()
            return
        if self.selected_idx is None:
            return
        if event.key in ['a', 'd', 'w', 'x']:
            self.latest_key = event.key
            current_time = time.time()
            if (
                self.latest_key == self.last_executed_key
                and current_time - self.last_update_time < self.key_cooldown
            ):
                return
            self.last_update_time = current_time
            x, y = self.vertices[self.selected_idx]
            if self.latest_key == 'a':
                x -= self.update_step
            elif self.latest_key == 'd':
                x += self.update_step
            elif self.latest_key == 'w':
                y -= self.update_step
            elif self.latest_key == 'x':
                y += self.update_step
            self.update_point(self.selected_idx, x, y)
            self.last_executed_key = self.latest_key

    def update_point(self, idx, x, y, draw_polygon=True):
        """
        Update the position of a vertex and redraw its marker and polygon.

        :param idx: Index of the vertex.
        :param x: New x coordinate.
        :param y: New y coordinate.
        :param draw_polygon: Whether to update the polygon connecting all vertices.
        """
        if idx < len(self.vertices):
            self.vertices[idx] = [x, y]
        else:
            self.vertices.append([x, y])
        if idx < len(self.markers):
            self.markers[idx].set_xdata([x])
            self.markers[idx].set_ydata([y])
        else:
            marker = self.ax.plot(x, y, marker='+', color='red', markersize=10, picker=10)[0]
            self.markers.append(marker)
        if draw_polygon and len(self.vertices) >= 2:
            if self.polygon is None:
                self.polygon = patches.Polygon(
                    np.array(self.vertices),
                    closed=False,
                    edgecolor='cyan',
                    linewidth=2,
                    fill=False,
                )
                self.ax.add_patch(self.polygon)
            else:
                self.polygon.set_xy(self.vertices)
            if len(self.vertices) == 4:
                self.polygon.set_closed(True)
        self.update_magnifier(x, y)
        self.ax.figure.canvas.draw()
        self.app.root.update_idletasks()

    def update_magnifier(self, x, y, add_magnifier_rect=True):
        """
        Update the magnifier inset around the (x, y) location.

        :param x: x coordinate.
        :param y: y coordinate.
        :param add_magnifier_rect: Whether to draw a rectangle on the main axes.
        """
        x1, x2 = int(x - self.zoom_size / 2), int(x + self.zoom_size / 2)
        y1, y2 = int(y - self.zoom_size / 2), int(y + self.zoom_size / 2)
        x1, x2 = max(0, x1), min(self.app.loaded_image.shape[1] - 1, x2)
        y1, y2 = max(0, y1), min(self.app.loaded_image.shape[0] - 1, y2)
        zoomed_region = self.app.loaded_image[y1:y2, x1:x2]
        self.clear_magnifier_rect()
        self.magnifier_ax.imshow(cv2.cvtColor(zoomed_region, cv2.COLOR_BGR2RGB))
        self.magnifier_ax.invert_yaxis()
        self.magnifier_ax.plot(
            self.zoom_size / 2,
            self.zoom_size / 2,
            marker='+',
            color='red',
            markersize=30,
            linewidth=3,
        )
        if add_magnifier_rect:
            self.magnifier_rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, edgecolor='yellow', linewidth=2, fill=False
            )
            self.ax.add_patch(self.magnifier_rect)
        self.ax.figure.canvas.draw()

    def clear_magnifier_rect(self):
        """Clear the magnifier inset and remove any magnifier rectangle from the main axes."""
        if self.magnifier_ax is not None:
            self.magnifier_ax.clear()
            self.magnifier_ax.set_xlim(0, self.zoom_size)
            self.magnifier_ax.set_ylim(0, self.zoom_size)
            self.magnifier_ax.axis('off')
        if self.magnifier_rect:
            self.magnifier_rect.remove()
            self.magnifier_rect = None
            self.ax.figure.canvas.draw()


class FrameTabs(tk.Frame):
    """
    A simple tab container that lets the user switch between UI frames.
    """

    def __init__(self, master, tabs, *args, **kwargs):
        """
        :param tabs: List of tuples (tab_name, tab_frame)
        """
        super().__init__(master, *args, **kwargs)
        # Frame for tab buttons
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side='top', fill='x')
        # Container for tab contents
        self.content_frame = tk.Frame(self)
        self.content_frame.pack(side='top', fill='both', expand=True)
        self.tabs = {}
        self.buttons = {}
        for name, frame in tabs:
            self.tabs[name] = frame
            frame.place(in_=self.content_frame, x=0, y=0, relwidth=1, relheight=1)
            btn = tk.Button(self.button_frame, text=name, command=lambda n=name: self.show_tab(n))
            btn.pack(side='left', padx=2, pady=2)
            self.buttons[name] = btn
        if tabs:
            self.show_tab(tabs[0][0])

    def show_tab(self, tab_name):
        """
        Bring the selected tab to the front and update button styles.

        :param tab_name: Name of the tab to display.
        """
        for name, frame in self.tabs.items():
            frame.lower()
        self.tabs[tab_name].lift()
        for name, btn in self.buttons.items():
            btn.config(relief='sunken' if name == tab_name else 'raised')


class ScrollableFrame(tk.Frame):
    def __init__(self, container, default_height, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        # Create a canvas and a vertical scrollbar.
        self.canvas = tk.Canvas(self, height=default_height, highlightthickness=0)
        self.vscrollbar = tk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscrollbar.set)
        self.vscrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.default_height = default_height

        # Create an inner frame inside the canvas.
        self.inner_frame = tk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')

        # Update scroll region when the inner frame changes.
        self.inner_frame.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        # Bind global mouse wheel events when pointer enters the inner frame.
        self.inner_frame.bind('<Enter>', self._bind_all_mousewheel)
        self.inner_frame.bind('<Leave>', self._unbind_all_mousewheel)

        # Create the results label (we will preserve this when clearing widgets)
        self.results_label = tk.Label(
            self.inner_frame,
            text='DeltaE analysis results will appear here',
            fg='gray',
            font=('Arial', 12),
        )
        # Initially pack the results label.
        self.results_label.pack(fill='both', expand=False)

    def _on_frame_configure(self, event):
        # Update the scroll region to encompass the inner frame.
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        # Adjust the inner frame's width to match the canvas.
        self.canvas.itemconfig(self.window, width=event.width)

    def _on_mousewheel(self, event):
        system = platform.system()
        if system == 'Windows':
            self.canvas.yview_scroll(-1 * int(event.delta / 120), 'units')
        elif system == 'Darwin':  # macOS
            self.canvas.yview_scroll(-1 * int(event.delta), 'units')
        else:
            # For Linux, wheel events are Button-4 (up) and Button-5 (down)
            if event.num == 4:
                self.canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                self.canvas.yview_scroll(1, 'units')

    def _bind_all_mousewheel(self, event):
        system = platform.system()
        if system in ['Windows', 'Darwin']:
            self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        else:
            self.canvas.bind_all('<Button-4>', self._on_mousewheel)
            self.canvas.bind_all('<Button-5>', self._on_mousewheel)

    def _unbind_all_mousewheel(self, event):
        system = platform.system()
        if system in ['Windows', 'Darwin']:
            self.canvas.unbind_all('<MouseWheel>')
        else:
            self.canvas.unbind_all('<Button-4>')
            self.canvas.unbind_all('<Button-5>')

    def clear(self):
        """
        Clear the inner frame of all widgets except the results_label.
        """
        for widget in self.inner_frame.winfo_children():
            if widget is not self.results_label:
                widget.destroy()
        # Re-pack the results label if necessary.
        self.results_label.pack(fill='both', expand=False)
        self.update_idletasks()

    def adjust_height(self, row_height=30, max_height=500):
        """
        Adjust the canvas height based on the number of rows (widgets)
        in the inner_frame, with a given row height and maximum height.
        """
        # Count the rows (if you want to exclude the results_label, filter it out)
        rows = [w for w in self.inner_frame.winfo_children() if w is not self.results_label]
        new_height = len(rows) * row_height
        # Ensure the height is at least the default and no more than max_height
        new_height = max(self.default_height, min(new_height, max_height))
        self.canvas.config(height=new_height)
        self.update_idletasks()


# =============================================================================
# Main Application Class
# =============================================================================
class ColorCheckerApp:
    """
    Main application class encapsulating the Tkinter UI, callbacks, and overall logic.
    """

    def __init__(self):
        # Determine directories relative to script location.
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.script_dir, '../data/')

        # State variables.
        self.loaded_image = None
        self.srgb_lut = None
        self.colorchecker_selector = None

        # Create root window.
        self.root = tk.Tk()
        self.root.title('ColorChecker Analysis')
        self.root.geometry('1200x800')
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        if platform.system() == 'Windows':
            self.root.state('zoomed')
        else:
            self.root.attributes('-zoomed', True)

        # Create the main layout.
        self._create_layout()
        # Create UI elements (file tree, canvases, toolbar, etc.).
        self._create_widgets()

        # Bind window resize events.
        self.root.bind('<Configure>', self.update_sashes)

    def _create_layout(self):
        """
        Create the main layout using vertical and horizontal PanedWindows.
        """
        self.initial_v_ratio = 0.9
        self.initial_h_ratio = 0.25

        self.main_vpaned = tk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.main_vpaned.pack(fill=tk.BOTH, expand=True)

        self.main_content = tk.Frame(self.main_vpaned)
        self.main_vpaned.add(self.main_content)

        self.main_hpaned = tk.PanedWindow(self.main_content, orient=tk.HORIZONTAL)
        self.main_hpaned.pack(fill=tk.BOTH, expand=True)

    def _create_widgets(self):
        """
        Create all the UI widgets.
        """
        # Left frame: file tree with file filter.
        self.left_frame = tk.Frame(self.main_hpaned, width=350, bg='lightgray')
        self.left_frame.pack_propagate(False)
        self.main_hpaned.add(self.left_frame, minsize=100)
        self._create_file_tree()

        # Right frame: tabbed display for image and analysis.
        self.right_frame = tk.Frame(self.main_hpaned)
        self.main_hpaned.add(self.right_frame)
        self._create_tabs()

    def create_entry_with_scrollbar(self, parent, var):
        """
        Create an Entry widget with a horizontal scrollbar.

        :param parent: Parent widget.
        :param var: Tkinter StringVar to associate with the Entry.
        :return: Tuple (frame, entry widget).
        """
        frame = tk.Frame(parent)
        entry = tk.Entry(frame, textvariable=var)
        entry.grid(row=0, column=0, sticky='ew')
        scrollbar = tk.Scrollbar(frame, orient='horizontal', command=entry.xview)
        scrollbar.grid(row=1, column=0, sticky='ew')
        entry.config(xscrollcommand=scrollbar.set)
        frame.columnconfigure(0, weight=1)
        return frame, entry

    def _create_file_tree(self):
        """
        Create the file tree on the left side with a data directory selector and a filtering Entry.
        The filter expression is applied to the file path relative to self.data_dir.
        """
        # --- Data Directory Selection ---
        dir_frame = tk.Frame(self.left_frame)
        dir_frame.pack(fill=tk.X, padx=5, pady=5)
        dir_frame.columnconfigure(1, weight=1)

        dir_label = tk.Label(dir_frame, text='Data directory:', anchor='w')
        dir_label.grid(row=0, column=0, sticky='w', padx=(0, 5))

        self.data_dir_var = tk.StringVar(value=self.data_dir)
        dir_entry = tk.Entry(dir_frame, textvariable=self.data_dir_var)
        dir_entry.grid(row=0, column=1, sticky='ew')

        browse_btn = tk.Button(dir_frame, text='Browse', command=self.change_data_dir)
        browse_btn.grid(row=0, column=2, padx=(5, 0))

        refresh_btn = tk.Button(dir_frame, text='Refresh', command=self.populate_tree)
        refresh_btn.grid(row=0, column=3, padx=(5, 0))

        # --- Filter Entry (existing code) ---
        filter_frame = tk.Frame(self.left_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        filter_frame.columnconfigure(1, weight=1)
        filter_label = tk.Label(filter_frame, text='Filter files:', anchor='w')
        filter_label.grid(row=0, column=0, sticky='n', padx=(0, 5))
        self.filter_expr_var = tk.StringVar()
        filter_entry_frame, filter_entry = self.create_entry_with_scrollbar(
            filter_frame, self.filter_expr_var
        )
        filter_entry_frame.grid(row=0, column=1, sticky='ew')
        filter_entry.bind('<Return>', lambda event: self.populate_tree())

        # --- Treeview ---
        self.tree = ttk.Treeview(
            self.left_frame, columns=('modified', 'size'), show='tree headings'
        )
        self.tree.heading('#0', text='Name', anchor='w')
        self.tree.heading(
            'modified',
            text='Last Modified',
            anchor='w',
        )
        self.tree.heading(
            'size',
            text='Size',
            anchor='w',
        )
        vscrollbar = ttk.Scrollbar(self.left_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vscrollbar.set)
        self.tree.pack(side='left', fill=tk.BOTH, expand=True)
        vscrollbar.pack(side='right', fill='y')

        self.tree.bind('<Button-1>', self.on_tree_left_click)
        self.tree.bind('<Double-1>', self.on_tree_select)
        self.tree.bind('<Button-3>', self.on_tree_right_click)
        self.left_frame.bind('<Configure>', self.adjust_tree_columns)
        self.current_menu = None

        self.populate_tree()

    def on_tree_left_click(self, event):
        # If there's any open context menu, close it.
        if self.current_menu is not None:
            self.current_menu.unpost()
            self.current_menu.destroy()
            self.current_menu = None

    def change_data_dir(self):
        """Open a directory selection dialog to change self.data_dir and repopulate the tree."""
        new_dir = filedialog.askdirectory(
            initialdir=self.data_dir_var.get(), title='Select Data Directory'
        )
        if new_dir:
            self.data_dir = new_dir  # update the application's data_dir
            self.data_dir_var.set(new_dir)  # update the entry widget
            self.populate_tree()  # repopulate tree with new directory

    def imread_unicode(self, filepath, flags=cv2.IMREAD_COLOR):
        """
        Read an image from a file with a Unicode file path (e.g., with Swedish characters).

        :param filepath: Path to the image file.
        :param flags: Flag for cv2.imdecode (default: cv2.IMREAD_COLOR).
        :return: Image as a numpy array, or None if an error occurs.
        """
        try:
            # Read file into a numpy array of type uint8.
            stream = np.fromfile(filepath, dtype=np.uint8)
            # Decode the image from the byte stream.
            img = cv2.imdecode(stream, flags)
            return img
        except Exception as e:
            print('Error reading file:', e)
            return None

    def _create_tabs(self):
        """
        Create the tabbed display in the right frame.
        """
        self.tab1_frame = tk.Frame(self.right_frame)
        self.tab2_frame = tk.Frame(self.right_frame)
        self.tab2_frame.pack(fill=tk.BOTH, expand=True)
        tabs = [
            ('ColorChecker Pattern Selection', self.tab1_frame),
            ('DeltaE Analysis', self.tab2_frame),
        ]
        self.tabs = FrameTabs(self.right_frame, tabs)
        self.tabs.pack(fill='both', expand=True)

        # ----- Tab 1: Image display and pattern selection -----
        self.tab1_fig, self.tab1_fig_ax = plt.subplots(figsize=(8, 6))
        self.tab1_fig_text = self.tab1_fig.text(0.02, 0.02, '', fontsize=9, color='blue')
        self.tab1_fig_canvas = FigureCanvasTkAgg(self.tab1_fig, master=self.tab1_frame)
        self.tab1_fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tab1_fig_canvas.draw()
        self.tab1_fig_canvas.get_tk_widget().bind(
            '<Button-3>', lambda event: self.show_context_menu(event, self.tab1_fig)
        )

        # Toolbar for Tab 1.
        toolbar_container = tk.Frame(self.tab1_frame)
        toolbar_container.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar_container.grid_columnconfigure(0, weight=1)

        toolbar_frame = tk.Frame(toolbar_container)
        toolbar_frame.grid(row=0, column=0, sticky='ew')

        self.logger_frame = tk.Frame(toolbar_container, height=100, bg='lightgray')
        self.logger_frame.grid(row=1, column=0, sticky='ew')
        self.logger_frame.pack_propagate(False)
        self.logger_frame.grid_remove()  # Hide by default

        # Set up logging output to the status text widget.
        self._setup_logging()

        # Set up toolbar
        self.toolbar = ToolBar(
            master_frame=toolbar_frame,
            ax=self.tab1_fig_ax,
            fig_text=self.tab1_fig_text,
            app=self,
        )
        self.toolbar.pack(side=tk.RIGHT, fill=tk.X)

        # ----- Tab 2: DeltaE Analysis Display -----
        tab2_canvas_container = tk.Frame(self.tab2_frame)
        tab2_canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        left_frame = tk.Frame(tab2_canvas_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        right_frame = tk.Frame(tab2_canvas_container)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))

        self.tab2_fig_left, self.tab2_fig_left_ax = plt.subplots(figsize=(4, 6))
        self.tab2_canvas_left = FigureCanvasTkAgg(self.tab2_fig_left, master=left_frame)
        self.tab2_canvas_left.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tab2_canvas_left.draw()
        self.tab2_canvas_left.get_tk_widget().bind(
            '<Button-3>', lambda event: self.show_context_menu(event, self.tab2_fig_left)
        )

        self.tab2_fig_right, self.tab2_fig_right_ax = plt.subplots(figsize=(4, 6))
        self.tab2_canvas_right = FigureCanvasTkAgg(self.tab2_fig_right, master=right_frame)
        self.tab2_canvas_right.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tab2_canvas_right.draw()
        self.tab2_canvas_right.get_tk_widget().bind(
            '<Button-3>', lambda event: self.show_context_menu(event, self.tab2_fig_right)
        )

        # Create a canvas and a vertical scrollbar for the results.
        self.tab2_scrollable_results_container = ScrollableFrame(
            self.tab2_frame, default_height=30
        )
        self.tab2_scrollable_results_container.pack(
            side=tk.BOTTOM, fill='x', expand=False, padx=5, pady=5
        )

        self._clear_tabs()

    def _clear_tabs(self, tabs_to_clear=None):
        """
        Clears both Tab 1 and Tab 2 canvases and resets text messages.

        :param tabs_to_clear: Optional iterable of tab names to clear.
        """
        if tabs_to_clear is None:
            tabs_to_clear = self.tabs.tabs.keys()

        if 'ColorChecker Pattern Selection' in tabs_to_clear:
            # Clear Tab 1.
            self.tab1_fig_ax.clear()
            self.tab1_fig_ax.text(
                0.5,
                0.5,
                'No image loaded',
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.tab1_fig_ax.transAxes,
                fontsize=16,
            )
            self.tab1_fig_ax.axis('off')
            self.tab1_fig_ax.autoscale(False)
            self.tab1_fig_text.set_text('')
            self.tab1_fig_canvas.draw()
            self.toolbar.update()
            self.toolbar.compute_deltae_btn.config(state='disabled')

        if 'DeltaE Analysis' in tabs_to_clear:
            # Clear Tab 2 Left.
            self.tab2_fig_left_ax.clear()
            self.tab2_fig_left_ax.text(
                0.5,
                0.5,
                'No analysis performed',
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.tab2_fig_left_ax.transAxes,
                fontsize=16,
            )
            self.tab2_fig_left_ax.axis('off')
            self.tab2_fig_left_ax.autoscale(False)
            self.tab2_canvas_left.draw()

            # Clear Tab 2 Right.
            self.tab2_fig_right_ax.clear()
            self.tab2_fig_right_ax.text(
                0.5,
                0.5,
                'No analysis performed',
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.tab2_fig_right_ax.transAxes,
                fontsize=16,
            )
            self.tab2_fig_right_ax.axis('off')
            self.tab2_fig_right_ax.autoscale(False)
            self.tab2_canvas_right.draw()

            self.tabs.buttons['DeltaE Analysis'].config(text='DeltaE Analysis')

            # # Reset analysis result text.
            self.tab2_scrollable_results_container.clear()

    def _setup_logging(self):
        """
        Set up logging output to the status (log) text widget.
        """
        self.status_text = tk.Text(self.logger_frame, state='disabled', bg='black', fg='white')
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar = tk.Scrollbar(self.logger_frame, command=self.status_text.yview)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        text_handler = TextHandler(self.status_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        text_handler.setFormatter(formatter)
        self.logger.addHandler(text_handler)
        self.logger.info('Application started.')

    def custom_save_dialog(self):
        """
        Create a custom dialog to ask for filename, DPI, and figure size.
        Returns a tuple: (filename, dpi, width, height) or (None, None, None, None) if cancelled.

        :return: Tuple of save options.
        """
        dialog = tk.Toplevel()
        dialog.title('Save Plot Options')
        dialog.grab_set()  # make modal

        tk.Label(dialog, text='Filename:').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        filename_var = tk.StringVar()
        filename_entry = tk.Entry(dialog, textvariable=filename_var, width=40)
        filename_entry.grid(row=0, column=1, padx=5, pady=5)

        def browse():
            fname = filedialog.asksaveasfilename(
                defaultextension='.png', filetypes=[('PNG Files', '*.png'), ('All Files', '*.*')]
            )
            if fname:
                filename_var.set(fname)

        ttk.Button(dialog, text='Browse...', command=browse).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(dialog, text='DPI:').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        dpi_var = tk.StringVar(value=f'{SAVED_FIG_DPI}')
        tk.Entry(dialog, textvariable=dpi_var, width=10).grid(
            row=1, column=1, sticky='w', padx=5, pady=5
        )

        tk.Label(dialog, text='Figure Width (inches):').grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        width_var = tk.StringVar(value=f'{SAVED_FIG_WIDTH_INCHES}')
        tk.Entry(dialog, textvariable=width_var, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=5
        )

        tk.Label(dialog, text='Figure Height (inches):').grid(
            row=3, column=0, sticky='w', padx=5, pady=5
        )
        height_var = tk.StringVar(value=f'{SAVED_FIG_HEIGHT_INCHES}')
        tk.Entry(dialog, textvariable=height_var, width=10).grid(
            row=3, column=1, sticky='w', padx=5, pady=5
        )

        result = {}

        def on_ok():
            result['filename'] = filename_var.get()
            result['dpi'] = float(dpi_var.get())
            result['width'] = float(width_var.get())
            result['height'] = float(height_var.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(dialog, text='OK', command=on_ok).grid(row=4, column=0, padx=5, pady=5)
        ttk.Button(dialog, text='Cancel', command=on_cancel).grid(row=4, column=1, padx=5, pady=5)
        dialog.wait_window()

        if 'filename' in result and result['filename']:
            return result['filename'], result['dpi'], result['width'], result['height']
        else:
            return None, None, None, None

    def save_plot(self, fig):
        """
        Prompt the user for save options and save the given matplotlib figure.

        :param fig: Matplotlib figure to save.
        """
        filename, dpi, new_width, new_height = self.custom_save_dialog()
        if filename:
            original_size = fig.get_size_inches()
            fig.set_size_inches(new_width, new_height)
            original_visibility = [txt.get_visible() for txt in fig.texts]
            for txt in fig.texts:
                txt.set_visible(False)
            fig.savefig(filename, bbox_inches='tight', dpi=dpi)
            for txt, visible in zip(fig.texts, original_visibility):
                txt.set_visible(visible)
            fig.set_size_inches(original_size)

    def show_context_menu(self, event, fig):
        """
        Display a right–click context menu on the given figure with a Save Plot option.

        :param event: Tkinter event.
        :param fig: Matplotlib figure associated with the event.
        """
        menu = tk.Menu(event.widget, tearoff=0)
        menu.add_command(label='Save Plot', command=lambda: self.save_plot(fig))
        menu.post(event.x_root, event.y_root)

    # ------------------------------
    # Callback Methods
    # ------------------------------
    def load_image_from_file(self, file_path):
        """
        Load an image file.

        :param file_path: Path to the image file.
        """
        image = self.imread_unicode(file_path)
        # Schedule the UI update on the main thread.
        self.root.after(0, lambda: self.process_loaded_image(image, file_path))

    def process_loaded_image(self, image, file_path):
        """
        Process the loaded image by updating the display, and initializing the ColorChecker selection.

        :param image: Loaded image (BGR).
        :param file_path: Path to the loaded image file.
        """
        self._clear_tabs()
        self.loaded_image = image
        self.logger.info(f'Loaded image from file: {file_path}')
        self.tab1_fig_ax.clear()
        self.tab1_fig_ax.imshow(cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB))
        self.tab1_fig_ax.axis('off')
        self.tab1_fig_ax.autoscale(False)
        self.tab1_fig_ax.set_title(f'{os.path.basename(file_path)}')
        if self.colorchecker_selector is None:
            self.colorchecker_selector = ColorCheckerSelector(
                self.tab1_fig_ax, self.tab1_fig_text, self
            )
            self.toolbar.colorchecker_selector = self.colorchecker_selector
        if len(self.colorchecker_selector.vertices) == 4:
            self.tab1_fig_text.set_text(
                'Applying the previous selection of ColorChecker pattern...'
            )
        self.colorchecker_selector.clear_magnifier_rect()
        self.tab1_fig_canvas.draw()
        self.tabs.show_tab('ColorChecker Pattern Selection')
        self.root.update_idletasks()
        self.colorchecker_selector.update_image()
        self.current_file = file_path

    def auto_detect_callback(self):
        """
        Callback to trigger auto-detection of the ColorChecker pattern.
        """
        if self.colorchecker_selector is not None:
            threading.Thread(target=self.colorchecker_selector.auto_detect, daemon=True).start()

    def manual_selection_callback(self, enable=True):
        """
        Callback to enable or disable manual selection mode.

        :param enable: True to enable manual selection, False to disable.
        """
        if self.colorchecker_selector is not None:
            self.colorchecker_selector.control_manual_selection(enable)

    def copy_value(self, var):
        """Helper to copy the given StringVar’s value to the clipboard."""
        self.tab2_frame.clipboard_clear()
        self.tab2_frame.clipboard_append(var.get())

    def find_tree_item_by_path(self, target_path, parent=''):
        """
        Recursively search for a tree item whose full path matches the target path.

        :param target_path: The full path of the file to find.
        :param parent: The parent item ID to start searching from.
        :return: The tree item ID if found, otherwise None.
        """
        target_path_norm = os.path.normpath(target_path)
        for item in self.tree.get_children(parent):
            file_path = self.get_full_path(item, self.data_dir)
            if os.path.normpath(file_path) == target_path_norm:
                return item
            # Recursively search in children.
            found = self.find_tree_item_by_path(target_path, item)
            if found:
                return found
        return None

    def compute_deltae_callback(self):
        """
        Updated DeltaE analysis callback.
        If only the current image is selected, perform a single DeltaE analysis as before.
        Otherwise (if additional files are selected) change Tab2’s title and create a row for each file.
        """
        if self.colorchecker_selector is None or len(self.colorchecker_selector.vertices) != 4:
            self.logger.error('Please select the ColorChecker pattern first.')
            return
        self.toolbar.starting_an_operation('Computing DeltaE...')
        self.logger.info('Starting DeltaE analysis...')
        self._clear_tabs(['DeltaE Analysis'])
        self.colorchecker_selector.clear_magnifier_rect()
        self.tabs.show_tab('DeltaE Analysis')

        # Get selected items from the file tree (tree allows multiple selection)
        selected_items = self.tree.selection()
        selected_files = []
        for item in selected_items:
            file_path = self.get_full_path(item, self.data_dir)
            if os.path.isfile(file_path) and (
                file_path.lower().endswith('.png')
                or file_path.lower().endswith('.jpg')
                or file_path.lower().endswith('.jpeg')
            ):
                selected_files.append(file_path)

        # Always include the currently loaded file
        if self.current_file not in selected_files:
            selected_files.insert(0, self.current_file)

        item_id = self.find_tree_item_by_path(self.current_file)
        if item_id:
            current_sel = set(selected_items)
            current_sel.add(item_id)
            self.tree.selection_set(list(current_sel))
            self.tree.focus(item_id)

        # If only one file (the original) is to be processed, do the single analysis.
        if len(selected_files) == 1:
            # (Reset Tab2 title if it was renamed previously.)
            self.logger.info('Performing single-file DeltaE analysis...')
        else:
            # Multi–file analysis:
            self.logger.info('Performing multi-file DeltaE analyses...')
            # Rename Tab2 button
            self.tabs.buttons['DeltaE Analysis'].config(text='Multi-file DeltaE analyses')
        self.tab2_scrollable_results_container.results_label.pack_forget()

        # Function to run analysis on a given file and update its row
        def run_analysis_for_file(file_path, result_var, mean_delta_e_values_list):
            image = self.imread_unicode(file_path)
            if image is None:
                result = 'Error loading image'
            else:
                left_ax = None
                if file_path == self.current_file:
                    left_ax = self.tab2_fig_left_ax
                # Use the same ColorChecker vertices as set in the current session.
                # (Note: In a real multi–file scenario, you might need to let the user adjust vertices per file.)
                delta_e_values, mean_delta_e, ref_colors = DeltaEAnalysis.run(
                    image.copy(),
                    np.array(self.colorchecker_selector.vertices, dtype=np.float32),
                    left_ax,
                    tk.StringVar(),
                    self.current_file,
                )
                result = f'{mean_delta_e}'
                mean_delta_e_values_list.append(mean_delta_e)

                if file_path == self.current_file:
                    # --- Plot histogram in right_ax ---
                    self.tab2_fig_right_ax.clear()
                    patch_indices = list(range(1, len(delta_e_values) + 1))
                    self.tab2_fig_right_ax.bar(patch_indices, delta_e_values, color=ref_colors)
                    self.tab2_fig_right_ax.set_xlabel('Patch Index')
                    self.tab2_fig_right_ax.set_ylabel('DeltaE')
                    self.tab2_fig_right_ax.set_xticks(patch_indices)
                    self.tab2_fig_right_ax.set_title(f'{os.path.basename(file_path)}')
                    # Draw a horizontal line for mean DeltaE
                    self.tab2_fig_right_ax.axhline(
                        mean_delta_e,
                        color='black',
                        linestyle='--',
                        linewidth=2,
                        label=f'Mean DeltaE: {mean_delta_e:.2f}',
                    )
                    self.tab2_fig_right_ax.legend()
                    self.tab2_fig_right_ax.figure.canvas.draw()

            result_var.set(result)

        # Update height of the inner frame to fit the new number of rows.
        self.tab2_scrollable_results_container.config(height=min(30 * len(selected_files), 500))
        self.tab2_scrollable_results_container.update_idletasks()

        if len(selected_files) > 1:
            # Create a row frame that fills horizontally.
            row = tk.Frame(self.tab2_scrollable_results_container.inner_frame)
            row.pack(fill='x', pady=2)

            # # Left frame for the file label.
            left_frame = tk.Frame(row)
            left_frame.pack(side='left', fill='x', expand=True)

            file_label_text = f'Average DeltaE of {len(selected_files)} images'

            file_label = tk.Label(left_frame, text=file_label_text, anchor='w')
            file_label.pack(fill='x')

            # Right frame for the result entry and copy button.
            right_frame = tk.Frame(row)
            right_frame.pack(side='right')

            avg_var = tk.StringVar(value='Computing...')
            result_entry = ttk.Entry(right_frame, textvariable=avg_var, state='readonly', width=50)
            result_entry.pack(side='left', padx=(0, 5))

            copy_btn = tk.Button(
                right_frame, text='Copy', command=lambda v=avg_var: self.copy_value(v)
            )
            copy_btn.pack(side='left')
            self.tab2_scrollable_results_container.adjust_height()

        mean_delta_e_values_list = []

        threads = []
        # For each file, create a UI row and launch its analysis in a separate thread.
        for idx, file_path in enumerate(selected_files):
            # Create a row frame that fills horizontally.
            row = tk.Frame(self.tab2_scrollable_results_container.inner_frame)
            row.pack(fill='x', pady=2)

            # # Left frame for the file label.
            left_frame = tk.Frame(row)
            left_frame.pack(side='left', fill='x', expand=True)

            file_label_text = f'{idx + 1}. {os.path.basename(file_path)}'
            if file_path == self.current_file:
                file_label_text += ' (Current)'

            file_label = tk.Label(left_frame, text=file_label_text, anchor='w')
            file_label.pack(fill='x')

            # Right frame for the result entry and copy button.
            right_frame = tk.Frame(row)
            right_frame.pack(side='right')

            result_var = tk.StringVar(value='Computing...')
            result_entry = ttk.Entry(
                right_frame, textvariable=result_var, state='readonly', width=50
            )
            result_entry.pack(side='left', padx=(0, 5))

            copy_btn = tk.Button(
                right_frame, text='Copy', command=lambda v=result_var: self.copy_value(v)
            )
            copy_btn.pack(side='left')
            self.tab2_scrollable_results_container.adjust_height()

            # Launch the analysis for this file in its own thread.
            t = threading.Thread(
                target=lambda fp=file_path, var=result_var: run_analysis_for_file(
                    fp, var, mean_delta_e_values_list
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        def check_threads():
            if any(t.is_alive() for t in threads):
                self.tab2_scrollable_results_container.update_idletasks()
                self.root.after(100, check_threads)
            else:
                if len(selected_files) > 1 and mean_delta_e_values_list:
                    overall_average = sum(mean_delta_e_values_list) / len(mean_delta_e_values_list)
                    avg_var.set(overall_average)

                self.toolbar.end_ongoing_operation()

        self.root.after(100, check_threads)

    def populate_tree(self):
        """
        Populate the file tree (left frame) based on the given filter expression.
        """
        self.tree.delete(*self.tree.get_children())
        self._insert_items('', self.data_dir, self.filter_expr_var.get())

    def adjust_tree_columns(self, event):
        # Get the current width of the frame containing the treeview.
        total_width = event.width
        # Optionally subtract a few pixels for padding or scrollbar width.
        total_width = max(total_width - 20, 100)  # ensure a minimum width

        # Define the desired width proportions.
        name_pct = 0.5
        modified_pct = 0.35
        size_pct = 0.15

        self.tree.column('#0', width=int(total_width * name_pct))
        self.tree.column('modified', width=int(total_width * modified_pct))
        self.tree.column('size', width=int(total_width * size_pct))

    def evaluate_filter_expr(self, relative_path, filter_expr):
        """
        Evaluate the filter expression on the relative file path.
        Supports boolean operators (and, or, not) and parentheses.

        :param relative_path: File path relative to the data directory.
        :param filter_expr: Filter expression.
        :return: True if the file matches the filter, False otherwise.
        """
        if not filter_expr.strip():
            return True
        try:
            tokens = re.findall(r'\w+|\S', filter_expr)
            new_tokens = []
            for token in tokens:
                if token.lower() in ('and', 'or', 'not') or token in ('(', ')'):
                    new_tokens.append(token)
                else:
                    new_tokens.append(f'("{token}" in path)')
            expr = ' '.join(new_tokens)
            return eval(expr, {'__builtins__': None}, {'path': relative_path.lower()})
        except Exception as e:
            print('Error evaluating filter expression:', e)
            return False

    def on_tree_right_click(self, event):
        """
        Handler for right-click on the tree. Displays a context menu for expanding/collapsing folders or opening PNG files.
        """
        # If there's an open context menu, close it first.
        if self.current_menu is not None:
            self.current_menu.unpost()
            self.current_menu.destroy()
            self.current_menu = None

        item_id = self.tree.identify_row(event.y)
        menu = tk.Menu(self.tree, tearoff=0)
        if not item_id:
            menu.add_command(label='Expand All', command=self.expand_all)
            menu.add_command(label='Collapse All', command=self.collapse_all)
        else:
            file_path = self.get_full_path(item_id, self.data_dir)
            if os.path.isdir(file_path):
                menu.add_command(
                    label='Expand Folder', command=lambda: self.expand_folder(item_id)
                )
                menu.add_command(
                    label='Collapse Folder', command=lambda: self.collapse_folder(item_id)
                )
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                menu.add_command(
                    label='Open',
                    command=lambda: threading.Thread(
                        target=lambda: self.load_image_from_file(file_path), daemon=True
                    ).start(),
                )
        # Post the menu and save the reference.
        menu.post(event.x_root, event.y_root)
        self.current_menu = menu

    def expand_folder(self, item_id):
        """Recursively expand the folder in the tree."""
        self.tree.item(item_id, open=True)
        for child in self.tree.get_children(item_id):
            self.expand_folder(child)

    def collapse_folder(self, item_id):
        """Recursively collapse the folder in the tree."""
        self.tree.item(item_id, open=False)
        for child in self.tree.get_children(item_id):
            self.collapse_folder(child)

    def expand_all(self):
        """Expand all items in the tree."""

        def _expand_recursive(item_id):
            self.tree.item(item_id, open=True)
            for child in self.tree.get_children(item_id):
                _expand_recursive(child)

        for child in self.tree.get_children():
            _expand_recursive(child)

    def collapse_all(self):
        """Collapse all items in the tree."""

        def _collapse_recursive(item_id):
            self.tree.item(item_id, open=False)
            for child in self.tree.get_children(item_id):
                _collapse_recursive(child)

        for child in self.tree.get_children():
            _collapse_recursive(child)

    def _insert_items(self, parent_item, path, filter_expr=''):
        """
        Recursively insert items from the file system into the tree view.

        :param parent_item: Parent item ID in the tree.
        :param path: File system path.
        :param filter_expr: Filter expression to apply.
        :return: True if any items were inserted, False otherwise.
        """
        inserted_any = False
        try:
            for item in sorted(os.listdir(path), reverse=True):
                item_path = os.path.join(path, item)
                relative_path = os.path.relpath(item_path, self.data_dir)
                if os.path.isdir(item_path):
                    modified = datetime.datetime.fromtimestamp(
                        os.path.getmtime(item_path)
                    ).strftime('%Y-%m-%d %H:%M:%S')

                    node = self.tree.insert(
                        parent_item, 'end', text=item, open=True, values=(modified, '')
                    )
                    children_inserted = self._insert_items(node, item_path, filter_expr)
                    if children_inserted or self.evaluate_filter_expr(relative_path, filter_expr):
                        inserted_any = True
                    else:
                        self.tree.delete(node)
                else:
                    if item.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if self.evaluate_filter_expr(relative_path, filter_expr):
                            modified = datetime.datetime.fromtimestamp(
                                os.path.getmtime(item_path)
                            ).strftime('%Y-%m-%d %H:%M:%S')
                            size_bytes = os.path.getsize(item_path)
                            size_text = self.human_readable_size(size_bytes)
                            self.tree.insert(
                                parent_item,
                                'end',
                                text=item,
                                open=True,
                                values=(modified, size_text),
                            )
                            inserted_any = True
            return inserted_any
        except Exception as e:
            print(e)
            return inserted_any

    def human_readable_size(self, size, decimal_places=1):
        """
        Convert a file size (in bytes) to a human-readable string with appropriate unit.
        """
        if size < 1024:
            return f'{size} B'
        elif size < 1024**2:
            return f'{size / 1024:.{decimal_places}f} KB'
        elif size < 1024**3:
            return f'{size / 1024**2:.{decimal_places}f} MB'
        else:
            return f'{size / 1024**3:.{decimal_places}f} GB'

    def parse_size(self, size_str):
        """
        Parse a human-readable size string back to a numeric value in bytes for sorting.
        Expects format like "1.2 KB", "512 B", "3.4 MB", or "1.1 GB".
        """
        try:
            value, unit = size_str.split()
            value = float(value)
            if unit == 'B':
                return value
            elif unit == 'KB':
                return value * 1024
            elif unit == 'MB':
                return value * 1024**2
            elif unit == 'GB':
                return value * 1024**3
        except Exception:
            return 0

    def on_tree_select(self, event):
        """
        Callback when a file is selected from the tree.

        :param event: Tkinter event.
        """
        current_item = self.tree.focus()
        if hasattr(self, 'last_selected_item') and self.last_selected_item == current_item:
            return
        self.last_selected_item = current_item
        file_path = self.get_full_path(current_item, self.data_dir)
        if os.path.isfile(file_path) and (
            file_path.lower().endswith('.png')
            or file_path.lower().endswith('.jpg')
            or file_path.lower().endswith('.jpeg')
        ):
            # Start a thread to load the image from disk.
            threading.Thread(
                target=self.load_image_from_file, args=(file_path,), daemon=True
            ).start()

    def get_full_path(self, item_id, root_dir):
        """
        Reconstruct the full file path from the tree item.

        :param item_id: Tree item ID.
        :param root_dir: Root directory.
        :return: Full file system path.
        """
        path_parts = []
        while item_id:
            path_parts.insert(0, self.tree.item(item_id, 'text'))
            item_id = self.tree.parent(item_id)
        return os.path.join(root_dir, *path_parts)

    def update_sashes(self, event):
        """
        Update paned window sashes upon window resize.

        :param event: Tkinter event.
        """
        panes = self.main_vpaned.panes()
        if len(panes) > 1:
            total_height = self.main_vpaned.winfo_height()
            self.main_vpaned.sash_place(0, 0, int(total_height * self.initial_v_ratio))

    def copy_to_clipboard(self):
        """
        Copy the DeltaE result text to the clipboard.
        """
        self.tab2_frame.clipboard_clear()
        self.tab2_frame.clipboard_append(self.tab2_result_text_var.get())
        self.result_entry.focus_set()
        self.result_entry.selection_range(0, tk.END)

    def on_close(self):
        """
        Clean up resources and close the application.
        """
        plt.close('all')
        self.root.destroy()

    def run(self):
        """Run the main application loop."""
        self.root.mainloop()


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    app = ColorCheckerApp()
    app.run()
