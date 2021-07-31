# A very simple Flask Hello World app for you to get started with...

from io import BytesIO
from skimage import io
import numpy as np
from flask import Flask, render_template, request, redirect
from skimage.filters import threshold_otsu, threshold_local
from PIL import Image
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage, stats
import math
#import cv2

app = Flask(__name__)
@app.route("/")
def hello_world():
    return "Hello from flask"

"""def threshold_image(image, clear_background=True, block_size=35):
    if not image.any():
        return
    if len(image.shape) > 2:
        image = image[:, :, 2]
    if not clear_background:
        thresh = threshold_local(image, block_size)
    else:
        thresh = threshold_otsu(image)
    if np.mean(image) > threshold_otsu(image):
        binary = image < thresh
    else:
        binary = image < thresh
    return binary.astype(np.uint8) * 255


def image_to_contours_list(image, clear_background=True, block_size=35, min_ratio=0.1, max_ratio=3, min_distance=10):
    gray_shape = rgb2gray(image).shape
    # Threshold the image, and fill holes in contours
    thresh = ndimage.morphology.binary_fill_holes(
        threshold_image(image, clear_background, block_size))
    # Find peaks and convert to labels
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(
        D, indices=False, min_distance=min_distance, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    all_contours = []
    # Find contours in labels, and add largest contour to list of contours
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray_shape, dtype="uint8")
        mask[labels == label] = 255
        cnts, hierarchy = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        all_contours.append(c)
    # Remove outliers in contour area
    contour_areas = list(map(cv2.contourArea, all_contours))
    all_contours = [contour for idx, contour in enumerate(
        all_contours) if abs(stats.zscore(contour_areas)[idx]) < 5]
    contour_areas = list(map(cv2.contourArea, all_contours))
    return [contour for contour in all_contours if min_ratio * np.mean(contour_areas) < cv2.contourArea(contour)]


def remove_cells_within_cells(contours):
    i1 = 0
    coords = []
    # Remove outlines that are swallowed up by other outlines
    while i1 < len(contours):
        c1 = contours[i1]
        ((x1, y1), r1) = cv2.minEnclosingCircle(c1)
        coords.append(cv2.minEnclosingCircle(c1))
        i2 = 0
        while i2 < len(contours):
            c2 = contours[i2]
            if not np.array_equal(c1, c2):
                ((x2, y2), r2) = cv2.minEnclosingCircle(c2)
                if ((((x2 - x1)**2) + ((y2-y1)**2))**0.5) < r1:
                    contours.pop(i2)
                else:
                    i2 += 1
            else:
                i2 += 1
        i1 += 1
    return contours, coords


def count_cells(image, clear_background=True, block_size=35, min_ratio=0.1, max_ratio=3, min_distance=10, return_outlines=False):
    if not image.any():
        return -1
    # Calculate area once to get an idea of the area of each cell in pixels, then calculate again with new minimum distance which takes into account the new areas
    all_contours = image_to_contours_list(image, clear_background=clear_background, block_size=block_size,
                                          min_ratio=min_ratio, max_ratio=max_ratio, min_distance=min_distance)
    contour_areas = list(map(cv2.contourArea, all_contours))
    new_min_distance = round(math.sqrt(np.mean(contour_areas)) / 3)
    all_contours = image_to_contours_list(image, clear_background=clear_background, block_size=block_size,
                                          min_ratio=min_ratio, max_ratio=max_ratio, min_distance=new_min_distance)
    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
    all_contours, coords = remove_cells_within_cells(all_contours)
    if return_outlines:
        return all_contours, coords
    return len(all_contours)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    else:
        if request.form.get("url"):
            try:
                img = io.imread(request.form.get("url"))
            except:
                return "Please enter a valid url"
        elif 'file' not in request.files:
            return "No file"
        else:
            file = request.files["file"]
            try:
                img = np.asarray(Image.open(BytesIO(file.read())))
            except:
                return "Invalid file"
        try:
            num_cells = count_cells(img)
        except:
            return "Invalid file"
        return render_template("results.html", num_cells=num_cells)"""
