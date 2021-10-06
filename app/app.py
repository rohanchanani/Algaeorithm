# A very simple Flask Hello World app for you to get started with...

from io import BytesIO
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from flask import Flask, render_template, request, redirect, send_file
from skimage.filters import threshold_otsu, threshold_local
from PIL import Image
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage, stats
import cv2
import json
import base64
import math

app = Flask(__name__)


def threshold_image(image, clear_background=True, block_size=35):
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
    localMax = peak_local_max(D, indices=False, min_distance=min_distance, labels=thresh)
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
    if not len(all_contours):
        return []
    # Remove outliers in contour area
    contour_areas = list(map(cv2.contourArea, all_contours))
    if len(all_contours) > 1:
        all_contours = [contour for idx, contour in enumerate(all_contours) if abs(stats.zscore(contour_areas)[idx]) < 5]
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
    if not all_contours:
        return [], []
    contour_areas = list(map(cv2.contourArea, all_contours))
    new_min_distance = round(math.sqrt(np.mean(contour_areas)) / 3)
    all_contours = image_to_contours_list(image, clear_background=clear_background, block_size=block_size,
                                          min_ratio=min_ratio, max_ratio=max_ratio, min_distance=new_min_distance)
    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
    all_contours, coords = remove_cells_within_cells(all_contours)
    return all_contours, coords


def get_img_from_fig(fig, tight=True, dpi=180):
    buf = BytesIO()
    if tight:
        fig.savefig(buf, format="jpeg", dpi=dpi, bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig(buf, format="jpeg", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def annotate_image(image, outlines, circles=False):
    if not outlines:
        return image_array_to_base64(image)
    if circles:
        fig = plt.figure()
        ax = plt.axes(frameon=False)
        ax.axis("off")
        ax.get_xaxis().tick_bottom()
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.set_aspect('equal')
        ax.imshow(image)
        for ((x1, y1), r1) in outlines[1]:
            circ = Circle((x1, y1), r1)
            ax.add_patch(circ)
        fig.add_axes(ax)
        return image_array_to_base64(get_img_from_fig(fig))
    else:
        cv2.drawContours(image, outlines[0], -1, (255, 0, 0), 2)
        return image_array_to_base64(image)


def image_array_to_base64(arr):
    img = Image.fromarray(arr.astype('uint8'))
    file_object = BytesIO()
    try:
        img.save(file_object, format="JPEG")
    except:
        img.save(file_object, format="PNG")
    img_str = base64.b64encode(file_object.getvalue())
    file_object.close()
    return img_str.decode("utf-8")

def calculate_concentration(image, contours, cell_volume, mm_depth):
    avg_area = np.mean(list(map(cv2.contourArea, contours)))
    expected_area = (((cell_volume * 3 / 4 / math.pi) ** (1/3)) ** 2) * math.pi
    return len(contours) / ((expected_area / avg_area * image.shape[0] * image.shape[1]) * 10 ** -8 * mm_depth / 10)

def load_response(key, filename, filedata, counts, concentrations):
    final_data[key][filename] = {}
    if filename==filedata:
        try:
            img = np.asarray(io.imread(filename))
        except:
            final_data[key][filename]["count"] = "{} (error)".format(filename)
            return 0
    else:  
        try:
            img = np.asarray(Image.open(BytesIO(filedata.read())))
        except:
            final_data[key][filename]["count"] = "{} (error)".format(filename)
            return 0
    try:
        cell_results = count_cells(img)
        concentration = calculate_concentration(img, cell_results[0], 271.8, 0.1) / 1000000
    except:
        final_data[key][filename]["count"] = "{} (error)".format(filename)
        final_data[key][filename]["image"] = image_array_to_base64(img)
        return 0
    final_data[key][filename] = {}
    final_data[key][filename]["count"] = "{} ({} cells)".format(filename, len(cell_results[0]))
    final_data[key][filename]["image"] = image_array_to_base64(img)
    final_data[key][filename]["outlines"] = annotate_image(img, cell_results)
    final_data[key][filename]["circles"] = annotate_image(img, cell_results, True)
    final_data[key][filename]["concentration"] = concentration
    counts.append(len(cell_results[0]))
    concentrations.append(concentration)
    
def boxplot(data, metric):
    units = {"Count": "# of Cells", "Concentration": "Cells / mL (millions)"}
    fig = plt.figure()
    ax = plt.axes()
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_xlabel(units[metric])
    ax.yaxis.set_ticklabels([])
    ax.set_title("Cell " + metric + " Box Plot")
    fig.add_axes(ax)
    return(image_array_to_base64(get_img_from_fig(fig, False)))

def histogram(data, metric):
    units = {"Count": "# of Cells", "Concentration": "Cells / mL (millions)"}
    fig = plt.figure()
    ax = plt.axes()
    ax.hist(data)
    ax.set_xlabel(units[metric])
    ax.set_ylabel("Frequency")
    ax.set_title("Cell " + metric + " Histogram")
    fig.add_axes(ax)
    return(image_array_to_base64(get_img_from_fig(fig, False)))   

@app.route('/')
def index_get():
    return render_template("index1.html")


@app.route('/', methods=["POST"])
def index_post():        
    global final_data
    final_data = {"file_counts": {}, "url_counts": {}}
    counts = []
    concentrations = []
    for filename, file in request.files.items():
        load_response("file_counts", filename, file, counts, concentrations)
    for image_url in json.loads(request.form.get("url")):
        load_response("url_counts", image_url, image_url, counts, concentrations)
    if len(concentrations) > 1 and len(counts) > 1:
        final_data["stats"] = {"Count": {"Mean": str(round(np.mean(counts), 2)), "Range": str(round(np.ptp(counts), 2)), "Standard Deviation": str(round(np.std(counts), 2)), "iqr": list(map(lambda x: str(round(x, 2)), np.percentile(counts, [25, 50, 75])))}, "Concentration": {"Mean": str(round(np.mean(concentrations), 2)), "Range": str(round(np.ptp(concentrations), 2)), "Standard Deviation": str(round(np.std(concentrations), 2)), "iqr": list(map(lambda x: str(round(x, 2)), np.percentile(concentrations, [25, 50, 75])))}}
        final_data["graphs"] = {"Count": {"Box Plot": boxplot(counts, "Count"), "Histogram": histogram(counts, "Count")}, "Concentration": {"Box Plot": boxplot(concentrations, "Concentration"), "Histogram": histogram(concentrations, "Concentration")}}
    else:
        final_data["stats"] = "No data available"
    return final_data
