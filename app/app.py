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
from scipy import ndimage, stats, optimize
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

def load_response(key, filename, filedata, counts, concentrations, csv_rows):
    final_data[key][filename] = {}
    row_to_append = [filename, "N/A", "N/A"]
    depth = request.form.get("depth") if request.form.get("depth") else request.form.get("depth-"+filename)
    row_to_append.insert(1, depth)
    if request.form.get("time-unit"):
        if request.form.get("time-"+filename):
            row_to_append.insert(1, request.form.get("time-"+filename))
        else:
            row_to_append.insert(1, "N/A")
    if filename==filedata:
        try:
            img = np.asarray(io.imread(filename))
        except:
            final_data[key][filename]["count"] = "N/A"
            final_data[key][filename]["concentration"] = "N/A"
            csv_rows.append(row_to_append)
            return 0
    else:  
        try:
            img = np.asarray(Image.open(BytesIO(filedata.read())))
        except:
            final_data[key][filename]["count"] = "N/A"
            final_data[key][filename]["concentration"] = "N/A"
            csv_rows.append(row_to_append)
            return 0
    try:
        cell_results = count_cells(img)
        concentration = calculate_concentration(img, cell_results[0], 271.8, float(depth))
    except:
        final_data[key][filename]["count"] = "N/A"
        final_data[key][filename]["concentration"] = "N/A"
        csv_rows.append(row_to_append)
    if request.form.get("time-unit"):
        if request.form.get("time-"+filename):
            time_x.append(float(request.form.get("time-"+filename)))
            counts_y.append(len(cell_results[0]))
            concentrations_y.append(concentration)
    row_to_append[-2] = str(len(cell_results[0]))
    row_to_append[-1] = str(round(concentration, 2))
    csv_rows[filename] = row_to_append           
    final_data[key][filename] = {}
    final_data[key][filename]["count"] = "{}".format(len(cell_results[0]))
    final_data[key][filename]["concentration"] = "{:.2e}".format(concentration)
    final_data[key][filename]["image"] = image_array_to_base64(img)
    final_data[key][filename]["outlines"] = annotate_image(img, cell_results)
    final_data[key][filename]["circles"] = annotate_image(img, cell_results, True)
    counts.append(len(cell_results[0]))
    concentrations.append(concentration)
    
def boxplot(data, metric):
    units = {"Count": "# of Cells", "Concentration": "Cells / mL"}
    fig = plt.figure()
    ax = plt.axes()
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_xlabel(units[metric])
    ax.yaxis.set_ticklabels([])
    ax.set_title("Cell " + metric + " Box Plot")
    fig.add_axes(ax)
    return image_array_to_base64(get_img_from_fig(fig, False))

def histogram(data, metric):
    units = {"Count": "# of Cells", "Concentration": "Cells / mL"}
    fig = plt.figure()
    ax = plt.axes()
    ax.hist(data)
    ax.set_xlabel(units[metric])
    ax.set_ylabel("Frequency")
    ax.set_title("Cell " + metric + " Histogram")
    fig.add_axes(ax)
    return image_array_to_base64(get_img_from_fig(fig, False))

def const_string(number):
    if number < 0:
        return " - " + str(abs(number))
    elif number > 0:
        return " + " + str(number)
    else:
        return ""

annotate_x = 0.5

def linear_regression(x, y, metric, unit):
    units = {"Count": "# of Cells", "Concentration": "Cells / mL"}
    z = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(z)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, y, "o", x, poly1d_fn(x), "-")
    ax.set_xlabel("Time (" + unit + ")")
    ax.set_ylabel(units[metric])
    ax.set_title("Cell " + metric + " Linear Growth")
    rounded_b = round(z[1], 2)
    linear_formula = f"y = {round(z[0], 2)}x{const_string(rounded_b)}"
    ax.annotate(linear_formula, (np.median(x), z[0] * np.median(x) + z[1]), (max(x) - annotate_x * np.ptp(x), max(y) - 0.1 * np.ptp(y)), arrowprops=dict(facecolor='black', shrink=0.05), fontsize="large")
    fig.add_axes(ax)
    return image_array_to_base64(get_img_from_fig(fig, False))

def exponential_f(x, a, b, c):
    return a * np.exp(b*x) + c

def exponential_regression(x, y, metric, unit):
    units = {"Count": "# of Cells", "Concentration": "Cells / mL"}
    try:
        (a_, b_, c_), _ = optimize.curve_fit(exponential_f, x, y)
    except:
        return False
    y_fit = exponential_f(x, a_, b_, c_)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, y, 'o')
    ax.plot(x, y_fit, '-')
    rounded_c = round(c_, 2)
    string = "$%.2f{e}^{%.2fx}$%s" % (round(a_, 2), round(b_, 2), const_string(rounded_c))
    ax.annotate(string, (np.median(x), exponential_f(np.median(x), a_, b_, c_)), (max(x) - annotate_x * np.ptp(x), max(y) - .1 * np.ptp(y)), arrowprops=dict(facecolor='black', shrink=0.05), fontsize="large")
    ax.set_xlabel("Time (" + unit + ")")
    ax.set_ylabel(units[metric])
    ax.set_title("Cell " + metric + " Exponential Growth")
    fig.add_axes(ax)
    return image_array_to_base64(get_img_from_fig(fig, False))

def logistic_f(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b

def logistic_regression(x, y, metric, unit):
    units = {"Count": "# of Cells", "Concentration": "Cells / mL"}
    try:
        (a_, b_, c_, d_), _ = optimize.curve_fit(logistic_f, x, y, method="trf")
    except:
        return False
    y_fit = logistic_f(x, a_, b_, c_, d_)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, y, 'o')
    ax.plot(x, y_fit, '-')
    string = "y = $\dfrac{%.2f}{1 + {e}^{%.2f * (x%s)}}$%s" % (round(a_, 2), -1 * round(c_, 2), const_string(-1 * round(d_, 2)), const_string(round(b_, 2)))
    ax.annotate(string, (np.median(x), logistic_f(np.median(x), a_, b_, c_, d_)), (max(x) - annotate_x * np.ptp(x), max(y) - .1 * np.ptp(y)), arrowprops=dict(facecolor='black', shrink=0.05), fontsize="large")
    ax.set_xlabel("Time (" + unit + ")")
    ax.set_ylabel(units[metric])
    ax.set_title("Cell " + metric + " Logistic Growth")
    fig.add_axes(ax)
    return image_array_to_base64(get_img_from_fig(fig, False))

def load_graphs(counts, concentrations):
    final_data["graphs"] = {"Count": {}, "Concentration": {}}
    metric_to_list = {"Count": [counts], "Concentration": [concentrations]}
    for metric in ["Count", "Concentration"]:
        final_data["graphs"][metric]["Box Plot"] = boxplot(metric_to_list[metric], metric)
        final_data["graphs"][metric]["Histogram"] = histogram(metric_to_list[metric], metric)
        if request.form.get("time-unit") and len(time_x) > 2:
            metric_to_y = {"Count": np.array(counts_y), "Concentration": np.array(concentrations_y)}
            final_data["graphs"][metric]["Linear Growth"] = linear_regression(np.array(time_x), metric_to_y[metric], metric, request.form.get("time-unit"))
            is_exponential = exponential_regression(np.array(time_x), metric_to_y[metric], metric, request.form.get("time-unit"))
            is_logistic = logistic_regression(np.array(time_x), metric_to_y[metric], metric, request.form.get("time-unit"))
            if is_exponential:
                final_data["graphs"][metric]["Exponential Growth"] = is_exponential
            if is_logistic:
                final_data["graphs"][metric]["Logistic Growth"] = is_logistic
@app.route('/')
def index_get():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def index_post():       
    global final_data
    final_data = {"file_counts": {}, "url_counts": {}}
    counts = []
    concentrations = []
    csv_rows = {"header": ["Name", "Depth (mm)", "Count (cells)", "Concentration (cells / mL)"]}
    if request.form.get("time-unit"):
        csv_rows["header"].insert(1, "Time ("+request.form.get("time-unit")+")")
        global counts_y
        global time_x
        time_x = []
        counts_y = []
        global concentrations_y
        concentrations_y = []
    for filename, file in request.files.items():
        load_response("file_counts", filename, file, counts, concentrations, csv_rows)
    for image_url in json.loads(request.form.get("url")):
        load_response("url_counts", image_url, image_url, counts, concentrations, csv_rows)
    csv_string = ""
    print(list(csv_rows.values()))
    for row in list(csv_rows.values()):
        csv_string += ",".join(row) + "\r\n"
    print(csv_string)
    final_data["csv string"] = csv_string
    final_data["csv"] = csv_rows
    if len(concentrations) > 1 and len(counts) > 1:
        final_data["stats"] = {"Count": {"Mean": str(round(np.mean(counts), 2)), "Range": str(round(np.ptp(counts), 2)), "Standard Deviation": str(round(np.std(counts), 2)), "iqr": list(map(lambda x: str(round(x, 2)), np.percentile(counts, [25, 50, 75])))}, "Concentration": {"Mean": "{:.2e}".format(np.mean(concentrations)), "Range": "{:.2e}".format(np.ptp(concentrations)), "Standard Deviation": "{:.2e}".format(np.std(concentrations)), "iqr": list(map(lambda x: "{:.2e}".format(x), np.percentile(concentrations, [25, 50, 75])))}}
        load_graphs(counts, concentrations)
    else:
        final_data["stats"] = "No data available"
    return final_data
