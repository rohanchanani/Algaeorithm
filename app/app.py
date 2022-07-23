from io import BytesIO
import boto3
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import os
import numpy as np
from scipy import optimize, stats
import cv2
import json
import base64
import math
import os
import tensorflow as tf
from models.object_detection.utils import label_map_util
from models.object_detection.utils import visualization_utils as viz_utils
from models.object_detection.builders import model_builder
from models.object_detection.utils import config_util
import random

app = Flask(__name__)

# Load pipeline config and build a detection model
cell_configs = config_util.get_configs_from_pipeline_file(os.path.join("app", "static", "cells", "pipeline.config"))
cell_detection_model = model_builder.build(model_config=cell_configs['model'], is_training=False)

# Restore checkpoint
cell_ckpt = tf.compat.v2.train.Checkpoint(model=cell_detection_model)
cell_ckpt.restore(os.path.join("app", "static", "cells", 'ckpt-3')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(os.path.join("app", "static", "cells", "label_map.pbtxt"))

def cells_fn(image):
    image, shapes = cell_detection_model.preprocess(image)
    prediction_dict = cell_detection_model.predict(image, shapes)
    detections = cell_detection_model.postprocess(prediction_dict, shapes)
    return detections

#Load pipeline config and build a detection model
diatom_configs = config_util.get_configs_from_pipeline_file(os.path.join("app", "static", "diatom", "pipeline.config"))
diatom_detection_model = model_builder.build(model_config=diatom_configs['model'], is_training=False)

# Restore checkpoint
diatom_ckpt = tf.compat.v2.train.Checkpoint(model=diatom_detection_model)
diatom_ckpt.restore(os.path.join("app", "static", "diatom", 'ckpt-3')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(os.path.join("app", "static", "cells", "label_map.pbtxt"))

def diatom_fn(image):
    image, shapes = diatom_detection_model.preprocess(image)
    prediction_dict = diatom_detection_model.predict(image, shapes)
    detections = diatom_detection_model.postprocess(prediction_dict, shapes)
    return detections

def cells_fn(image):
    image, shapes = cell_detection_model.preprocess(image)
    prediction_dict = cell_detection_model.predict(image, shapes)
    detections = cell_detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load pipeline config and build a detection model
crop_configs = config_util.get_configs_from_pipeline_file(os.path.join("app", "static", "crop", "pipeline.config"))
crop_detection_model = model_builder.build(model_config=crop_configs['model'], is_training=False)

# Restore checkpoint
crop_ckpt = tf.compat.v2.train.Checkpoint(model=crop_detection_model)
crop_ckpt.restore(os.path.join("app", "static", "crop", 'ckpt-3')).expect_partial()

def crop_fn(image):
    image, shapes = crop_detection_model.preprocess(image)
    prediction_dict = crop_detection_model.predict(image, shapes)
    detections = crop_detection_model.postprocess(prediction_dict, shapes)
    return detections

def auto_crop(image_data, threshold = 0.8):
    image_np = np.array(image_data)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = crop_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    if detections["detection_scores"][0] < threshold:
        return image_np
    
    height = image_np.shape[0]
    width = image_np.shape[1]
    crop_box = detections["detection_boxes"][0]

    return cv2.cvtColor(image_np[int(crop_box[0]*height):int(crop_box[2]*height), int(crop_box[1]*width):int(crop_box[3]*width), :], cv2.COLOR_BGR2RGB)

def chlamy_concentration(cell_boxes, total_area, cell_volume=271.8, depth=0.1):
    avg_radius = np.mean(list(map(lambda box: (abs((box[3] - box[1]) / 2) + abs((box[2] - box[0]) / 2)) / 2, cell_boxes)))
    expected_radius = (cell_volume * 3 / 4 / math.pi) ** (1/3)
    return len(cell_boxes) / (((expected_radius ** 2) / (avg_radius ** 2) * total_area) * 10 ** -9 * depth)

def diatom_concentration(cell_boxes, image_area, cell_length, mm_depth):
    avg_length = np.mean(list(map(lambda x: ((x[2] - x[0]) ** 2 + (x[3] - x[1]) ** 2) ** 0.5, cell_boxes)))
    return len(cell_boxes) / (((cell_length / avg_length) ** 2 * image_area) * 10 ** -9 * mm_depth)

def intersection_over_union(box1, box2):
    box1_x1 = box1[1]
    box1_y1 = box1[0]
    box1_x2 = box1[3]
    box1_y2 = box1[2]
    box2_x1 = box2[1]
    box2_y1 = box2[0]
    box2_x2 = box2[3]
    box2_y2 = box2[2]
    x1 = np.max(np.array([box1_x1, box2_x1]))
    y1 = np.max(np.array([box1_y1, box2_y1]))
    x2 = np.min(np.array([box1_x2, box2_x2]))
    y2 = np.min(np.array([box1_y2, box2_y2]))
    intersection = np.max(np.array([x2 - x1, 0])) * np.max(np.array([y2 - y1, 0]))
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def suppress_boxes(detections, confidence_threshold=0.1, iou_threshold=0.5):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    final_boxes = enumerate(detections["detection_boxes"])

    final_boxes = sorted([bbox for bbox in final_boxes if detections["detection_scores"][bbox[0]] > confidence_threshold], key=lambda bbox: detections["detection_scores"][bbox[0]], reverse=True)
    if len(final_boxes) < 2:
        final_boxes = list(enumerate(detections["detection_boxes"]))
    remaining_bboxes = []
    while final_boxes:
        top_bbox = final_boxes.pop(0)
        final_boxes = [bbox for bbox in final_boxes if intersection_over_union(top_bbox[1], bbox[1]) < iou_threshold]
        remaining_bboxes.append(top_bbox[1])

    box_areas = list(map(lambda x: (x[3] - x[1]) * (x[2] - x[0]), remaining_bboxes))
    adjusted_bboxes = [bbox for idx, bbox in enumerate(remaining_bboxes) if abs(stats.zscore(box_areas)[idx]) < 3]
    return adjusted_bboxes

def count_concentration_detections(image, cell_type, image_name, threshold=0.1, cell_volume=271.8, cell_length=16, depth=0.1):
    cropped_image = auto_crop(image)
    img_height = cropped_image.shape[0]
    img_width = cropped_image.shape[1]
    patch_size = round(np.mean(np.array([img_height, img_width])) * 0.4)
    total_area = patch_size ** 2
    patch_results = {}
    width_offset = max(round((img_width - patch_size) / 2), 0)
    height_offset = max(round((img_height - patch_size) / 2), 0)
    img_patch = cropped_image[height_offset:min(img_height, height_offset + patch_size), width_offset:min(img_width, width_offset + patch_size), :]
    #image_to_upload = np.array(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
    #image_to_upload = Image.fromarray(image_to_upload.astype('uint8'))
    #in_mem_file = BytesIO()
    #image_to_upload.save(in_mem_file, format="jpeg")
    #in_mem_file.seek(0)
    #client = boto3.client("s3")
    #client.put_object(Body=in_mem_file, Bucket="algaeorithm-photos", Key=image_name)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_patch, 0), dtype=tf.float32)
    if cell_type=="chlamy":
        detections = cells_fn(input_tensor)
    else:
        detections = diatom_fn(input_tensor)

    patch_width = img_patch.shape[1]
    patch_height = img_patch.shape[0]

    all_boxes = suppress_boxes(detections, threshold)
        
    scaled_boxes = list(map(lambda x: [x[0] * patch_height, x[1] * patch_width, x[2]*patch_height, x[3] * patch_width], all_boxes))
    patch_results["patch"] = img_patch
    patch_results["detections"] = np.array(all_boxes)
    estimated_cell_count = len(scaled_boxes) * img_height * img_width / total_area
    concentration = chlamy_concentration(scaled_boxes, total_area, cell_volume, depth) if cell_type=="chlamy" else diatom_concentration(scaled_boxes, total_area, cell_length, depth)
    return estimated_cell_count, concentration, patch_results

def annotate_patch(patch_results):
    patch, detections = patch_results["patch"], patch_results["detections"]
    patch_with_detections = patch.copy()
    if detections.shape[0] > 0:
        viz_utils.draw_bounding_boxes_on_image_array(patch_with_detections, detections)
    return image_array_to_base64(np.array(cv2.cvtColor(patch_with_detections, cv2.COLOR_BGR2RGB)))

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

def image_array_to_base64(arr):
    img = Image.fromarray(arr.astype('uint8'))
    file_object = BytesIO()
    try:
        img.save(file_object, format="JPEG")
    except:
        img.save(file_object, format="PNG")
    img_str = base64.b64encode(file_object.getvalue())
    file_object.close()
    return "data:image/jpeg;base64,"+img_str.decode("utf-8")

def calculate_concentration(image, contours, cell_volume, mm_depth):
    avg_area = np.mean(list(map(cv2.contourArea, contours)))
    expected_area = (((cell_volume * 3 / 4 / math.pi) ** (1/3)) ** 2) * math.pi
    return len(contours) / ((expected_area / avg_area * image.shape[0] * image.shape[1]) * 10 ** -8 * mm_depth / 10)

def load_response(key, filename, filedata, counts, concentrations, csv_rows, image_type):
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
            img = np.asarray(cv2.imread(filename))
        except:
            final_data[key][filename]["count"] = "N/A"
            final_data[key][filename]["concentration"] = "N/A"
            csv_rows[filename] = row_to_append 
            return 0
    else:  
        try:
            img = np.asarray(Image.open(BytesIO(filedata.read())))
        except:
            print("error reading file")
            final_data[key][filename]["count"] = "N/A"
            final_data[key][filename]["concentration"] = "N/A"
            csv_rows[filename] = row_to_append 
            return 0
    #try: 
    threshold = 0.5 if image_type=="chlamy" else 0.1
    estimated_count, concentration, patch_results = count_concentration_detections(img, image_type, filename, threshold)
    #except:
    #    final_data[key][filename]["count"] = "N/A"
    #    final_data[key][filename]["concentration"] = "N/A"
    #    final_data[key][filename]["image"] = image_array_to_base64(img)
    #    csv_rows[filename] = row_to_append
    #    return 0 
    if request.form.get("time-unit"):
        if request.form.get("time-"+filename):
            time_x.append(float(request.form.get("time-"+filename)))
            counts_y.append(estimated_count)
            concentrations_y.append(concentration)
    row_to_append[-2] = str(round(estimated_count, 2))
    row_to_append[-1] = str(round(concentration, 2))
    csv_rows[filename] = row_to_append           
    final_data[key][filename] = {}
    final_data[key][filename]["count"] = "{:.2e}".format(estimated_count)
    final_data[key][filename]["concentration"] = "{:.2e}".format(concentration)
    final_data[key][filename]["image"] = image_array_to_base64(img)
    final_data[key][filename]["output"] = annotate_patch(patch_results)
    counts.append(estimated_count)
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
    return render_template("bad_index.html")

@app.route("/health")
def health_check():
    return "200 OK"

@app.route("/content")
def content():
    return render_template("content.html")

@app.route("/press")
def press():
    return render_template("press.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'logos'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

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
    image_type = request.form.get("cell_type")
    for filename, file in request.files.items():
        load_response("file_counts", filename, file, counts, concentrations, csv_rows, image_type)
    for image_url in json.loads(request.form.get("url")):
        load_response("url_counts", image_url, image_url, counts, concentrations, csv_rows, image_type)
    csv_string = ""
    for row in list(csv_rows.values()):
        csv_string += ",".join(row) + "\r\n"
    final_data["csv string"] = csv_string
    final_data["csv"] = csv_rows
    if len(concentrations) > 1 and len(counts) > 1:
        final_data["stats"] = {"Count": {"Mean": str(round(np.mean(counts), 2)), "Range": str(round(np.ptp(counts), 2)), "Standard Deviation": str(round(np.std(counts), 2)), "iqr": list(map(lambda x: str(round(x, 2)), np.percentile(counts, [25, 50, 75])))}, "Concentration": {"Mean": "{:.2e}".format(np.mean(concentrations)), "Range": "{:.2e}".format(np.ptp(concentrations)), "Standard Deviation": "{:.2e}".format(np.std(concentrations)), "iqr": list(map(lambda x: "{:.2e}".format(x), np.percentile(concentrations, [25, 50, 75])))}}
        load_graphs(counts, concentrations)
    else:
        final_data["stats"] = "No data available"
    return final_data
