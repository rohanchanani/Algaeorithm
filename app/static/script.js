let imageIndex = 0;
let patchIndices = [];
let imagesList = [];
let fileList = document.getElementById("fileList");
let fileInput = new FormData();
let urlToSubmit = [];
let parsedResponse = {};

const checkBox = (boxId) => {
    document.getElementById(boxId).checked = !document.getElementById(boxId).checked;
}

const checkRadio = (radioId) => {
    document.getElementById(radioId).checked = true;
    if (radioId == "consistent" || radioId == "varied") {
        changeDepth();
    }
}

const changeImageType = (imageType) => {
    checkRadio(imageType);
    document.getElementById("img-"+imageType).setAttribute("class", "selected-photo-type photo-type");
    if (imageType == "clear") {
        document.getElementById("img-"+"hemocytometer").setAttribute("class", "photo-type");
    } else {
        document.getElementById("img-"+"clear").setAttribute("class", "photo-type");
    }
}

const getRadioValue = (radioName) => {
    for (let radioButton of document.getElementsByName(radioName)) {
        if (radioButton.checked) {
            return radioButton.value;
        }
    }
}

const isConsistentDepth = () => {
    let depthSensitive = getRadioValue("depth-sensitive") == "consistent";
    return depthSensitive;
}

const changeInterval = () => {
    if (document.getElementById("interval").checked) {
        document.getElementById("interval-label").setAttribute("class", "settings-label");
    } else {
        document.getElementById("interval-label").setAttribute("class", "hidden");
    }
    changeTime();
}

const changeIntervalValue = () => {
    labelElement = document.getElementById("interval-label")
    labelElement.innerHTML = labelElement.children[0].outerHTML + " " + document.getElementById("time-units").value;
}

const changeTime = () => {
    document.getElementById("time-setting").setAttribute("class", "setting");
    if (document.getElementById("time-sensitive").checked && !document.getElementById("interval").checked) {
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.tagName.toLowerCase() != "div") {
                continue;
            }
            previewElement.children[1].setAttribute("class", "specific-inputs");
            previewElement.children[1].children[0].setAttribute("class", "preview-time")
        }
    } else {
        if (!document.getElementById("time-sensitive").checked) {
            document.getElementById("time-setting").setAttribute("class", "hidden");
        }
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.tagName.toLowerCase() != "div") {
                continue;
            }
            previewElement.children[1].children[0].setAttribute("class", "hidden");
            if (isConsistentDepth()) {
                previewElement.children[1].setAttribute("class", "hidden");
            }
        }
    }
}

const changeDepth = () => {
    if (!isConsistentDepth()) {
        document.getElementById("default-label").setAttribute("class", "settings-label");
        document.getElementById("all-depth-label").setAttribute("class", "hidden");
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.tagName.toLowerCase() != "div") {
                continue;
            }
            previewElement.children[1].setAttribute("class", "specific-inputs");
            previewElement.children[1].children[1].setAttribute("class", "preview-depth")
        }
    } else {
        document.getElementById("default-label").setAttribute("class", "hidden");
        document.getElementById("all-depth-label").setAttribute("class", "settings-label");
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.tagName.toLowerCase() != "div") {
                continue;
            }
            previewElement.children[1].children[1].setAttribute("class", "hidden");
            if (!document.getElementById("time-sensitive").checked) {
                previewElement.children[1].setAttribute("class", "hidden");
            }
        }
    }
}

const createPreview = (imageName, imageSrc, imageType="file") => {
    let previewDiv = document.createElement("div");
    previewDiv.setAttribute("class", "preview");
    previewDiv.setAttribute("id", "img-"+imageName);
    let previewImage = document.createElement("img");
    previewImage.setAttribute("class", "preview-image");
    previewImage.src = imageSrc;
    previewImage.title = imageName;
    previewDiv.appendChild(previewImage);
    previewDiv.innerHTML += document.getElementById("sample-preview").children[1].outerHTML;
    let deleteButton = document.createElement("button");
    deleteButton.setAttribute("class", "delete-photo");
    deleteButton.setAttribute("onclick", "deleteImage('" + imageName + "', '" + imageType + "')");
    previewDiv.appendChild(deleteButton);
    document.getElementById("previews").appendChild(previewDiv);
}

const addFileToList = () => {
    const stagingInput = document.getElementById("staging");
    if (stagingInput.files.length===0) {
        alert("No files selected");
    }
    else {
        for (let newFile of stagingInput.files) {
            if (fileInput.has(newFile.name)) {
                alert(newFile.name + " has already been uploaded");
                continue;
            }
            fileInput.append(newFile.name, newFile);
            createPreview(newFile.name, URL.createObjectURL(newFile));
        }
    }
    document.getElementById("analyze-button").setAttribute("class", "analyze-button");
}

const deleteImage = (imageName, imageType="file") => {
    let imageElement = document.getElementById("img-"+imageName);
    imageElement.parentElement.removeChild(imageElement);
    if (imageType=="file") {
        fileInput.delete(imageName);
    }
    else {
        urlToSubmit.splice(urlToSubmit.indexOf(imageName), 1);
    }
    if (document.getElementById("previews").children.length == 3) {
        document.getElementById("analyze-button").setAttribute("class", "hidden");
    }
}

const getFilename = (fullName, nameType) => {
    if (nameType=="url") {
        return fullName.slice(fullName.lastIndexOf("/")+1);
    } else {
        return fullName;
    }
}

const changePalette = (elementId) => {
    if (document.getElementById(elementId).getAttribute("class") == "palette") {
        return;
    }
    let allPalettes = ["inputInfo", "overview", "resultsInfo", "table"];
    allPalettes.splice(allPalettes.indexOf(elementId), 1)
    for (let deleteId of allPalettes) {
        document.getElementById(deleteId).setAttribute("class", "hidden");
    }
    document.getElementById(elementId).setAttribute("class", "palette");
}

const changeImage = (change) => {
    displayImage(imageIndex + change, "Patches");
}

const changePatch = (change) => {
    setPatch(imageIndex, patchIndices[imageIndex] + change);
}

const clickDownload = () => {
    document.getElementById("download-link").click();
}

const setPatch = (imageIndex, patchIndex) => {
    patchIndices[imageIndex] = patchIndex;
    [imageName, imageOutput, dataType] = imagesList[imageIndex]
    resultsImage.setAttribute("src", imageOutput["patches"][patchIndex]);
    imageFilename = getFilename(imageName, dataType)
    patchName = imageFilename.slice(0, imageFilename.lastIndexOf(".")) + "_patch" + (patchIndex+1).toString() + imageFilename.slice(imageFilename.lastIndexOf("."));
    resultsImage.setAttribute("title", "Download "+ patchName);
    resultsImage.setAttribute("class", "patch");
    document.getElementById("lastPatch").setAttribute("class", "hidden");
    document.getElementById("nextPatch").setAttribute("class", "hidden");
    if (patchIndex) {
        document.getElementById("lastPatch").setAttribute("class", "patch-nav");
    }
    if (patchIndex < imageOutput["patches"].length - 1) {
        document.getElementById("nextPatch").setAttribute("class", "patch-nav");
    }
    let download_link = document.getElementById("download-link");
    download_link.setAttribute("href", imageOutput["patches"][patchIndex]);
    download_link.setAttribute("download", patchName);
}

const setImage = (index, method, dataType) => {
    let resultsImage = document.getElementById("resultsImage");
    [imageName, imageOutput, dataType] = imagesList[index];
    if (method=="Image") {
        resultsImage.setAttribute("src", imageOutput["image"]);
        resultsImage.setAttribute("title", "Download "+ imageName);
        resultsImage.setAttribute("class", "algae");
        document.getElementById("lastPatch").setAttribute("class", "hidden");
        document.getElementById("nextPatch").setAttribute("class", "hidden");
        let download_link = document.getElementById("download-link");
        download_link.setAttribute("href", imageOutput["image"]);
        download_link.setAttribute("download", getFilename(imageName, dataType));
    } else {
        setPatch(index, patchIndices[index], dataType);
    }
}

const displayImage = (newIndex, method)  => {
    document.getElementById("show-images").setAttribute("class", "selected-tab");
    document.getElementById("show-overview").setAttribute("class", "tab");
    document.getElementById("show-table").setAttribute("class", "tab");
    imageIndex = newIndex;
    changePalette("resultsInfo");
    let [imageName, imageOutput, dataType] = imagesList[imageIndex];
    document.getElementById("img-nav").setAttribute("class", "hidden");
    document.getElementById("lastImage").setAttribute("class", "invisible");
    document.getElementById("nextImage").setAttribute("class", "invisible");
    document.getElementById("show-pictures").innerHTML = "";
    document.getElementById("image-number").innerHTML = "";
    document.getElementById("resultsCount").innerHTML = imageOutput["count"];
    document.getElementById("resultsConcentration").innerHTML = imageOutput["concentration"]
    
    if (imageOutput["count"].includes("N/A")) {
        document.getElementById("resultsImage").removeAttribute("src");
        if (Object.keys(imageOutput).length > 2) {
            setImage(newIndex, "Image", dataType);
        }
    }
    else {
        document.getElementById("resultsCount").innerHTML += "<span class='results-unit'> cells</span>";
        document.getElementById("resultsConcentration").innerHTML += "<span class='results-unit'> cells / mL</span>";
        let methods = ["Image", "Patches"];
        setImage(newIndex, method, dataType);
        for (let i = 0; i < 2; i++) {
            let anchor = document.createElement("a");
            anchor.setAttribute("onclick", `displayImage(${newIndex}, '${methods[i]}')`);
            anchor.innerHTML = methods[i];
            if (methods[i]==method) {
                anchor.setAttribute("class", "image-type clicked");
            } else {
                anchor.setAttribute("class", "image-type");
            }
            document.getElementById("show-pictures").appendChild(anchor);
        }
    }
    let navbar = document.getElementById("img-nav");
    navbar.setAttribute("class", "img-nav");
    if (newIndex) {
        let anchor = document.getElementById("lastImage");
        anchor.setAttribute("onclick", "changeImage(-1)");
        anchor.setAttribute("class", "last");
    }
    document.getElementById("image-number").innerHTML = (newIndex + 1).toString() + " of " + imagesList.length.toString();
    if (newIndex < imagesList.length - 1) {
        let anchor = document.getElementById("nextImage");
        anchor.setAttribute("onclick", "changeImage(1)");
        anchor.setAttribute("class", "next");
    }    
}

const setGraph = (selectedGraph, metric) => {
    let graphList = Object.keys(parsedResponse["graphs"][metric]);
    document.getElementById(metric.toLowerCase()+"s-graph").src = parsedResponse["graphs"][metric][selectedGraph];
    let changeGraph = document.getElementById(metric.toLowerCase()+"s-change");
    changeGraph.innerHTML = "";
    for (let graphType of graphList) {
        let graphAnchor = document.createElement("a");
        graphAnchor.innerHTML = graphType;
        graphAnchor.setAttribute("onclick", "setGraph('" + graphType + "', '" + metric + "')");
        if (graphType==selectedGraph) {
            graphAnchor.setAttribute("class", "graph-button clicked");
        } else {
            graphAnchor.setAttribute("class", "graph-button");
        }
        changeGraph.appendChild(graphAnchor);
    }
}

const setTable = () => {
    changePalette("table");
    document.getElementById("show-table").setAttribute("class", "selected-tab");
    document.getElementById("show-images").setAttribute("class", "tab");
    document.getElementById("show-overview").setAttribute("class", "tab");
}

const setOverview = () => {
    changePalette("overview");
    document.getElementById("tab-nav").setAttribute("class", "tab-nav");
    document.getElementById("show-overview").setAttribute("class", "selected-tab");
    document.getElementById("show-images").setAttribute("class", "tab");
    document.getElementById("show-table").setAttribute("class", "tab");
    if (parsedResponse["stats"] == "No data available") {
        displayImage(0, "Patches");
        document.getElementById("tab-nav").setAttribute("class", "hidden");
    } else {
        document.getElementById("counts-stats").innerHTML = "";
        document.getElementById("concentrations-stats").innerHTML = "";
        let units = {"Count": " cells", "Concentration": " cells / mL"};
        for (let metric of ["Count", "Concentration"]) {
            let sectionDiv = document.getElementById(metric.toLowerCase()+"s-stats");
            for (let stat of ["Mean", "Range", "Standard Deviation"]) {
                let statSection = document.createElement("div");
                statSection.setAttribute("class", "stats-section");
                statSection.innerHTML = stat + ": " + parsedResponse["stats"][metric][stat];
                sectionDiv.appendChild(statSection);
            }
            let medianBox = document.createElement("div");
            medianBox.setAttribute("class", "stats-section");
            let iqrBox = document.createElement("div");
            iqrBox.setAttribute("class", "stats-section");
            iqrList = parsedResponse["stats"][metric]["iqr"]
            medianBox.innerHTML = "Median: " + iqrList[1];
            iqrBox.innerHTML = "Interquartile Range: " + iqrList[2] + " - " + iqrList[0];
            sectionDiv.appendChild(medianBox);
            sectionDiv.appendChild(iqrBox);
            let graphs = Object.keys(parsedResponse["graphs"]["Count"]);
            setGraph(graphs[0], metric);
        }
    }
}

const addInformation = () => {
    let csvRows = parsedResponse["csv"]
    let table = document.getElementById("csv-table");
    let thead = document.createElement("thead");
    for (let header of csvRows["header"]) {
        let td = document.createElement("td");
        td.innerHTML = header;
        thead.appendChild(td)
    }
    table.appendChild(thead);
    for (let [data, dataType] of [[parsedResponse["file_counts"], "file"], [parsedResponse["url_counts"], "url"]]) {
        for (const [imageName, imageOutput] of Object.entries(data)) {
            imagesList.push([imageName, imageOutput, dataType]);
            patchIndices.push(0);
            let linkIndex = imagesList.length - 1;
            let tr = document.createElement("tr");
            tr.setAttribute("onclick", "displayImage("+linkIndex+", 'Patches')");
            tr.setAttribute("title", imageName);
            for (let columnNum in csvRows[imageName]) {
                let td = document.createElement("td");
                td.setAttribute("class", "data-"+columnNum.toString());
                td.innerHTML = csvRows[imageName][columnNum];
                tr.appendChild(td);
            }
            table.appendChild(tr);
        }
    }
    document.getElementById("download-csv").setAttribute("href", "data:text/csv;charset=utf-8,"+parsedResponse["csv string"]);
}

const checkIfNumber = (value) => {
    if (!value) {
        return false;
    }
    try {
        parseFloat(value);
    } catch (error) {
        return false;
    }
    return true;
}

const checkSensitives = () => {
    imageColor = 2;
    if (document.getElementById("hemocytometer").checked) {
        imageColor = 0;
    }
    fileInput.append("color", imageColor);
    if (document.getElementById("time-sensitive").checked) {
        if (!document.getElementById("time-unit").value) {
            alert("Please enter a valid unit for time sensitive data");
            return false;   
        }
        if (document.getElementById("interval").checked) {
            if (!checkIfNumber(document.getElementById("time-interval").value)) {
                alert("Please enter a valid interval");
                return false;
            } else {
                let intervalDay = 0;
                for (let previewElement of document.getElementById("previews").children) {
                    if (previewElement.id == "sample-preview" || previewElement.tagName.toLowerCase() != "div") {
                        continue;
                    }
                    fileInput.append("time-"+previewElement.children[0].title, intervalDay.toString());
                    intervalDay += parseFloat(document.getElementById("time-interval").value);
                }
            }
        } else {
            for (let previewElement of document.getElementById("previews").children) {
                if (previewElement.id == "sample-preview" || previewElement.tagName.toLowerCase() != "div") {
                    continue;
                }
                if (checkIfNumber(previewElement.children[1].children[0].children[0].value)) {
                    fileInput.append("time-"+previewElement.children[0].title, previewElement.children[1].children[0].children[0].value);
                }
            }
        }
        fileInput.append("time-unit", document.getElementById("time-unit").value);
    }
    if (!isConsistentDepth()) {
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.id == "sample-preview" || previewElement.tagName.toLowerCase() != "div") {
                continue;
            }
            if (checkIfNumber(previewElement.children[1].children[1].children[0].value)) {
                fileInput.append("depth-"+previewElement.children[0].title, previewElement.children[1].children[1].children[0].value);
            }
            else {
                if (!checkIfNumber(document.getElementById("default-depth").value)) {
                    alert("Please enter a depth for each picture or a valid default depth");
                    return false;
                } 
                else {
                    fileInput.append("depth-"+previewElement.children[0].title, document.getElementById("default-depth").value);
                }
            }
        }
        return true;
    } else {
        if (!checkIfNumber(document.getElementById("all-depth").value)) {
            alert("Please enter a valid depth");
            return false;
        } 
        else {
            fileInput.append("depth", document.getElementById("all-depth").value);
            return true;
        }
    }
}

const cancelRequest = () => {
    document.getElementById("analyze-button").setAttribute("class", "analyze-button");
    document.getElementById("loader-wrapper").setAttribute("class", "hidden");
}

const loadInformation = () => {
    if (!document.getElementById("previews").children.length) {
        alert("No images added");
        return;
    }
    let request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (request.readyState == 4 && document.getElementById("loader-wrapper").getAttribute("class") != "hidden") {
            imagesIndex = 0;
            imagesList = [];
            patchIndices = [];
            parsedResponse = JSON.parse(request.response);
            addInformation();
            document.getElementById("loader-wrapper").setAttribute("class", "hidden");
            setOverview(parsedResponse);
        }    
    }
    if (!checkSensitives()) {
        return;
    }
    fileInput.append("url", JSON.stringify(urlToSubmit));
    fileInput.append("num_patches", document.getElementById("num-patches").value);
    request.open("POST", "/");
    request.send(fileInput);
    document.getElementById("analyze-button").setAttribute("class", "hidden");
    document.getElementById("loader-wrapper").removeAttribute("class");
}
document.addEventListener("keydown", function(event) {
    if (event.key === "Enter" && event.ctrlKey) {
        loadInformation();
    }
});
document.getElementById("staging").addEventListener("change", addFileToList);
document.getElementById("time-sensitive").addEventListener("change", changeTime);
document.getElementById("interval").addEventListener("change", changeInterval);
document.getElementById("time-unit").addEventListener("keyup", changeIntervalValue);