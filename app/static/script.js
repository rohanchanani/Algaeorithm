let imageIndex = 0;
let imagesList = [];
let fileList = document.getElementById("fileList");
let urlList = document.getElementById("urlList");
let fileInput = new FormData();
let urlToSubmit = [];
let parsedResponse = {};

const changeAdvanced = () => {
    if (document.getElementById("advanced-options").innerHTML == "Advanced") {
        document.getElementById("advanced-input").setAttribute("class", "advanced-input");
        document.getElementById("advanced-options").innerHTML = "Hide Advanced";
    } else {
        document.getElementById("advanced-input").setAttribute("class", "hidden");
        document.getElementById("advanced-options").innerHTML = "Advanced";
    }
}

const updateTimePreviews = () => {
    for (let previewElement of document.getElementById("previews").children) {
        if (previewElement.getAttribute("id") == "sample-preview") {
            previewElement.children[0].setAttribute("class", "specific-inputs");
            continue;
        }
        previewElement.children[1].setAttribute("class", "specific-inputs");
        previewElement.children[1].children[0].setAttribute("class", "preview-time");
        let currentVal = previewElement.children[1].children[0].children[0].value;
        previewElement.children[1].children[0].innerHTML = previewElement.children[1].children[0].children[0].outerHTML + document.getElementById("time-unit").value;
        previewElement.children[1].children[0].children[0].value = currentVal;
    }
}

const changeTime = () => {
    if (document.getElementById("time-sensitive").checked) {
        document.getElementById("time-unit-box").setAttribute("class", "inner-advanced-box");
        updateTimePreviews();
    } else {
        document.getElementById("time-unit-box").setAttribute("class", "invisible margin-five");
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.getAttribute("id") == "sample-preview") {
                if (document.getElementById("depth-sensitive").checked) {
                    previewElement.children[0].setAttribute("class", "hidden");
                }
                continue;
            }
            if (document.getElementById("depth-sensitive").checked) {
                previewElement.children[1].setAttribute("class", "hidden");
            }
            previewElement.children[1].children[0].setAttribute("class", "hidden");
            let currentVal = previewElement.children[1].children[0].children[0].value;
            previewElement.children[1].children[0].innerHTML = previewElement.children[1].children[0].children[0].outerHTML;
            previewElement.children[1].children[0].children[0].value = currentVal;
        }
    }
}

const changeDepth = () => {
    if (document.getElementById("depth-sensitive").checked) {
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.getAttribute("id") == "sample-preview") {
                if (!document.getElementById("time-sensitive").checked) {
                    previewElement.children[0].setAttribute("class", "hidden");
                }
                continue;
            }
            if (!document.getElementById("time-sensitive").checked) {
                previewElement.children[1].setAttribute("class", "hidden");
            }
            previewElement.children[1].children[1].setAttribute("class", "hidden");
        }
    } else {
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.getAttribute("id") == "sample-preview") {
                previewElement.children[0].setAttribute("class", "specific-inputs");
                continue;
            }
            previewElement.children[1].setAttribute("class", "specific-inputs");
            previewElement.children[1].children[1].setAttribute("class", "preview-depth");
        }
    }
}

const createPreview = (imageName, imageType, imageSrc) => {
    let previewDiv = document.createElement("div");
    previewDiv.setAttribute("class", "preview");
    previewDiv.setAttribute("id", "img-"+imageName);
    let imageContainer = document.createElement("div");
    imageContainer.setAttribute("class", "preview-image-container");
    let previewImage = document.createElement("img");
    previewImage.setAttribute("class", "preview-image");
    previewImage.src = imageSrc;
    previewImage.title = imageName;
    previewImage.setAttribute("onclick", "deleteImage('" + imageType + "', '" + imageName + "')");
    imageContainer.appendChild(previewImage);
    previewDiv.appendChild(imageContainer);
    previewDiv.innerHTML += document.getElementById("sample-preview").innerHTML;
    if (document.getElementById("time-sensitive").checked) {
        previewDiv.children[1].children[0].setAttribute("class", "preview-time");
        previewDiv.children[1].children[0].innerHTML += " " + document.getElementById("time-unit").value;
    }
    if (!document.getElementById("depth-sensitive").checked) {
        previewDiv.children[1].children[1].setAttribute("class", "preview-depth");
    }
    document.getElementById("previews").appendChild(previewDiv);
    document.getElementById("lower").setAttribute("class", "lower");
    document.getElementById("upper").style.setProperty("margin-bottom", "0vh");
}


const addURL = () => {
    if (!document.getElementById("urlInput").value){
        alert("Please enter a URL");
    }
    else {
        let givenURL = document.getElementById("urlInput").value;
        if (urlToSubmit.includes(givenURL)) {
            alert("Please enter a unique URL");
            document.getElementById("urlInput").value = "";
            return;
        }
        urlToSubmit.push(givenURL);
        createPreview(givenURL, 'url', givenURL);
        document.getElementById("urlInput").value = "";
    }
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
            createPreview(newFile.name, "file", URL.createObjectURL(newFile));
        }
    }
}

const deleteImage = (imageType, imageName) => {
    let imageElement = document.getElementById("img-"+imageName);
    imageElement.parentElement.removeChild(imageElement);
    if (imageType=="file") {
        fileInput.delete(imageName);
    }
    else {
        urlToSubmit.splice(urlToSubmit.indexOf(imageName), 1);
    }
    if (document.getElementById("previews").children.length == 1) {
        document.getElementById("lower").setAttribute("class", "hidden");
        document.getElementById("upper").style.setProperty("margin-bottom", "20vh");
    }
}

const getFilename = (fullName, method, nameType) => {
    let filename = "";
    if (nameType=="url") {
        filename = fullName.slice(fullName.lastIndexOf("/")+1);
    } else {
        filename = fullName;
    }
    return filename.slice(0, filename.lastIndexOf(".")) + method + filename.slice(filename.lastIndexOf("."));
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
    displayImage(imageIndex + change, "Outlines");
}

const clickDownload = () => {
    document.getElementById("download-link").click();
}

const setImage = (index, method, type) => {
    let resultsImage = document.getElementById("resultsImage");
    resultsImage.setAttribute("src", "data:image/jpeg;base64,"+imagesList[index][0][method.toLowerCase()]);
    resultsImage.setAttribute("title", "Download "+imagesList[index][1]);
    let download_link = document.getElementById("download-link");
    download_link.setAttribute("href", "data:image/jpeg;base64,"+imagesList[index][0][method.toLowerCase()]);
    download_link.setAttribute("download", getFilename(imagesList[index][1], method, type));
}

const displayImage = (imagesIndex, method)  => {
    document.getElementById("show-images").setAttribute("class", "selected-tab");
    document.getElementById("show-overview").setAttribute("class", "tab");
    document.getElementById("show-table").setAttribute("class", "tab");
    imageIndex = imagesIndex;
    changePalette("resultsInfo");
    let count = imagesList[imagesIndex][0]["count"];
    let location = imagesList[imagesIndex][1];
    let type = imagesList[imagesIndex][2];
    let resultsElement = document.getElementById("visual");
    let i = 0;
    while (i < resultsElement.childNodes.length) {
        if (resultsElement.childNodes[i].nodeName.toLowerCase() == "a") {
            resultsElement.childNodes[i].remove();
        }
        else {
            i++;
        }
    }
    document.getElementById("img-nav").setAttribute("class", "hidden");
    document.getElementById("lastImage").setAttribute("class", "invisible");
    document.getElementById("nextImage").setAttribute("class", "invisible");
    document.getElementById("show-pictures").innerHTML = "";
    document.getElementById("image-number").innerHTML = "";
    document.getElementById("resultsCount").innerHTML = count;
    document.getElementById("resultsConcentration").innerHTML = imagesList[imagesIndex][0]["concentration"];
    
    if (count.includes("N/A")) {
        document.getElementById("resultsImage").removeAttribute("src");
        if (Object.keys(imagesList[imagesIndex][0]).length > 1) {
            setImage(imagesIndex, "Image", type);
        }
    }
    else {
        document.getElementById("resultsCount").innerHTML += "<span class='results-unit'> cells</span>";
        document.getElementById("resultsConcentration").innerHTML += "<span class='results-unit'> cells / mL</span>";
        let methods = ["Image", "Outlines", "Circles"];
        setImage(imagesIndex, method, type);
        for (let i = 0; i < 3; i++) {
            let anchor = document.createElement("a");
            anchor.setAttribute("onclick", `displayImage(${imagesIndex}, '${methods[i]}')`);
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
    if (imagesIndex) {
        let anchor = document.getElementById("lastImage");
        anchor.setAttribute("onclick", "changeImage(-1)");
        anchor.setAttribute("class", "last");
    }
    document.getElementById("image-number").innerHTML = (imagesIndex + 1).toString() + " of " + imagesList.length.toString();
    if (imagesIndex < imagesList.length - 1) {
        let anchor = document.getElementById("nextImage");
        anchor.setAttribute("onclick", "changeImage(1)");
        anchor.setAttribute("class", "next");
    }    
}

const setGraph = (selectedGraph, metric) => {
    let graphList = Object.keys(parsedResponse["graphs"][metric]);
    document.getElementById(metric.toLowerCase()+"s-graph").src = "data:image/jpeg;base64,"+parsedResponse["graphs"][metric][selectedGraph];
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
        displayImage(0, "Outlines");
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

const addInformation = (csv_rows) => {
    let table = document.getElementById("csv-table");
    let thead = document.createElement("thead");
    for (let header of csv_rows["header"]) {
        let td = document.createElement("td");
        td.innerHTML = header;
        thead.appendChild(td)
    }
    table.appendChild(thead);
    for (let object of [[parsedResponse["file_counts"], "file"], [parsedResponse["url_counts"], "url"]]) {
        for (let key of Object.keys(object[0])) {
            imagesList.push([object[0][key], key, object[1]]);
            let linkIndex = imagesList.length - 1;
            let tr = document.createElement("tr");
            tr.setAttribute("onclick", "displayImage("+linkIndex+", 'Outlines')");
            tr.setAttribute("title", key);
            for (let index in csv_rows[key]) {
                let td = document.createElement("td");
                td.setAttribute("class", "data-"+index.toString());
                td.innerHTML = csv_rows[key][index];
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
        parseInt(value);
    } catch (error) {
        return false;
    }
    return true;
}

const checkSensitives = () => {
    if (document.getElementById("time-sensitive").checked) {
        if (!document.getElementById("time-unit").value) {
            alert("Please enter a valid unit for time sensitive data");
            return false;   
        }
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.id == "sample-preview") {
                continue;
            }
            if (checkIfNumber(previewElement.children[1].children[0].children[0].value)) {
                fileInput.append("time-"+previewElement.children[0].children[0].title, previewElement.children[1].children[0].children[0].value);
            }
        }
        fileInput.append("time-unit", document.getElementById("time-unit").value);
    }
    if (!document.getElementById("depth-sensitive").checked) {
        for (let previewElement of document.getElementById("previews").children) {
            if (previewElement.id == "sample-preview") {
                continue;
            }
            if (checkIfNumber(previewElement.children[1].children[1].children[0].value)) {
                fileInput.append("depth-"+previewElement.children[0].children[0].title, previewElement.children[1].children[1].children[0].value);
            }
            else {
                if (!checkIfNumber(document.getElementById("all-depth").value)) {
                    alert("Please enter a depth for each picture or a valid default depth");
                    return false;
                } 
                else {
                    fileInput.append("depth-"+previewElement.children[0].children[0].title, document.getElementById("all-depth").value);
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
    document.getElementById("analyze-link").setAttribute("class", "analyze-button");
    document.getElementById("loader-wrapper").setAttribute("class", "hidden");
}

const loadInformation = () => {
    if (!document.getElementById("previews").children.length) {
        alert("Please add a file or url");
        return;
    }
    let request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (request.readyState == 4 && document.getElementById("loader-wrapper").getAttribute("class") != "hidden") {
            imagesIndex = 0;
            imagesList = [];
            parsedResponse = JSON.parse(request.response);
            addInformation(parsedResponse["csv"]);
            document.getElementById("loader-wrapper").setAttribute("class", "hidden");
            setOverview();
        }    
    }
    if (!checkSensitives()) {
        return;
    }
    fileInput.append("url", JSON.stringify(urlToSubmit));
    request.open("POST", "/");
    request.send(fileInput);
    document.getElementById("analyze-link").setAttribute("class", "hidden");
    document.getElementById("loader-wrapper").removeAttribute("class");
}
document.addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        if (event.ctrlKey) {
            loadInformation();
        } else {
            addURL();
        }
    }
});
document.getElementById("staging").addEventListener("change", addFileToList);
document.getElementById("time-sensitive").addEventListener("change", changeTime);
document.getElementById("depth-sensitive").addEventListener("change", changeDepth);
document.getElementById("time-unit").addEventListener("keyup", updateTimePreviews);