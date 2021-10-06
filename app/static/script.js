let imageIndex = 0;
let imagesList = [];
let fileList = document.getElementById("fileList");
let urlList = document.getElementById("urlList");
let fileInput = new FormData();
let urlToSubmit = [];
let parsedResponse = {};


const addURL = () => {
    if (!document.getElementById("urlInput").value){
        alert("Please enter a url");
    }
    else {
        urlToSubmit.push(document.getElementById("urlInput").value);
        let previewImage = document.createElement("img");
        previewImage.setAttribute("class", "preview");
        previewImage.src = document.getElementById("urlInput").value;
        document.getElementById("urlInput").value = "";
        document.getElementById("previews").appendChild(previewImage);
    }
}
const addFileToList = () => {
    const stagingInput = document.getElementById("staging");
    if (stagingInput.files.length===0) {
        alert("No files selected");
    }
    else {
        for (let newFile of stagingInput.files) {
            fileInput.append(newFile.name, newFile);
            /*let previewFigure = document.createElement("figure");
            previewFigure.setAttribute("class", "preview");*/
            let previewImage = document.createElement("img");
            previewImage.setAttribute("class", "preview");
            previewImage.src = URL.createObjectURL(newFile);
            /*previewFigure.appendChild(previewImage);
            let previewFigcaption = document.createElement("figcaption");
            previewFigcaption.innerHTML = "Delete " + newFile.name;
            previewFigcaption.setAttribute("onclick", "deleteFile()");
            previewFigure.appendChild(previewFigcaption);*/
            document.getElementById("previews").appendChild(previewImage);
        }
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
    let allPalettes = ["inputInfo", "overview", "resultsInfo"];
    allPalettes.splice(allPalettes.indexOf(elementId), 1)
    for (let deleteId of allPalettes) {
        document.getElementById(deleteId).setAttribute("class", "hidden");
    }
    document.getElementById(elementId).setAttribute("class", "palette");
}

const changeImage = (change) => {
    displayImage(imageIndex + change, "Image");
}

const setImage = (index, method) => {
    let resultsImage = document.getElementById("resultsImage");
    resultsImage.setAttribute("src", "data:image/jpeg;base64,"+imagesList[index][0][method.toLowerCase()])
}

const displayImage = (imagesIndex, method)  => {
    imageIndex = imagesIndex;
    changePalette("resultsInfo");
    //let downloadLink = document.getElementById("resultsDownload");
    //downloadLink.innerHTML = "";
    let count = imagesList[imagesIndex][0]["count"];
    let location = imagesList[imagesIndex][1];
    let type = imagesList[imagesIndex][2];
    let resultsElement = document.getElementById("visual");
    let i = 0;
    while (i < resultsElement.childNodes.length) {
        if (resultsElement.childNodes[i].nodeName.toLowerCase() == "a") {
            console.log(resultsElement.childNodes[i].getAttribute("id"));
            resultsElement.childNodes[i].remove();
        }
        else {
            i++;
        }
    }
    document.getElementById("img-nav").setAttribute("class", "hidden");
    document.getElementById("lastImage").setAttribute("class", "hidden");
    document.getElementById("nextImage").setAttribute("class", "hidden");
    document.getElementById("show-pictures").innerHTML = "";
    document.getElementById("image-number").innerHTML = "";
    //document.getElementById("resultsTitle").innerHTML = location;
    document.getElementById("resultsCount").innerHTML = count;
    if (count.includes("There was an error processing")) {
        document.getElementById("resultsImage").removeAttribute("src");
        if (Object.keys(imagesList[imagesIndex][0]).length > 1) {
            setImage(imagesIndex, "Image");
        }
    }
    else {
        let methods = ["Image", "Outlines", "Circles"];
        methods.splice(methods.indexOf(method), 1);
        setImage(imagesIndex, method);
        for (let i = 0; i < 2; i++) {
            let anchor = document.createElement("a");
            anchor.setAttribute("onclick", `displayImage(${imagesIndex}, '${methods[i]}')`);
            anchor.innerHTML = "Show " + methods[i];
            anchor.setAttribute("class", "image-type");
            document.getElementById("show-pictures").appendChild(anchor);
        }
        //downloadLink.innerHTML = "<img src='static/download_icon.svg'>";
        //downloadLink.setAttribute("href", document.getElementById("resultsImage").getAttribute("src"));
        //downloadLink.setAttribute("download", getFilename(location, method, type));
        //resultsElement.appendChild(downloadLink);
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

const addInformation = (responseObject, imageType) => {
    if ($.isEmptyObject(responseObject)) {
        return;
    }
    for (let key of Object.keys(responseObject)) {
        imagesList.push([responseObject[key], key, imageType]);
    }
}

const setGraph = (selectedGraph, metric) => {
    let graphList = Object.keys(parsedResponse["graphs"][metric]);
    graphList.splice(graphList.indexOf(selectedGraph), 1);
    document.getElementById(metric.toLowerCase()+"s-graph").src = "data:image/jpeg;base64,"+parsedResponse["graphs"][metric][selectedGraph];
    let changeGraph = document.getElementById(metric.toLowerCase()+"s-change");
    changeGraph.innerHTML = "";
    for (let otherGraph of graphList) {
        let graphAnchor = document.createElement("a");
        graphAnchor.innerHTML = "Show " + otherGraph;
        graphAnchor.setAttribute("onclick", "setGraph('" + otherGraph + "', '" + metric + "')");
        console.log("setGraph('" + otherGraph + "', '" + metric + "')");
        graphAnchor.setAttribute("class", "graph-button");
        changeGraph.appendChild(graphAnchor);
    }
}

const setOverview = () => {
    changePalette("overview");
    if (parsedResponse["stats"] == "No data available") {
        document.getElementById("counts-stats").innerHTML = "No data available";
        document.getElementById("concentrations-stats").innerHTML = "No data available";
        document.getElementById("counts-graph").src = "static/graph_not_available.jpg";
        document.getElementById("concentrations-graph").src = "static/graph_not_available.jpg";
    } else {
        document.getElementById("counts-stats").innerHTML = "";
        document.getElementById("concentrations-stats").innerHTML = "";
        let units = {"Count": " cells", "Concentration": " cells/mL (millions)"};
        for (let metric of ["Count", "Concentration"]) {
            let firstDiv = document.createElement("div");
            firstDiv.setAttribute("class", "stats-section");
            for (let stat of ["Mean", "Range", "Standard Deviation"]) {
                let statBox = document.createElement("div");
                statBox.setAttribute("class", "stat-box");
                statBox.innerHTML = stat + ": " + parsedResponse["stats"][metric][stat] + units[metric];
                firstDiv.appendChild(statBox);
            }
            document.getElementById(metric.toLowerCase()+"s-stats").appendChild(firstDiv);
            let secondDiv = document.createElement("div");
            secondDiv.setAttribute("class", "stats-section");
            let medianBox = document.createElement("div");
            medianBox.setAttribute("class", "stat-box");
            let iqrBox = document.createElement("div");
            iqrBox.setAttribute("class", "stat-box");
            iqrList = parsedResponse["stats"][metric]["iqr"]
            medianBox.innerHTML = "Median: " + iqrList[1] + units[metric];
            iqrBox.innerHTML = "Interquartile Range: " + iqrList[2] + " - " + iqrList[0] + "(" + (parseInt(iqrList[2]) - parseInt(iqrList[0])).toString() + ")" + units[metric];
            secondDiv.appendChild(medianBox);
            secondDiv.appendChild(iqrBox);
            document.getElementById(metric.toLowerCase()+"s-stats").appendChild(secondDiv);
            let graphs = Object.keys(parsedResponse["graphs"]["Count"]);
            setGraph(graphs[0], metric);
        }
    }
}

const loadInformation = () => {
    if (!document.getElementById("previews").children.length) {
        alert("Please add a file or url");
        return;
    }
    let request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (request.readyState == 4) {
            imagesIndex = 0;
            imagesList = [];
            parsedResponse = JSON.parse(request.response);
            console.log(parsedResponse["stats"])
            addInformation(parsedResponse["file_counts"], "file");
            addInformation(parsedResponse["url_counts"], "url");
            setOverview();
        }    
    }
    fileInput.set("url", JSON.stringify(urlToSubmit));
    request.open("POST", "/");
    request.send(fileInput);
    document.getElementById("resultsImage").setAttribute("src", "static/person_counting.jpg");
    //document.getElementById("resultsTitle").innerHTML = "Counting Cells";
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