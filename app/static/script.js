let imageIndex = 0;
let imagesList = [];
let fileList = document.getElementById("fileList");
let urlList = document.getElementById("urlList");
let fileInput = new FormData();
let urlToSubmit = [];


const addURL = () => {
    if (!document.getElementById("urlInput").value){
        alert("Please enter a url");
    }
    else {
        urlToSubmit.push(document.getElementById("urlInput").value);
        let listItem = document.createElement("li");
        listItem.innerHTML = document.getElementById("urlInput").value;
        urlList.appendChild(listItem);
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
            fileInput.append(newFile.name, newFile);
            let listItem = document.createElement("li");
            listItem.innerHTML = newFile.name;
            fileList.appendChild(listItem);
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

const changeImage = (change) => {
    imageIndex += change;
    displayImage(imageIndex, "Image");
}

const setImage = (index, method) => {
    let resultsImage = document.getElementById("resultsImage");
    resultsImage.setAttribute("src", "data:image/jpeg;base64,"+imagesList[index][0][method.toLowerCase()])
}

const displayImage = (imagesIndex, method)  => {
    let downloadLink = document.getElementById("resultsDownload");
    downloadLink.innerHTML = "";
    let count = imagesList[imagesIndex][0]["count"];
    let location = imagesList[imagesIndex][1];
    let type = imagesList[imagesIndex][2];
    let resultsElement = document.getElementById("results");
    let i = 0;
    while (i < resultsElement.childNodes.length) {
        if (resultsElement.childNodes[i].nodeName.toLowerCase() == "button") {
            resultsElement.childNodes[i].remove();
        }
        else {
            i++;
        }
    }
    document.getElementById("resultsTitle").innerHTML = location;
    document.getElementById("resultsCount").innerHTML = count;
    if (count.includes("There was an error processing")) {
        document.getElementById("resultsImage").removeAttribute("src");
        if (imagesList[imagesIndex][0].keys().length > 1) {
            setImage(imagesIndex, "Image");
        }
    }
    else {
        let methods = ["Image", "Outlines", "Circles"];
        methods.splice(methods.indexOf(method), 1);
        setImage(imagesIndex, method);
        for (let i = 0; i < 2; i++) {
            let button = document.createElement("button");
            button.setAttribute("onclick", `displayImage(${imagesIndex}, '${methods[i]}')`);
            button.innerHTML = "Show " + methods[i];
            resultsElement.appendChild(button);
        }
        downloadLink.innerHTML = "<img src='static/download_icon.svg'>";
        downloadLink.setAttribute("href", document.getElementById("resultsImage").getAttribute("src"));
        downloadLink.setAttribute("download", getFilename(location, method, type));
        resultsElement.appendChild(downloadLink);
    }
    if (imagesIndex) {
        let button = document.createElement("button");
        button.setAttribute("onclick", "changeImage(-1)");
        let arrow = document.createElement("img");
        arrow.setAttribute("src", "static/left_arrow_icon.svg");
        arrow.setAttribute("height", 17)
        button.appendChild(arrow);
        resultsElement.appendChild(button); 
    }
    if (imagesIndex < imagesList.length - 1) {
        let button = document.createElement("button");
        button.setAttribute("onclick", "changeImage(1)");
        let arrow = document.createElement("img");
        arrow.setAttribute("src", "static/right_arrow_icon.svg");
        arrow.setAttribute("height", 17)
        button.appendChild(arrow);
        resultsElement.appendChild(button);
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

const loadInformation = () => {
    if (!urlList.children.length && !fileList.children.length) {
        alert("Please add a file or url");
        return;
    }
    let request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (request.readyState == 4) {
            imagesIndex = 0;
            imagesList = [];
            let parsedResponse = JSON.parse(request.response);
            addInformation(parsedResponse["file_counts"], "file");
            addInformation(parsedResponse["url_counts"], "url");
            displayImage(0, "Image");
        }    
    }
    fileInput.append("url", JSON.stringify(urlToSubmit));
    request.open("POST", "/");
    request.send(fileInput);
    let i = 0;
    let resultsElement = document.getElementById("results");
    while (i < resultsElement.childNodes.length) {
        let nameOfNode = resultsElement.childNodes[i].nodeName.toLowerCase();
        if (nameOfNode == "button" || nameOfNode == "text") {
            resultsElement.childNodes[i].remove();
        }
        else {
            resultsElement.childNodes[i].innerHTML = "";
            if (nameOfNode=="img") {
                resultsElement.childNodes[i].src = "static/person_counting.jpg";
            }
            i++;
        }
    }
    document.getElementById("resultsTitle").innerHTML = "Counting Cells";
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