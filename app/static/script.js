let imageIndex = 0;
let fileList = document.getElementById("fileList");
let urlList = document.getElementById("urlList");
let fileInput = new FormData();
let urlToSubmit = [];
const addURL = () => {
    if (!document.getElementById("url").value){
        alert("Please enter a url");
    }
    else {
        urlToSubmit.push(document.getElementById("url").value);
        let listItem = document.createElement("li");
        listItem.innerHTML = document.getElementById("url").value;
        urlList.appendChild(listItem);
        document.getElementById("url").value = "";
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

const addCountsToOutput = (responseObject, imageType) => {
    if ($.isEmptyObject(responseObject)) {
        return;
    }
    let responseList = document.getElementById("responseList");
    for (let key of Object.keys(responseObject)) {
        let imageCount = document.createElement("li");
        let hiddenName = document.createElement("p");
        hiddenName.setAttribute("style", "display: none;");
        hiddenName.innerHTML = key;
        imageCount.appendChild(hiddenName);
        let countText = document.createElement("p");
        countText.innerHTML = responseObject[key];
        imageCount.appendChild(countText);
        if (!responseObject[key].includes("There was an error processing")) {
            for (let displayMethod of ["Image", "Outlines", "Circles"]) {
                let getImageButton = document.createElement("button");
                getImageButton.setAttribute("type", "button");
                getImageButton.setAttribute("onclick", `showPicture(this, '${displayMethod}', '${imageType}')`);
                let imageButtonText = document.createElement("p");
                imageButtonText.innerHTML = "Show " + displayMethod;
                getImageButton.appendChild(imageButtonText);
                imageCount.appendChild(getImageButton);
            }
        }
        responseList.appendChild(imageCount);
    }
}
const sendFiles = () => {
    imageIndex = 0;
    let request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (request.readyState == 4) {
            console.log(request.response)
            document.getElementById("responseList").innerHTML = "";
            let parsedResponse = JSON.parse(request.response);
            console.log(parsedResponse);
            addCountsToOutput(parsedResponse["file_counts"], "file");
            addCountsToOutput(parsedResponse["url_counts"], "url");
        }
    }
    console.log(urlToSubmit);
    fileInput.append("url", JSON.stringify(urlToSubmit));
    console.log(fileInput.get("url"));
    request.open("POST", "/");
    request.send(fileInput);
}
const showPicture = (element, method, type) => {
    console.log(method);
    console.log(type);
    let imageForm = new FormData();
    if (type=="url") {
        imageForm.append(type, element.parentNode.childNodes[0].innerHTML);
    } else {
        imageForm.append(type, fileInput.get(element.parentNode.childNodes[0].innerHTML));
    }
    imageForm.append("display", method.toLowerCase());
    let request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (request.readyState == 4) {
            image = document.createElement("img");
            image.setAttribute("src", "data:image/jpeg;base64,"+request.response);
            element.appendChild(image);
            let downloadLink = document.createElement("a");
            downloadLink.setAttribute("href", "data:image/jpeg;base64,"+request.response);
            let filename = "";
            if (type=="url") {
                filename = imageForm.get("url").slice(imageForm.get("url").lastIndexOf("/")+1);
            } else {
                filename = element.parentNode.childNodes[0].innerHTML;
            }
            let nameToSave = filename.slice(0, filename.lastIndexOf(".")) + method + filename.slice(filename.lastIndexOf("."));
            downloadLink.setAttribute("download", nameToSave);
            downloadLink.innerHTML = "<img src='static/download_icon.svg'>";
            element.appendChild(downloadLink);
            element.childNodes[0].innerHTML = "Hide " + method;
            element.setAttribute("onclick", `removePicture(this, '${method}', '${type}')`);
        }
    }
    request.open("POST", "/");
    request.send(imageForm);
}
const removePicture = (element, method, type) => {
    element.childNodes[1].remove();
    element.childNodes[1].remove();
    element.childNodes[0].innerHTML = "Show " + method;
    element.setAttribute("onclick", `showPicture(this, '${method}', '${type}')`);
}