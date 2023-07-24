const setOverview = () => {
    changePalette("overview");
    document.getElementById("tab-nav").setAttribute("class", "tab-nav");
    document.getElementById("show-overview").setAttribute("class", "selected-tab");
    document.getElementById("show-images").setAttribute("class", "tab");
    document.getElementById("show-table").setAttribute("class", "tab");
    if (parsedResponse["stats"] == "No data available") {
        displayImage(0, "output");
        document.getElementById("tab-nav").setAttribute("class", "hidden");
    } else {
        document.getElementById("counts-stats").innerHTML = "";
        document.getElementById("concentrations-stats").innerHTML = "";
        let units = {"Count": " cells", "Concentration": " cells / mL"};
        let shorthands = {"Mean":"mean", "Range":"range", "Standard Deviation":"stddev"};
        for (let metric of ["Count", "Concentration"]) {
            for (let stat of ["Mean", "Range", "Standard Deviation"]) {
                let statSection = document.getElementById(metric.toLowerCase()+"s-"+shorthands[stat]);
                statSection.innerHTML = parsedResponse["stats"][metric][stat];
            }
            let medianBox = document.getElementById(metric.toLowerCase()+"s-median");
            let iqrBox = document.getElementById(metric.toLowerCase()+"s-iqr");
            iqrList = parsedResponse["stats"][metric]["iqr"]
            medianBox.innerHTML = iqrList[1];
            iqrBox.innerHTML = iqrList[2] + " - " + iqrList[0];
            let graphs = Object.keys(parsedResponse["graphs"]["Count"]);
            setGraph(graphs[0], metric);
        }
    }
}

const setGraph = (selectedGraph, metric) => {
    let graphList = Object.keys(parsedResponse["graphs"][metric]);
    document.getElementById(metric.toLowerCase()+"s-graph").src = parsedResponse["graphs"][metric][selectedGraph];
    let changeGraph = document.getElementById(metric.toLowerCase()+"s-change");
    changeGraph.innerHTML = "<a class='download-photo ghost'></a><a id='"+metric+"s-graph-download' class='download-photo'></a>";
    for (let graphType of graphList) {
        let graphAnchor = document.createElement("a");
        graphAnchor.innerHTML = graphType;
        graphAnchor.setAttribute("onclick", "setGraph('" + graphType + "', '" + metric + "')");
        if (graphType==selectedGraph) {
            graphAnchor.setAttribute("class", "graph-type-clicked");
        } else {
            graphAnchor.setAttribute("class", "graph-type");
        }
        changeGraph.appendChild(graphAnchor);
    }
    let download_link = document.getElementById(metric+"s-graph-download");
    download_link.setAttribute("href", parsedResponse["graphs"][metric][selectedGraph]);
    download_link.setAttribute("download", selectedGraph+".jpg");
}