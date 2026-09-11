const imageInput = document.getElementById("imageInput");
const browseButton = document.getElementById("browseButton");
const uploadArea = document.getElementById("uploadArea");
const previewArea = document.getElementById("previewArea");
const checkButton = document.getElementById("checkButton");
const result = document.getElementById("result");

browseButton.addEventListener("click", function () {
    imageInput.click();
});

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    if (file) {
        showImage(file);
    }
});

uploadArea.addEventListener("dragover", function (event) {
    event.preventDefault();
    uploadArea.classList.add("dragging");
});

uploadArea.addEventListener("dragleave", function () {
    uploadArea.classList.remove("dragging");
});

uploadArea.addEventListener("drop", function (event) {
    event.preventDefault();
    uploadArea.classList.remove("dragging");

    const file = event.dataTransfer.files[0];

    if (file && file.type.startsWith("image/")) {
        showImage(file);
    }
});

function showImage(file) {
    const imageUrl = URL.createObjectURL(file);

    previewArea.innerHTML = `
        <img src="${imageUrl}" alt="Selected image">
    `;

    previewArea.style.display = "block";

    result.innerHTML = `
        <div class="result-status">Ready</div>
        <p>The image is ready to be checked.</p>
    `;
}

checkButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {
        result.innerHTML = `
            <div class="result-status">—</div>
            <p>Please select an image first.</p>
        `;
        return;
    }

    result.innerHTML = `
        <div class="result-status">Checking...</div>
        <p>Please wait while the image is being classified.</p>
    `;

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/api/verify/",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (data.success) {
            result.innerHTML = `
                <div class="result-status">${data.result}</div>
                <p>Image classification completed.</p>
            `;
        } else {
            result.innerHTML = `
                <div class="result-status">Error</div>
                <p>${data.message}</p>
            `;
        }

    } catch (error) {
        result.innerHTML = `
            <div class="result-status">Error</div>
            <p>Unable to connect to the classification server.</p>
        `;
    }
});