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
            "https://building-image-classifier.onrender.com/api/verify/",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (response.ok) {
            result.innerHTML = `
                <div class="result-status">
                    ${data.matching_status ? "YES" : "NO"}
                </div>
                <p>Image classification completed.</p>
            `;
        } else {
            result.innerHTML = `
                <div class="result-status">Error</div>
                <p>Unable to classify the image.</p>
            `;
        }

    } catch (error) {
        result.innerHTML = `
            <div class="result-status">Error</div>
            <p>Unable to connect to the classification server.</p>
        `;
    }
});