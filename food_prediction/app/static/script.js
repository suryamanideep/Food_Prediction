// Preview selected image
document.getElementById("image").addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;

    const preview = document.getElementById("preview");
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
});

// Upload + predict
document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("image");
    if (fileInput.files.length === 0) {
        alert("Please choose an image!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const resultBox = document.getElementById("result");
    resultBox.textContent = "Predicting...";

    const res = await fetch("/predict", {
        method: "POST",
        body: formData,
    });

    const data = await res.json();

    if (data.error) {
        resultBox.textContent = "Error: " + data.error;
        return;
    }

    resultBox.innerHTML = `
        <p><strong>Prediction:</strong> ${data.label}</p>
        <p><strong>Calories/100g:</strong> ${data.calories}</p>
        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
        <br>
        <strong>Top 3 Predictions:</strong>
        <ul>
            ${data.predictions.map(p => `<li>${p.label} - ${(p.confidence*100).toFixed(1)}%</li>`).join("")}
        </ul>
    `;
});
function showPreview(event) {
    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(event.target.files[0]);
    img.style.display = "block";
}
