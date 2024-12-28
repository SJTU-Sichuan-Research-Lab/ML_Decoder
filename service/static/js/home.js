const apiBase = "localhost:8889" || "172.20.1.240:8888"

const previewId = (filename) => {
    return `preview-${filename.split('.')[0]}`
}

document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");
    const imagePreviews = document.getElementById("imagePreviews");
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImg");
    const caption = document.getElementById("caption");
    const modalLabels = document.getElementById("modalLabels");
    const uploadButton = document.getElementById("uploadButton");
    const closeBtn = document.querySelector(".close");

    // Buttons for navigating between images
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");

    // Array to hold image preview elements
    let previewElements = [];
    let currentIndex = 0; // Track the current image index

    imageInput.addEventListener("change", function (event) {
        const files = Array.from(event.target.files);
        imagePreviews.innerHTML = "";  // Clear previous previews

        // Create promises for each image load
        const imagePromises = files.map((file, index) => {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();

                reader.onload = function (e) {
                    const div = document.createElement('div');
                    div.classList.add('preview');
                    div.id = previewId(file.name) // use pure file name as the preview element id

                    // Create an image element
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.alt = file.name;
                    img.className = "preview";

                    // Create a div for the filename
                    const fileNameDiv = document.createElement('div');
                    fileNameDiv.classList.add('file-name');
                    fileNameDiv.textContent = file.name;

                    // Create a div for the label
                    const labelDiv = document.createElement('div')
                    labelDiv.classList.add('scene-label');
                    labelDiv.textContent = ""

                    // Add click event to show the modal with larger image
                    img.addEventListener("click", function () {
                        currentIndex = index; // Update the current image index
                        showImageInModal(currentIndex);
                    });

                    // Append the image and filename to the preview div
                    div.appendChild(img);
                    div.appendChild(fileNameDiv);
                    div.appendChild(labelDiv);

                    // Push the preview element into the array
                    previewElements.push({div, img, labelDiv, fileName: file.name});

                    // Resolve the promise when image is loaded
                    resolve();
                };

                reader.onerror = function (error) {
                    reject(error);
                };

                reader.readAsDataURL(file);  // Convert image to base64 for preview
            });
        });

        // Wait for all images to load, then insert into DOM
        Promise.all(imagePromises)
            .then(() => {
                // Clear the preview container and re-add sorted elements
                imagePreviews.innerHTML = "";
                previewElements.sort((a, b) => a.fileName.toLowerCase().localeCompare(b.fileName.toLowerCase()))
                previewElements.forEach(element => {
                    imagePreviews.appendChild(element.div);
                });
            })
            .catch(error => {
                console.error("Error loading images:", error);
            });
    });

    // Show the image in the modal
    function showImageInModal(index) {
        const selectedImage = previewElements[index];
        console.log(selectedImage)

        // Set the modal image and caption
        modalImg.src = selectedImage.img.src;
        caption.textContent = selectedImage.fileName;
        modalLabels.innerHTML = ""
        if (selectedImage.predictResults) {
            selectedImage.predictResults.forEach(p => {
                const modalLabelElement = document.createElement('div')
                modalLabelElement.className = 'modal-label-element'
                modalLabelElement.textContent = `${p.labelEnglish} ${p.labelChinese} ${p.confidence.toFixed(2)}`
                modalLabels.appendChild(modalLabelElement)
            })
        }
        // Show the modal
        modal.style.display = "block";
    }

    // Close the modal
    closeBtn.addEventListener("click", function () {
        modal.style.display = "none";
    });

    // Navigate to the previous image
    prevBtn.addEventListener("click", function () {
        if (currentIndex > 0) {
            currentIndex--; // Decrement the index
        } else {
            currentIndex = previewElements.length - 1; // Go to the last image
        }
        showImageInModal(currentIndex);
    });

    // Navigate to the next image
    nextBtn.addEventListener("click", function () {
        if (currentIndex < previewElements.length - 1) {
            currentIndex++; // Increment the index
        } else {
            currentIndex = 0; // Go to the first image
        }
        showImageInModal(currentIndex);
    });

    // When user clicks the upload button, upload each image individually
    uploadButton.addEventListener("click", function () {
        console.log("upload")
        const files = imageInput.files;
        if (files.length === 0) {
            alert("Please select images to upload.");
            return;
        }

        // Disable the button and show loading state
        uploadButton.textContent = "Uploading...";  // Change text to indicate upload
        uploadButton.classList.add("upload-loading");  // Optionally add a loading spinner
        uploadButton.disabled = true;  // Disable the button to prevent further clicks

        // Loop over each image file and upload them one by one
        const uploadPromises = Array.from(files).map((file, index) => {
            const formData = new FormData();
            formData.append("image", file);  // Append the single image file

            // Make a POST request to upload the file individually
            return fetch(`http://${apiBase}/nas/api/multiclass_labeling`, {
                method: "POST",
                body: formData,
            })
                .then(response => response.json())
                .then(data => {
                    console.log("Image uploaded successfully:", data);
                    const ele = previewElements.find(p => p.fileName === file.name)
                    const topKResults = data['top_k_results']
                    /** example
                     * [
                     *     {
                     *         "class_idx": 232,
                     *         "confidence": 0.20680318772792816,
                     *         "idx": 0,
                     *         "label": "232 mountain - 山"
                     *     },
                     *     {
                     *         "class_idx": 233,
                     *         "confidence": 0.1916370689868927,
                     *         "idx": 1,
                     *         "label": "233 mountain_path - 山间小路"
                     *     },
                     *     {
                     *         "class_idx": 81,
                     *         "confidence": 0.11439887434244156,
                     *         "idx": 2,
                     *         "label": "81 canyon - 峡谷"
                     *     },
                     *     {
                     *         "class_idx": 344,
                     *         "confidence": 0.10803626477718353,
                     *         "idx": 3,
                     *         "label": "344 valley - 山谷"
                     *     },
                     *     {
                     *         "class_idx": 73,
                     *         "confidence": 0.06980881839990616,
                     *         "idx": 4,
                     *         "label": "73 butte - 小山"
                     *     }
                     * ]
                     */
                    ele.predictResults = []
                    ele.labelDiv.textContent = ""
                    for (const prediction of topKResults) {
                        const labelSplit = prediction.label.split('-')
                        const labelClassIdx = labelSplit[0].trim()
                        const labelEnglish = labelSplit[1].trim()
                        const labelChinese = labelSplit[2].trim()
                        const labelElement = document.createElement('div')
                        labelElement.className = 'label-element'
                        labelElement.innerText = `${labelEnglish} ${labelChinese} ${prediction['confidence'].toFixed(2)}`
                        ele.predictResults.push({
                            labelChinese,
                            labelEnglish,
                            confidence: prediction['confidence']
                        })
                        ele.labelDiv.appendChild(labelElement)
                    }
                    // Optionally, handle the response (e.g., show a success message or update the UI)
                })
                .catch(error => {
                    console.error("Error uploading image:", error);
                    alert("An error occurred while uploading your image. Please try again.");
                });
        });
        // Wait for all uploads to complete
        Promise.all(uploadPromises).then(() => {
            console.log("All images uploaded successfully");

            // Re-enable the upload button and reset state
            uploadButton.textContent = "Upload";  // Reset text
            uploadButton.classList.remove("upload-loading");  // Remove loading spinner (if used)
            uploadButton.disabled = false;  // Re-enable the button
        });
    });
});
