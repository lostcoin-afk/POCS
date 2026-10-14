const captureButton = document.getElementById("capture-btn");

const sequenceInput = document.getElementById("expected-sequence");
const productCodeInput = document.getElementById("expected-product-code");

const resultBanner = document.getElementById("result-banner");
const resultType = document.getElementById("result-type");
const resultOrientation = document.getElementById("result-orientation");
const resultDetected = document.getElementById("result-detected");
const resultExpected = document.getElementById("result-expected");
const resultReason = document.getElementById("result-reason");

const captureImageWrap = document.getElementById("capture-image-wrap");
const captureImage = document.getElementById("capture-image");

function setResultNeutral(message) {
    resultBanner.className = "result neutral";
    resultBanner.textContent = message;
}

function setResult(result) {
    const correct = Boolean(result.correct);

    resultBanner.className = correct
        ? "result correct"
        : "result incorrect";

    resultBanner.textContent = correct
        ? "CORRECT"
        : "DEFECTIVE / INCORRECT";

    resultType.textContent = result.type || "-";
    resultOrientation.textContent = result.orientation || "-";
    resultDetected.textContent = result.detected_text || "-";

    if (result.validation) {
        resultExpected.textContent =
            result.validation.expected || "-";
    } else {
        resultExpected.textContent = "-";
    }

    resultReason.textContent = result.reason || "-";

    if (result.capture_url) {
        captureImage.src =
            `${result.capture_url}?t=${Date.now()}`;
        captureImageWrap.classList.remove("hidden");
    }
}

captureButton.addEventListener("click", async () => {
    captureButton.disabled = true;
    setResultNeutral("Capturing and running OCR...");

    try {
        const params = new URLSearchParams({
            expected_sequence: sequenceInput.value.trim(),
            expected_product_code: productCodeInput.value.trim()
        });

        const response = await fetch(`/capture?${params.toString()}`, {
            method: "POST"
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Capture request failed."
            );
        }

        setResult(data);
    } catch (error) {
        resultBanner.className = "result incorrect";
        resultBanner.textContent = "ERROR";
        resultReason.textContent = error.message;
    } finally {
        captureButton.disabled = false;
    }
});
