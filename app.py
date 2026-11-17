import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ImageLab | Image Processing",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #101827 100%);
        color: #f3f4f6;
    }
    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(148,163,184,.18);
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(30,41,59,.95), rgba(17,24,39,.92));
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.15rem; color: #f8fafc; }
    .hero p { color: #aebbd0; margin: .45rem 0 0 0; }
    .section-title { font-size: 1.25rem; font-weight: 700; margin: .4rem 0 1rem; }
    .small-note { color: #9ca3af; font-size: .9rem; }
    div[data-testid="stMetric"] {
        background: rgba(30,41,59,.65);
        border: 1px solid rgba(148,163,184,.18);
        padding: 12px;
        border-radius: 12px;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Image processing functions
# ---------------------------------------------------------
def to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)

def custom_grayscale(image):
    gray = (
        0.299 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.114 * image[:, :, 2]
    )
    return to_uint8(gray)

def adjust_brightness(image, operation="add", value=40):
    image_float = image.astype(np.float32)
    if operation == "Add":
        result = image_float + value
    elif operation == "Subtract":
        result = image_float - value
    elif operation == "Multiply":
        result = image_float * value
    elif operation == "Divide":
        if value == 0:
            raise ValueError("The division value cannot be zero.")
        result = image_float / value
    else:
        raise ValueError("Unsupported arithmetic operation.")
    return to_uint8(result)

def image_complement(image):
    return 255 - image

def solarization(image, threshold=128):
    result = image.copy()
    mask = image > threshold
    result[mask] = 255 - image[mask]
    return result

def eliminate_channel(image, channel="B"):
    result = image.copy()
    channel_index = {"R": 0, "G": 1, "B": 2}[channel]
    result[:, :, channel_index] = 0
    return result

def calculate_histogram(gray_image):
    histogram = np.zeros(256, dtype=np.int64)
    for pixel in gray_image.flatten():
        histogram[pixel] += 1
    return histogram

def histogram_stretching(gray_image):
    min_value = np.min(gray_image)
    max_value = np.max(gray_image)
    if max_value == min_value:
        return gray_image.copy()
    stretched = (gray_image.astype(np.float32) - min_value) * 255 / (max_value - min_value)
    return to_uint8(stretched)

def histogram_equalization(gray_image):
    return cv2.equalizeHist(gray_image)

def change_brightness_gray(gray_image, value=0):
    return to_uint8(gray_image.astype(np.float32) + value)

def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    return to_uint8(image.astype(np.float32) + noise)

def add_salt_pepper_noise(image, amount=0.04):
    noisy = image.copy()
    height, width = image.shape[:2]
    number = int(amount * height * width)

    y_coords = np.random.randint(0, height, number)
    x_coords = np.random.randint(0, width, number)
    noisy[y_coords, x_coords] = 255

    y_coords = np.random.randint(0, height, number)
    x_coords = np.random.randint(0, width, number)
    noisy[y_coords, x_coords] = 0
    return noisy

def apply_smoothing(noisy_image, noise_type="Salt-and-Pepper"):
    if noise_type == "Salt-and-Pepper":
        return cv2.medianBlur(noisy_image, 5)
    return cv2.GaussianBlur(noisy_image, (5, 5), 0)

def calculate_snr(original, processed):
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    noise = original - processed
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)

def blur_image(image):
    return cv2.GaussianBlur(image, (7, 7), 0)

def sharpen_image(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, kernel)

def detect_edges(image, method="Canny"):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "Canny":
        return cv2.Canny(gray, 100, 200)
    if method == "Sobel":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return to_uint8(magnitude)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return to_uint8(np.abs(laplacian))

# ---------------------------------------------------------
# Display and download helpers
# ---------------------------------------------------------
def show_pair(original, processed, original_title="Original Image", processed_title="Processed Image",
              original_gray=False, processed_gray=False):
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption=original_title, use_container_width=True,
                 clamp=True, channels="GRAY" if original_gray else "RGB")
    with col2:
        st.image(processed, caption=processed_title, use_container_width=True,
                 clamp=True, channels="GRAY" if processed_gray else "RGB")

def show_histogram(gray_image, title):
    histogram = calculate_histogram(gray_image)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(histogram, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Number of Pixels")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def download_button(image, filename="processed_image.png"):
    if image.ndim == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    st.download_button(
        "⬇️ Download Result",
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
        use_container_width=True
    )

def get_uploaded_image():
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="main_image"
    )
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🖼️ ImageLab")
    st.caption("Image Processing Lab — Tasks 1 to 4")
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Task 1 · RGB & Grayscale",
            "Task 2 · Image Operations",
            "Task 3 · Histogram Enhancement",
            "Task 4 · Filters & Edges"
        ]
    )
    st.markdown("---")
    st.caption("Python • OpenCV • NumPy • Streamlit")

# ---------------------------------------------------------
# Main header
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Image Processing Studio</h1>
    <p>A unified workspace for RGB analysis, image enhancement, noise reduction, sharpening and edge detection.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.markdown("### Welcome to ImageLab")
    st.write("Select a task from the sidebar to start processing an image.")
    st.markdown("#### Lab modules")
    cards = st.columns(4)
    modules = [
        ("01", "RGB & Grayscale", "Channel separation and custom grayscale conversion."),
        ("02", "Image Operations", "Arithmetic operations, complement, solarization and channel elimination."),
        ("03", "Histogram", "Histogram calculation, stretching and equalization."),
        ("04", "Filters & Edges", "Noise reduction, SNR, sharpening and edge detection.")
    ]
    for col, (number, title, description) in zip(cards, modules):
        with col:
            st.markdown(f"**{number} — {title}**")
            st.caption(description)
    st.info("Upload an image inside any task page. Each task has its own processing controls and result download.")

elif page == "Task 1 · RGB & Grayscale":
    st.markdown("### Task 1 — RGB Channel Separation & Custom Grayscale")
    image = get_uploaded_image()
    if image is not None:
        st.markdown("#### RGB Channel Separation")
        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        cols = st.columns(3)
        for col, channel, title in zip(cols, [red, green, blue], ["Red Channel", "Green Channel", "Blue Channel"]):
            with col:
                st.image(channel, caption=title, use_container_width=True, clamp=True)
        st.markdown("#### Custom Grayscale Conversion")
        gray = custom_grayscale(image)
        show_pair(image, gray, "Original Image", "Custom Grayscale", False, True)
        st.caption("Formula: Gray = 0.299R + 0.587G + 0.114B")
        download_button(gray, "task1_custom_grayscale.png")

elif page == "Task 2 · Image Operations":
    st.markdown("### Task 2 — Arithmetic, Complement, Solarization & Channel Elimination")
    image = get_uploaded_image()
    if image is not None:
        operation_tab, effect_tab, channel_tab = st.tabs(
            ["Arithmetic Operation", "Complement / Solarization", "Channel Elimination"]
        )

        with operation_tab:
            st.markdown("#### Brightness Correction")
            operation = st.selectbox("Arithmetic operation", ["Add", "Subtract", "Multiply", "Divide"])
            if operation in ["Add", "Subtract"]:
                value = st.slider("Value", 1, 100, 40)
            elif operation == "Multiply":
                value = st.slider("Factor", 1, 3, 1)
            else:
                value = st.slider("Divisor", 1, 5, 2)
            result = adjust_brightness(image, operation, value)
            show_pair(image, result, "Original Image", f"After {operation} ({value})")
            st.caption("Choose Add for a dark image or Subtract for an overly bright image.")
            download_button(result, "task2_arithmetic_result.png")

        with effect_tab:
            effect = st.radio("Select effect", ["Complement", "Solarization"], horizontal=True)
            if effect == "Complement":
                result = image_complement(image)
            else:
                threshold = st.slider("Solarization threshold", 0, 255, 128)
                result = solarization(image, threshold)
            show_pair(image, result, "Original Image", effect)
            download_button(result, "task2_effect_result.png")

        with channel_tab:
            channel = st.selectbox("Channel to eliminate", ["R", "G", "B"])
            result = eliminate_channel(image, channel)
            show_pair(image, result, "Original Image", f"After Removing {channel} Channel")
            st.caption("Select the channel that causes the color imbalance in the chosen image.")
            download_button(result, "task2_channel_elimination.png")

elif page == "Task 3 · Histogram Enhancement":
    st.markdown("### Task 3 — Histogram Enhancement")
    image = get_uploaded_image()
    if image is not None:
        gray = custom_grayscale(image)
        st.markdown("#### Original Grayscale Image & Histogram")
        left, right = st.columns([1, 1.5])
        with left:
            st.image(gray, caption="Original Grayscale Image", use_container_width=True, clamp=True)
        with right:
            show_histogram(gray, "Histogram Before Enhancement")

        method = st.radio("Enhancement method", ["Histogram Stretching", "Histogram Equalization"], horizontal=True)
        brightness = st.slider("Brightness correction", -100, 100, 0)
        if method == "Histogram Stretching":
            enhanced = histogram_stretching(gray)
        else:
            enhanced = histogram_equalization(gray)
        final_result = change_brightness_gray(enhanced, brightness)

        st.markdown("#### Enhanced Result")
        show_pair(gray, final_result, "Before Enhancement", "After Enhancement", True, True)
        st.markdown("#### Histogram After Enhancement")
        show_histogram(final_result, "Histogram After Enhancement")
        download_button(final_result, "task3_histogram_enhancement.png")

elif page == "Task 4 · Filters & Edges":
    st.markdown("### Task 4 — Noise Reduction, Detail Enhancement & Edge Detection")
    image = get_uploaded_image()
    if image is not None:
        noise_tab, sharp_tab, edge_tab = st.tabs(
            ["Noise Reduction & SNR", "Detail Enhancement", "Edge Detection"]
        )

        with noise_tab:
            st.markdown("#### Add Noise and Apply a Smoothing Filter")
            noise_type = st.selectbox("Noise type", ["Salt-and-Pepper", "Gaussian"])
            if noise_type == "Salt-and-Pepper":
                amount = st.slider("Noise amount", 0.01, 0.15, 0.04, 0.01)
                noisy = add_salt_pepper_noise(image, amount)
            else:
                sigma = st.slider("Gaussian noise standard deviation", 5, 60, 25)
                noisy = add_gaussian_noise(image, sigma=sigma)
            denoised = apply_smoothing(noisy, noise_type)

            st.markdown("#### Noisy Image")
            st.image(noisy, caption=f"Image with {noise_type} Noise", use_container_width=True)
            st.markdown("#### Smoothed Result")
            show_pair(noisy, denoised, "Noisy Image", "After Smoothing")
            snr_before = calculate_snr(image, noisy)
            snr_after = calculate_snr(image, denoised)
            m1, m2 = st.columns(2)
            m1.metric("SNR Before Smoothing", f"{snr_before:.2f} dB")
            m2.metric("SNR After Smoothing", f"{snr_after:.2f} dB")
            st.caption("SNR is calculated from scratch using the original clean image as the reference.")
            download_button(denoised, "task4_denoised_image.png")

        with sharp_tab:
            st.markdown("#### Blur and Sharpen")
            blurred = blur_image(image)
            sharpened = sharpen_image(blurred)
            col1, col2, col3 = st.columns(3)
            col1.image(image, caption="Original Image", use_container_width=True)
            col2.image(blurred, caption="Gaussian Blurred", use_container_width=True)
            col3.image(sharpened, caption="Sharpened Result", use_container_width=True)
            st.caption("A custom 3×3 sharpening kernel is applied to the blurred image.")
            download_button(sharpened, "task4_sharpened_image.png")

        with edge_tab:
            st.markdown("#### Edge Detection")
            method = st.selectbox("Edge detection operator", ["Canny", "Sobel", "Laplacian"])
            edges = detect_edges(image, method)
            show_pair(image, edges, "Original Image", f"{method} Edges", False, True)
            download_button(edges, "task4_edge_detection.png")

st.markdown("---")
st.caption("ImageLab · Image Processing Tasks 1–4")
