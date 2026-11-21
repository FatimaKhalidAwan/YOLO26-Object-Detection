
from pathlib import Path
import tempfile
import os
import io
import logging

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="YOLO26 Smart Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLO26 Object Detection System")

st.markdown("""
An interactive object detection application powered by a custom-trained
**YOLO26 Nano model**. Upload an image or MP4 video to detect objects
belonging to the 20 classes in the dataset.
""")


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    logger.info("Loading YOLO26 model from: %s", MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    logger.info("YOLO26 model loaded successfully.")

    return model


if not MODEL_PATH.exists():
    st.error(
        "Model checkpoint not found. "
        "Please place best.pt inside the project's models folder."
    )
    st.stop()

try:
    model = load_model()

except Exception:
    logger.exception("Model loading failed")

    st.error(
        "Unable to load the model. "
        "Please check the model checkpoint and installed dependencies."
    )
    st.stop()


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.expander("About the Model", expanded=True):

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "YOLO26n")
    col2.metric("Classes", len(model.names))
    col3.metric("Framework", "Ultralytics")

    st.write("**Supported classes:**")
    st.write(", ".join(model.names.values()))

    st.markdown("""
    **About YOLO26**

    YOLO26 uses an end-to-end, NMS-free object detection approach.

    **How to use the application**

    1. Select Image Detection or Video Detection.
    2. Upload a supported file.
    3. Adjust the confidence threshold in the sidebar.
    4. Run detection and review the results.
    5. Download the annotated output if required.

    **Video detection note:** Detection counts represent individual
    detection instances across processed frames, not unique physical objects.
    """)


# --------------------------------------------------
# DETECTION SETTINGS
# --------------------------------------------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Controls the minimum confidence required for a detection."
)

st.sidebar.info("The model runs on the CPU in this application.")


# --------------------------------------------------
# DETECTION EXTRACTION
# --------------------------------------------------

def extract_detections(result, frame_number=None):
    """
    Extract detected object information from a YOLO prediction result.

    Args:
        result: Ultralytics prediction result.
        frame_number: Optional video frame number.

    Returns:
        List of detection dictionaries.
    """

    records = []

    if result.boxes is None:
        return records

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    for box, class_id, conf in zip(boxes, classes, confidences):

        x1, y1, x2, y2 = box

        record = {
            "Class": result.names[class_id],
            "Confidence": round(float(conf), 4),
            "X1": round(float(x1), 2),
            "Y1": round(float(y1), 2),
            "X2": round(float(x2), 2),
            "Y2": round(float(y2), 2)
        }

        if frame_number is not None:
            record["Frame"] = frame_number

        records.append(record)

    return records


# --------------------------------------------------
# DETECTION SUMMARY
# --------------------------------------------------

def display_detection_summary(records):

    st.subheader("Detection Summary")

    if not records:
        st.warning(
            "No objects were detected at this confidence threshold."
        )
        return

    df = pd.DataFrame(records)

    st.metric("Total Detections", len(df))

    counts = (
        df["Class"]
        .value_counts()
        .reindex(model.names.values(), fill_value=0)
        .rename_axis("Class")
        .reset_index(name="Count")
    )

    st.write("**Per-Class Detection Counts**")

    st.dataframe(
        counts,
        use_container_width=True,
        hide_index=True
    )

    st.write("**Detection Details**")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# IMAGE DETECTION
# --------------------------------------------------

def image_detection():

    st.header("Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is None:
        st.info("Upload a JPG, JPEG or PNG image to begin.")
        return

    image_bytes = uploaded_image.getvalue()

    if not image_bytes:
        st.error("The uploaded image is empty.")
        return

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:
        logger.exception("Image validation failed")

        st.error("The uploaded image is invalid or corrupted.")
        return

    st.write("**Uploaded Image:**")

    st.image(
        image,
        use_container_width=True
    )

    if st.button("Run Image Detection", type="primary"):

        try:
            with st.spinner("Detecting objects..."):

                results = model.predict(
                    source=image,
                    conf=confidence,
                    device="cpu",
                    verbose=False
                )

            result = results[0]

            annotated_bgr = result.plot()

            annotated_rgb = cv2.cvtColor(
                annotated_bgr,
                cv2.COLOR_BGR2RGB
            )

            st.subheader("Detection Results")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Original Image**")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:
                st.write("**Annotated Image**")

                st.image(
                    annotated_rgb,
                    use_container_width=True
                )

            image_buffer = io.BytesIO()

            Image.fromarray(annotated_rgb).save(
                image_buffer,
                format="PNG"
            )

            st.download_button(
                "Download Annotated Image",
                data=image_buffer.getvalue(),
                file_name="annotated_image.png",
                mime="image/png"
            )

            records = extract_detections(result)

            display_detection_summary(records)

        except Exception:
            logger.exception("Image detection failed")

            st.error(
                "An error occurred during image detection. "
                "Please try another image."
            )


# --------------------------------------------------
# VIDEO DETECTION
# --------------------------------------------------

def video_detection():

    st.header("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload an MP4 video",
        type=["mp4"],
        key="video_upload"
    )

    if uploaded_video is None:
        st.info("Upload an MP4 video to begin.")
        return

    video_bytes = uploaded_video.getvalue()

    if not video_bytes:
        st.error("The uploaded video is empty.")
        return

    st.video(video_bytes)

    if st.button("Run Video Detection", type="primary"):

        input_path = None
        output_path = None
        cap = None
        writer = None

        try:

            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as input_file:

                input_file.write(video_bytes)
                input_path = input_file.name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Unable to read the uploaded video.")
                return

            # Read video properties
            fps = cap.get(cv2.CAP_PROP_FPS)

            if not fps or fps <= 0:
                fps = 25.0

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if width <= 0 or height <= 0:
                st.error("The uploaded video has invalid dimensions.")
                return

            if total_frames <= 0:
                st.error("The video contains no readable frames.")
                return

            # Create temporary output video
            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as output_file:

                output_path = output_file.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            if not writer.isOpened():
                st.error("Unable to create the output video.")
                return

            # Progress indicators
            progress = st.progress(0)
            status = st.empty()

            all_records = []
            frame_number = 0

            status.info("Processing video frames...")

            # Process video frame by frame
            while True:

                success, frame = cap.read()

                if not success:
                    break

                results = model.predict(
                    source=frame,
                    conf=confidence,
                    device="cpu",
                    verbose=False
                )

                result = results[0]

                annotated_frame = result.plot()

                writer.write(annotated_frame)

                all_records.extend(
                    extract_detections(
                        result,
                        frame_number + 1
                    )
                )

                frame_number += 1

                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

            # Release video resources before reading output
            cap.release()
            cap = None

            writer.release()
            writer = None

            if frame_number == 0:
                st.error("No frames could be processed.")
                return

            # Read processed video
            with open(output_path, "rb") as video_file:
                processed_video = video_file.read()

            if not processed_video:
                st.error("The processed video could not be generated.")
                return

            status.success(
                f"Video processing completed: "
                f"{frame_number} frames processed."
            )

            st.subheader("Annotated Video")

            st.video(processed_video)

            st.download_button(
                "Download Annotated Video",
                data=processed_video,
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )

            display_detection_summary(all_records)

        except Exception:
            logger.exception("Video detection failed")

            st.error(
                "An error occurred while processing the video. "
                "Please try another MP4 file."
            )

        finally:

            # Release resources if an error occurs
            if cap is not None:
                cap.release()

            if writer is not None:
                writer.release()

            # Remove temporary files
            for path in [input_path, output_path]:

                if path and os.path.exists(path):

                    try:
                        os.remove(path)

                    except OSError:
                        logger.exception(
                            "Unable to remove temporary file: %s",
                            path
                        )


# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------

image_tab, video_tab = st.tabs(
    ["🖼️ Image Detection", "🎥 Video Detection"]
)

with image_tab:
    image_detection()

with video_tab:
    video_detection()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "YOLO26 Custom Object Detection Project | "
    "Developed using Python, Ultralytics, OpenCV and Streamlit"
)
