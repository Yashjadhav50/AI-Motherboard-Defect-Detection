import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cv2
from io import BytesIO


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Motherboard Inspector",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.author-name {
    font-size: 20px;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #AAB2C0;
    margin-bottom: 25px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1E2028;
    border: 1px solid #30333D;
    text-align: center;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
}

.metric-label {
    color: #AAB2C0;
    font-size: 14px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "models/best.pt"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# =========================================================
# DEFECT CLASSES
# =========================================================

DEFECT_CLASSES = {
    "CPU_FAN_NO_Screws",
    "CPU_FAN_Screw_loose",
    "CPU_fan_port_detached",
    "Incorrect_Screws",
    "Loose_Screws",
    "No_Screws",
    "Scratch"
}


# =========================================================
# SESSION STATE
# =========================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

if "detections" not in st.session_state:
    st.session_state.detections = None


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🔧 AI-Powered Motherboard Inspector'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="author-name">'
    'Developed by Yash Jadhav'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'YOLO-based motherboard component and defect detection system'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Detection Settings")


confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)


st.sidebar.markdown("---")


# =========================================================
# MODEL INFORMATION
# =========================================================

st.sidebar.markdown("### 🤖 Model")

st.sidebar.write("**YOLO11n**")
st.sidebar.write("Input Size: **640 × 640**")


st.sidebar.markdown("---")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.sidebar.markdown("### 📊 Test Performance")

st.sidebar.write("mAP@50: **87.67%**")
st.sidebar.write("Precision: **83.69%**")
st.sidebar.write("Recall: **82.52%**")


st.sidebar.markdown("---")


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.sidebar.markdown("### 🧠 Project")

st.sidebar.write(
    "AI-powered computer vision system "
    "for motherboard component and defect inspection."
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📷 Upload a motherboard image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear motherboard or PCB image."
)


# =========================================================
# PROCESS UPLOADED IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    st.session_state.original_image = image_array


    # =====================================================
    # IMAGE PREVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">📷 Uploaded Image</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )


    # =====================================================
    # DETECTION BUTTON
    # =====================================================

    detect_button = st.button(
        "🔍 Detect Components & Defects",
        type="primary",
        use_container_width=True
    )


    # =====================================================
    # RUN YOLO
    # =====================================================

    if detect_button:

        with st.spinner(
            "🤖 AI model is analyzing the motherboard..."
        ):

            results = model.predict(
                source=image_array,
                conf=confidence,
                imgsz=640,
                verbose=False
            )


        result = results[0]

        st.session_state.result = result


        # =================================================
        # STORE DETECTIONS
        # =================================================

        detections = []


        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence_score = float(
                    box.conf[0]
                )

                class_name = result.names[class_id]


                if class_name in DEFECT_CLASSES:

                    object_type = "Defect"


                    if confidence_score >= 0.80:

                        priority = "High"

                    elif confidence_score >= 0.60:

                        priority = "Medium"

                    else:

                        priority = "Low"

                else:

                    object_type = "Component"

                    priority = "Normal"


                detections.append({

                    "Class": class_name,

                    "Confidence": confidence_score,

                    "Type": object_type,

                    "Priority": priority

                })


        st.session_state.detections = detections


        # =================================================
        # CREATE NORMAL ANNOTATED IMAGE
        # =================================================

        annotated_image = result.plot()

        # YOLO returns BGR
        # Streamlit expects RGB

        annotated_image = annotated_image[:, :, ::-1]

        st.session_state.annotated_image = annotated_image


        st.success(
            "✅ Inspection completed successfully."
        )


# =========================================================
# DISPLAY RESULTS
# =========================================================

if (
    st.session_state.result is not None
    and st.session_state.detections is not None
):

    result = st.session_state.result

    detections = st.session_state.detections

    original_image = st.session_state.original_image

    annotated_image = st.session_state.annotated_image


    # =====================================================
    # VISUALIZATION SETTINGS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Detection Visualization'
        '</div>',
        unsafe_allow_html=True
    )


    show_defects_only = st.checkbox(
        "🚨 Show only defects",
        value=False
    )


    # =====================================================
    # CREATE DEFECT-ONLY IMAGE
    # =====================================================

    if show_defects_only:

        defect_image = original_image.copy()


        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = result.names[class_id]

                confidence_score = float(
                    box.conf[0]
                )


                if class_name in DEFECT_CLASSES:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )


                    # Draw bounding box

                    cv2.rectangle(
                        defect_image,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        4
                    )


                    # Label

                    label = (
                        f"{class_name} "
                        f"{confidence_score:.2f}"
                    )


                    (text_width, text_height), _ = (
                        cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            2
                        )
                    )


                    # Label background

                    cv2.rectangle(
                        defect_image,
                        (
                            x1,
                            max(0, y1 - text_height - 12)
                        ),
                        (
                            x1 + text_width + 8,
                            y1
                        ),
                        (255, 0, 0),
                        -1
                    )


                    # Label text

                    cv2.putText(
                        defect_image,
                        label,
                        (
                            x1 + 4,
                            max(20, y1 - 6)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )


        display_image = defect_image


    else:

        display_image = annotated_image


    # =====================================================
    # ORIGINAL VS DETECTION
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### 📷 Original")

        st.image(
            original_image,
            use_container_width=True
        )


    with col2:

        if show_defects_only:

            st.markdown("### 🚨 Defects Only")

        else:

            st.markdown("### 🎯 AI Detection")


        st.image(
            display_image,
            use_container_width=True
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    if detections:

        df = pd.DataFrame(detections)


        total_detections = len(df)


        defect_count = len(
            df[df["Type"] == "Defect"]
        )


        component_count = len(
            df[df["Type"] == "Component"]
        )


        unique_classes = df["Class"].nunique()


        # =================================================
        # INSPECTION SUMMARY
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Inspection Summary'
            '</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {total_detections}
                </div>

                <div class="metric-label">
                Total Detections
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {defect_count}
                </div>

                <div class="metric-label">
                Defects
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {component_count}
                </div>

                <div class="metric-label">
                Components
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col4:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {unique_classes}
                </div>

                <div class="metric-label">
                Classes Found
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # DEFECT STATUS
        # =================================================

        st.markdown("")


        if defect_count > 0:

            st.error(
                f"⚠️ {defect_count} potential defect(s) detected."
            )

        else:

            st.success(
                "✅ No potential defects detected."
            )


        # =================================================
        # DETECTION DETAILS
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Detection Details'
            '</div>',
            unsafe_allow_html=True
        )


        display_df = df.copy()


        display_df["Confidence"] = (

            display_df["Confidence"] * 100

        ).round(2).astype(str) + "%"


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # DEFECT SUMMARY
        # =================================================

        defect_df = df[
            df["Type"] == "Defect"
        ]


        if not defect_df.empty:

            st.markdown(
                '<div class="section-title">'
                '🚨 Defect Summary'
                '</div>',
                unsafe_allow_html=True
            )


            defect_summary = (

                defect_df

                .groupby("Class")

                .agg(

                    Count=("Class", "count"),

                    Max_Confidence=(
                        "Confidence",
                        "max"
                    )

                )

                .reset_index()

            )


            defect_summary[
                "Max_Confidence"
            ] = (

                defect_summary[
                    "Max_Confidence"
                ]

                * 100

            ).round(2).astype(str) + "%"


            st.dataframe(
                defect_summary,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # DOWNLOAD SECTION
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📥 Export Results'
            '</div>',
            unsafe_allow_html=True
        )


        col1, col2 = st.columns(2)


        # =================================================
        # CSV DOWNLOAD
        # =================================================

        with col1:

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(

                label="📊 Download Detection Report",

                data=csv_data,

                file_name=(
                    "motherboard_detection_report.csv"
                ),

                mime="text/csv",

                use_container_width=True

            )


        # =================================================
        # IMAGE DOWNLOAD
        # =================================================

        with col2:

            image_to_download = Image.fromarray(
                display_image
            )


            image_buffer = BytesIO()


            image_to_download.save(
                image_buffer,
                format="JPEG"
            )


            st.download_button(

                label="🖼️ Download Annotated Image",

                data=image_buffer.getvalue(),

                file_name=(
                    "motherboard_detection_result.jpg"
                ),

                mime="image/jpeg",

                use_container_width=True

            )


    else:

        st.warning(
            "No objects detected at the selected "
            "confidence threshold."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Developed by Yash Jadhav | "
    "AI-Powered Motherboard Inspector | "
    "YOLO11n Computer Vision | "
    "Capstone 2 Project"
)