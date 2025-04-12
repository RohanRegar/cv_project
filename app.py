# import streamlit as st
# import importlib
# from PIL import Image
# import tempfile
# import subprocess
# import os
# import cv2
# import numpy as np
# import models.sgm as sgm_module
# st.set_page_config(page_title="Model Runner App")
# st.title("🧠 Model Runner App")

# model_options = {
#     "Depth Estimation (MiDaS)": "midas_depth",
#     "DenseDepth (FPN-DenseNet)": "densedepth",
#     "DepthAnything (EfficientNet-FPN)": "depth_anything_custom",
#     "Stereo Matching (SGM)": "sgm"
# }


# # 📥 Input type selection
# st.subheader("📥 Select Input Source")
# selected_model = st.selectbox("Select a model to run:", list(model_options.keys()))

# # For image-based models (MiDaS and DenseDepth)
# if selected_model in ["Depth Estimation (MiDaS)", "DenseDepth (FPN-DenseNet)", "DepthAnything (EfficientNet-FPN)"]:
#     input_source = st.radio("Choose input type:", ["Webcam", "Upload Image"])
#     uploaded_file = None
#     if input_source == "Upload Image":
#         uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# # For SGM model (stereo image pair)
# elif selected_model == "Stereo Matching (SGM)":
#     st.subheader("Upload Stereo Image Pair for SGM")
#     left_img = st.file_uploader("Left Image", type=["png", "jpg", "jpeg"], key="left")
#     right_img = st.file_uploader("Right Image", type=["png", "jpg", "jpeg"], key="right")
#     left_gt = st.file_uploader("Left GT Image (optional)", type=["png", "jpg", "jpeg"], key="left_gt")
#     right_gt = st.file_uploader("Right GT Image (optional)", type=["png", "jpg", "jpeg"], key="right_gt")
#     max_disp = st.slider("Max Disparity", min_value=1, max_value=100, step=1, value=64)
#     show_images = st.checkbox("Show Intermediate Images", value=True)
#     do_eval = st.checkbox("Evaluate Accuracy (needs GT)", value=False)

# # ▶️ Run Model
# if st.button("Run Model"):
#     try:
#         with st.spinner("Running model..."):
#             # MiDaS or DenseDepth
#             if selected_model in ["Depth Estimation (MiDaS)", "DenseDepth (FPN-DenseNet)", "DepthAnything (EfficientNet-FPN)"]:
#                 module = importlib.import_module(f"models.{model_options[selected_model]}")
#                 input_img = None

#                 # Webcam
#                 if input_source == "Webcam":
#                     cap = cv2.VideoCapture(0)
#                     ret, frame = cap.read()
#                     cap.release()
#                     if not ret:
#                         st.error("Could not capture webcam frame.")
#                         st.stop()
#                     input_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#                 # Uploaded file
#                 elif input_source == "Upload Image" and uploaded_file is not None:
#                     input_img = Image.open(uploaded_file).convert("RGB")

#                 if input_img is None:
#                     st.warning("No image provided.")
#                     st.stop()

#                 input_img, output_img = module.run(input_img)

#                 if input_img and output_img:
#                     st.subheader("📸 Input vs 🧠 Depth Output")
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.image(input_img, caption="Input Image", use_column_width=True)
#                     with col2:
#                         st.image(output_img, caption="Depth Map Output", use_column_width=True)
#                     st.success(f"{selected_model} ran successfully!")
#                 else:
#                     st.warning("Model returned empty result.")

#             # Stereo Matching (SGM)
#             elif selected_model == "Stereo Matching (SGM)":
#                 if not (left_img and right_img):
#                     st.error("Please upload both left and right images.")
#                 else:
#                     with tempfile.TemporaryDirectory() as tmpdir:
#                         left_path = os.path.join(tmpdir, "left.png")
#                         right_path = os.path.join(tmpdir, "right.png")
#                         out_path = os.path.join(tmpdir, "output.png")

#                         with open(left_path, "wb") as f:
#                             f.write(left_img.read())
#                         with open(right_path, "wb") as f:
#                             f.write(right_img.read())

#                         left_gt_path = os.path.join(tmpdir, "left_gt.png") if left_gt else None
#                         right_gt_path = os.path.join(tmpdir, "right_gt.png") if right_gt else None

#                         if left_gt:
#                             with open(left_gt_path, "wb") as f:
#                                 f.write(left_gt.read())
#                         if right_gt:
#                             with open(right_gt_path, "wb") as f:
#                                 f.write(right_gt.read())

#                         # ✅ Call the Python function directly
#                         left_disp_path, right_disp_path, recall_list = sgm_module.sgm(
#                             left_path=left_path,
#                             right_path=right_path,
#                             left_gt_path=left_gt_path,
#                             right_gt_path=right_gt_path,
#                             output_path=out_path,
#                             max_disp=max_disp,
#                             show_images=show_images,
#                             eval_mode=do_eval
#                         )

#                         st.subheader("📸 Input Images and 🧠 Disparity Outputs")
#                         col1, col2 = st.columns(2)

#                         with col1:
#                             st.image(Image.open(left_path), caption="Left Image", use_column_width=True)
#                             st.image(Image.open(left_disp_path), caption="Left Disparity Map", use_column_width=True)

#                         with col2:
#                             st.image(Image.open(right_path), caption="Right Image", use_column_width=True)
#                             st.image(Image.open(right_disp_path), caption="Right Disparity Map", use_column_width=True)

#                         if do_eval and recall_list:
#                             st.subheader("📊 Evaluation Results")
#                             for recall in recall_list:
#                                 st.info(recall)

#                         st.success("SGM model ran successfully!")
#     except Exception as e:
#         st.error(f"Error running {selected_model}: {e}")
# import streamlit as st
# import os
# import shutil
# import subprocess
# from PIL import Image
# import cv2

# # Constants
# ROOT_DIR = r"C:\Users\mundh\OneDrive\Coding\Simplycoding\pytorch-CycleGAN-and-pix2pix"
# DATAROOT = os.path.join(ROOT_DIR, "imgs_test")
# TRAINA = os.path.join(DATAROOT, "trainA")
# TRAINB = os.path.join(DATAROOT, "trainB")
# OUTPUT_PATH = os.path.join(ROOT_DIR, "results", "mat5", "test_latest", "images", "input_fake_B.png")

# # Title
# st.set_page_config(page_title="Pix2Pix Demo")
# st.title("🌀 Pix2Pix Image Translation")

# # Input
# input_mode = st.radio("Select input method:", ["Upload Image", "Webcam"])
# image_file = None

# if input_mode == "Upload Image":
#     image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# elif input_mode == "Webcam":
#     if st.button("📸 Capture from Webcam"):
#         cap = cv2.VideoCapture(0)
#         ret, frame = cap.read()
#         cap.release()
#         if ret:
#             image_file = "webcam_input.png"
#             cv2.imwrite(image_file, frame)
#             st.image(frame, caption="Captured Image", channels="BGR")
#         else:
#             st.error("Webcam capture failed.")

# # Run Pix2Pix
# if st.button("Run Pix2Pix") and image_file:
#     try:
#         # Clean folders
#         for folder in [TRAINA, TRAINB]:
#             if os.path.exists(folder):
#                 shutil.rmtree(folder)
#             os.makedirs(folder)

#         # Save image to trainA/trainB
#         filename = "input.png"
#         trainA_path = os.path.join(TRAINA, filename)
#         trainB_path = os.path.join(TRAINB, filename)

#         if isinstance(image_file, str):  # webcam
#             shutil.copy(image_file, trainA_path)
#             shutil.copy(image_file, trainB_path)
#         else:  # uploaded file
#             content = image_file.read()
#             with open(trainA_path, "wb") as f:
#                 f.write(content)
#             with open(trainB_path, "wb") as f:
#                 f.write(content)

#         # Run test.py
#         command = [
#             "python", "test.py",
#             "--dataroot", "imgs_test",
#             "--name", "mat5",
#             "--model", "pix2pix",
#             "--direction", "AtoB",
#             "--no_flip",
#             "--dataset_mode", "template",
#             "--input_nc", "3",
#             "--output_nc", "1"
#         ]
#         result = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)

#         if result.returncode != 0:
#             st.error("Error running model:")
#             st.code(result.stderr)
#         elif not os.path.exists(OUTPUT_PATH):
#             st.error("Output image not found.")
#         else:
#             st.subheader("🖼️ Input vs 🎨 Output")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(Image.open(trainA_path), caption="Input Image", use_column_width=True)
#             with col2:
#                 st.image(Image.open(OUTPUT_PATH), caption="Pix2Pix Output", use_column_width=True)
#             st.success("Pix2Pix model executed successfully!")

#     except Exception as e:
#         st.error(f"Error: {e}")


import streamlit as st
import importlib
from PIL import Image
import tempfile
import subprocess
import os
import shutil
import cv2
import numpy as np
import models.sgm as sgm_module

# Constants for Pix2Pix
PIX2PIX_ROOT = r"models\pytorch-CycleGAN-and-pix2pix"
DATAROOT = os.path.join(PIX2PIX_ROOT, "imgs_test")
TRAINA = os.path.join(DATAROOT, "trainA")
TRAINB = os.path.join(DATAROOT, "trainB")
PIX2PIX_OUTPUT_PATH = os.path.join(PIX2PIX_ROOT, "results", "mat5", "test_latest", "images", "input_fake_B.png")

st.set_page_config(page_title="Model Runner App")
st.title("🧠 Model Runner App")

model_options = {
    "Depth Estimation (MiDaS)": "midas_depth",
    "DenseDepth (FPN-DenseNet)": "densedepth",
    "DepthAnything (EfficientNet-FPN)": "depth_anything_custom",
    "Stereo Matching (SGM)": "sgm",
    "Pix2Pix Image Translation": "pix2pix"
}

# Input source selection
st.subheader("📥 Select Input Source")
selected_model = st.selectbox("Select a model to run:", list(model_options.keys()))

# Shared variables
uploaded_file = None

# For single image models
if selected_model in ["Depth Estimation (MiDaS)", "DenseDepth (FPN-DenseNet)", "DepthAnything (EfficientNet-FPN)"]:
    input_source = st.radio("Choose input type:", ["Webcam", "Upload Image"])
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

elif selected_model == "Stereo Matching (SGM)":
    st.subheader("Upload Stereo Image Pair for SGM")
    left_img = st.file_uploader("Left Image", type=["png", "jpg", "jpeg"], key="left")
    right_img = st.file_uploader("Right Image", type=["png", "jpg", "jpeg"], key="right")
    left_gt = st.file_uploader("Left GT Image (optional)", type=["png", "jpg", "jpeg"], key="left_gt")
    right_gt = st.file_uploader("Right GT Image (optional)", type=["png", "jpg", "jpeg"], key="right_gt")
    max_disp = st.slider("Max Disparity", min_value=1, max_value=100, step=1, value=64)
    show_images = st.checkbox("Show Intermediate Images", value=True)
    do_eval = st.checkbox("Evaluate Accuracy (needs GT)", value=False)

elif selected_model == "Pix2Pix Image Translation":
    uploaded_file = st.file_uploader("Upload an image for Pix2Pix", type=["png", "jpg", "jpeg"])

# Run Button
if st.button("Run Model"):
    try:
        with st.spinner("Running model..."):
            if selected_model in ["Depth Estimation (MiDaS)", "DenseDepth (FPN-DenseNet)", "DepthAnything (EfficientNet-FPN)"]:
                module = importlib.import_module(f"models.{model_options[selected_model]}")
                input_img = None

                if input_source == "Webcam":
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        st.error("Could not capture webcam frame.")
                        st.stop()
                    input_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                elif input_source == "Upload Image" and uploaded_file is not None:
                    input_img = Image.open(uploaded_file).convert("RGB")

                if input_img is None:
                    st.warning("No image provided.")
                    st.stop()

                input_img, output_img = module.run(input_img)

                st.subheader("📸 Input vs 🧠 Depth Output")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(input_img, caption="Input Image", use_column_width=True)
                with col2:
                    st.image(output_img, caption="Depth Map Output", use_column_width=True)
                st.success(f"{selected_model} ran successfully!")

            elif selected_model == "Stereo Matching (SGM)":
                if not (left_img and right_img):
                    st.error("Please upload both left and right images.")
                    st.stop()

                with tempfile.TemporaryDirectory() as tmpdir:
                    left_path = os.path.join(tmpdir, "left.png")
                    right_path = os.path.join(tmpdir, "right.png")
                    out_path = os.path.join(tmpdir, "output.png")

                    with open(left_path, "wb") as f:
                        f.write(left_img.read())
                    with open(right_path, "wb") as f:
                        f.write(right_img.read())

                    left_gt_path = os.path.join(tmpdir, "left_gt.png") if left_gt else None
                    right_gt_path = os.path.join(tmpdir, "right_gt.png") if right_gt else None

                    if left_gt:
                        with open(left_gt_path, "wb") as f:
                            f.write(left_gt.read())
                    if right_gt:
                        with open(right_gt_path, "wb") as f:
                            f.write(right_gt.read())

                    left_disp_path, right_disp_path, recall_list = sgm_module.sgm(
                        left_path, right_path, left_gt_path, right_gt_path,
                        out_path, max_disp, show_images, do_eval
                    )

                    st.subheader("📸 Input Images and 🧠 Disparity Outputs")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(Image.open(left_path), caption="Left Image", use_column_width=True)
                        st.image(Image.open(left_disp_path), caption="Left Disparity Map", use_column_width=True)
                    with col2:
                        st.image(Image.open(right_path), caption="Right Image", use_column_width=True)
                        st.image(Image.open(right_disp_path), caption="Right Disparity Map", use_column_width=True)

                    if do_eval and recall_list:
                        st.subheader("📊 Evaluation Results")
                        for recall in recall_list:
                            st.info(recall)

                    st.success("SGM model ran successfully!")

            elif selected_model == "Pix2Pix Image Translation":
                if uploaded_file is None:
                    st.warning("Please upload an image.")
                    st.stop()

                # Clear and create folders
                for folder in [TRAINA, TRAINB]:
                    if os.path.exists(folder):
                        shutil.rmtree(folder)
                    os.makedirs(folder)

                filename = "input.png"
                trainA_path = os.path.join(TRAINA, filename)
                trainB_path = os.path.join(TRAINB, filename)

                content = uploaded_file.read()
                with open(trainA_path, "wb") as f:
                    f.write(content)
                with open(trainB_path, "wb") as f:
                    f.write(content)

                command = [
                    "python", "test.py",
                    "--dataroot", "imgs_test",
                    "--name", "mat5",
                    "--model", "pix2pix",
                    "--direction", "AtoB",
                    "--no_flip",
                    "--dataset_mode", "template",
                    "--input_nc", "3",
                    "--output_nc", "1"
                ]

                result = subprocess.run(command, cwd=PIX2PIX_ROOT, capture_output=True, text=True)

                if result.returncode != 0:
                    st.error("Error running Pix2Pix model:")
                    st.code(result.stderr)
                elif not os.path.exists(PIX2PIX_OUTPUT_PATH):
                    st.error("Output image not found.")
                else:
                    st.subheader("🖼️ Input vs 🎨 Output")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(Image.open(trainA_path), caption="Input Image", use_column_width=True)
                    with col2:
                        st.image(Image.open(PIX2PIX_OUTPUT_PATH), caption="Pix2Pix Output", use_column_width=True)
                    st.success("Pix2Pix model ran successfully!")

    except Exception as e:
        st.error(f"Error running {selected_model}: {e}")
