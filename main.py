import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from datetime import datetime
import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
import PIL.Image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide"
)

# Main title with styling
st.markdown("""
# 🧠 Brain Tumor Detection System
""")

# Load the model at startup to avoid reloading it on each interaction
@st.cache_resource
def load_brain_tumor_model():
    try:
        model = load_model('brain_tumor_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_brain_tumor_model()

# Function to preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

# Function to generate GradCAM
def generate_gradcam(model, preprocessed_img, layer_name='conv5_block3_out'):
    # Create Gradcam object
    gradcam = Gradcam(model, 
                     model_modifier=None,
                     clone=False)
    
    # Generate heatmap
    cam = gradcam(lambda x: tf.keras.activations.sigmoid(x),
                 preprocessed_img,
                 penultimate_layer=layer_name)
    
    # Normalize
    cam = normalize(cam)
    
    return cam[0]

# Function to create heatmap overlay
def create_heatmap_overlay(original_img_path, heatmap, alpha=0.4):
    # Load original image
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

# Function to generate PDF report
def generate_pdf_report(patient_data, prediction, prediction_probability, gradcam_img_path):
    # Create a temporary file for the PDF
    pdf_buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Add custom styles
    styles.add(ParagraphStyle(name='Title',
                             parent=styles['Heading1'],
                             fontSize=18,
                             alignment=1,
                             spaceAfter=12))
    
    styles.add(ParagraphStyle(name='Subtitle',
                             parent=styles['Heading2'],
                             fontSize=14,
                             spaceAfter=10))
    
    styles.add(ParagraphStyle(name='Normal_Center',
                             parent=styles['Normal'],
                             alignment=1))
    
    elements = []
    
    # Report header
    elements.append(Paragraph("BRAIN MRI ANALYSIS REPORT", styles['Title']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Date and report ID
    report_date = datetime.now().strftime("%B %d, %Y")
    report_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Patient information section
    elements.append(Paragraph("PATIENT INFORMATION", styles['Subtitle']))
    
    patient_info = [
        ['Patient Name:', patient_data['name']],
        ['Patient ID:', patient_data['id']],
        ['Date of Birth:', patient_data['dob']],
        ['Gender:', patient_data['gender']],
        ['Report Date:', report_date],
        ['Report ID:', report_id]
    ]
    
    t = Table(patient_info, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))
    
    # Analysis results section
    elements.append(Paragraph("ANALYSIS RESULTS", styles['Subtitle']))
    
    # Format the probability as a percentage
    probability_percent = f"{prediction_probability:.2%}"
    
    # Define result text and color based on prediction
    if prediction == "Tumor Detected":
        result_text = f"POSITIVE - Brain Tumor Detected (Confidence: {probability_percent})"
        result_color = colors.red
    else:
        result_text = f"NEGATIVE - No Brain Tumor Detected (Confidence: {probability_percent})"
        result_color = colors.green
    
    result_style = ParagraphStyle(
        'ResultStyle', 
        parent=styles['Normal'],
        alignment=1,
        textColor=result_color,
        fontSize=12,
        fontName='Helvetica-Bold'
    )
    
    elements.append(Paragraph(result_text, result_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add original image and GradCAM visualization
    elements.append(Paragraph("MRI SCAN WITH AREA OF INTEREST", styles['Subtitle']))
    
    # Add GradCAM image
    img = Image(gradcam_img_path, width=5*inch, height=5*inch)
    elements.append(img)
    elements.append(Paragraph("GradCAM visualization highlights the regions that influenced the model's decision", styles['Normal_Center']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add disclaimer and notes
    elements.append(Paragraph("NOTES", styles['Subtitle']))
    
    disclaimer_text = """
    This report was generated by an automated AI system and should be reviewed by a qualified healthcare professional.
    The AI detection system has a limited accuracy and should not replace professional medical diagnosis.
    Please consult with a specialist for confirmation of these results and appropriate follow-up.
    """
    
    elements.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Build PDF document
    doc.build(elements)
    
    return pdf_buffer

# Create tabs for the app
tab1, tab2 = st.tabs(["Patient Information", "Results"])

with tab1:
    st.subheader("Patient Information")
    
    # Create form for patient information
    with st.form("patient_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Patient Name *", placeholder="Full Name")
            patient_id = st.text_input("Patient ID *", placeholder="Hospital ID")
            patient_dob = st.date_input("Date of Birth *")
        
        with col2:
            patient_gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
            patient_contact = st.text_input("Contact Number", placeholder="Phone Number")
            patient_email = st.text_input("Email Address", placeholder="Email")
        
        st.markdown("### MRI Scan Upload")
        uploaded_file = st.file_uploader("Upload brain MRI scan (JPEG, PNG, or JPG format)", type=["jpg", "jpeg", "png"])
                
        # Preview the uploaded image
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)
            with col2:
                st.write("Image successfully uploaded. Please verify that the image is clear and properly oriented.")
                st.write("The system works best with T1-weighted or T2-weighted MRI scans.")
        
        submitted = st.form_submit_button("Analyze MRI Scan")
    
    # Check if form is submitted and all required fields are filled
    if submitted:
        # Validate required fields
        if not patient_name or not patient_id or not patient_dob or not uploaded_file:
            st.error("Please fill all required fields and upload an MRI scan image.")
        else:
            # Show loading indicator
            with st.spinner("Processing MRI scan and generating report..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                
                # Preprocess image for model
                preprocessed_img = preprocess_image(temp_path)
                
                # Make prediction
                prediction_prob = model.predict(preprocessed_img)[0][0]
                
                # Interpret result
                if prediction_prob > 0.5:
                    prediction_label = "Tumor Detected"
                    prediction_probability = prediction_prob
                else:
                    prediction_label = "No Tumor Detected"
                    prediction_probability = 1 - prediction_prob
                
                # Generate GradCAM visualization
                gradcam_heatmap = generate_gradcam(model, preprocessed_img)
                heatmap_overlay = create_heatmap_overlay(temp_path, gradcam_heatmap)
                
                # Save GradCAM image temporarily
                gradcam_path = temp_path.replace('.jpg', '_gradcam.jpg')
                cv2.imwrite(gradcam_path, heatmap_overlay)
                
                # Patient data dictionary
                patient_data = {
                    'name': patient_name,
                    'id': patient_id,
                    'dob': patient_dob.strftime('%Y-%m-%d'),
                    'gender': patient_gender,
                    'contact': patient_contact,
                    'email': patient_email
                }
                
                # Generate PDF report
                pdf_buffer = generate_pdf_report(
                    patient_data, 
                    prediction_label, 
                    prediction_probability, 
                    gradcam_path
                )
                
                # Store results in session state to display in results tab
                st.session_state.patient_data = patient_data
                st.session_state.prediction_label = prediction_label
                st.session_state.prediction_probability = prediction_probability
                st.session_state.gradcam_path = gradcam_path
                st.session_state.pdf_buffer = pdf_buffer
                st.session_state.analysis_complete = True
                
                # Switch to results tab
                st.experimental_rerun()

with tab2:
    if st.session_state.get('analysis_complete', False):
        st.subheader("Analysis Results")
        
        # Display patient information
        st.write(f"**Patient:** {st.session_state.patient_data['name']} (ID: {st.session_state.patient_data['id']})")
        
        # Display prediction result with appropriate styling
        if st.session_state.prediction_label == "Tumor Detected":
            st.error(f"Result: {st.session_state.prediction_label} (Confidence: {st.session_state.prediction_probability:.2%})")
        else:
            st.success(f"Result: {st.session_state.prediction_label} (Confidence: {st.session_state.prediction_probability:.2%})")
        
        # Display GradCAM visualization
        st.image(st.session_state.gradcam_path, caption="GradCAM Visualization - Regions of Interest", use_column_width=True)
        
        # Provide download link for PDF report
        st.session_state.pdf_buffer.seek(0)
        st.download_button(
            label="Download Full Medical Report (PDF)",
            data=st.session_state.pdf_buffer,
            file_name=f"brain_mri_report_{st.session_state.patient_data['id']}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
        
        # Additional information
        st.info("""
        **Understanding the Report:**
        - The GradCAM visualization highlights areas that influenced the model's decision
        - Red/yellow areas indicate regions of higher importance to the prediction
        - This automated analysis should be reviewed by a healthcare professional
        """)
    else:
        st.info("Please enter patient information and upload an MRI scan in the Patient Information tab to see results here.")

# Add footer with disclaimer
st.markdown("---")
st.caption("""
**Disclaimer:** This application is for demonstration purposes only and is not intended to replace professional medical diagnosis.
Always consult with a qualified healthcare provider regarding medical conditions and treatment options.
""")