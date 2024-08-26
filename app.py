import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('MPox_Model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define the target size for images
TARGET_SIZE = (224, 224)  # Adjust this to match your model's input size

# Define symptoms
general_symptoms = ["Fever", "Sore throat", "Headache", "Muscle aches", "Back pain", "Low energy"]
skin_symptoms = ["Skin rash or lesions", "Swollen lymph nodes"]

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'personal_info' not in st.session_state:
    st.session_state.personal_info = {
        'name': '',
        'phone': '',
        'place': '',
        'age': ''
    }
if 'selected_general_symptoms' not in st.session_state:
    st.session_state.selected_general_symptoms = []
if 'selected_skin_symptoms' not in st.session_state:
    st.session_state.selected_skin_symptoms = []
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'show_errors' not in st.session_state:
    st.session_state.show_errors = False

def navigate_to_page(page_name):
    st.session_state.page = page_name

# Home Page
if st.session_state.page == 'Home':
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 10px;'>
            <h1>Mpox Identification App</h1>
        </div>
        <div style='text-align: justify; margin: 0 10px; padding: 0;'>
            <p>Mpox, also known as monkeypox, is a viral illness with global outbreaks. 
            Our app helps you quickly identify symptoms, keeping you informed and secure. Your well-being is our top priority.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Centered and styled button using Streamlit's `st.button`
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start", key='start_button', use_container_width=True):
            navigate_to_page('Personal Info')

    # Display the note at the bottom
    st.markdown(
        """
        <div style='text-align: center; margin-top: 30px; color: gray;'>
            <p><strong>Note:</strong> This app is part of a school project and is intended for educational purposes only.</p>
        </div>
        """, unsafe_allow_html=True
    )

# Page 1: Collect Personal Information
elif st.session_state.page == 'Personal Info':
    st.title("Personal Information")

    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input("Enter your first name:", max_chars=40)
        if st.session_state.show_errors and not name:
            st.markdown('<p style="color:red;">Name is required.</p>', unsafe_allow_html=True)
    
    with col1:
        phone = st.text_input("Enter your phone number (e.g., +91-XXXXX-XXXXX):", max_chars=10, help="Only Indian numbers are allowed.")
        if st.session_state.show_errors:
            if not phone:
                st.markdown('<p style="color:red;">Phone number is required.</p>', unsafe_allow_html=True)
            elif not phone.isdigit() or len(phone) != 10 or phone[0] not in ['6', '7', '8', '9']:
                st.markdown('<p style="color:red;">Please enter a valid 10-digit Indian phone number.</p>', unsafe_allow_html=True)
    
    with col1:
        place = st.text_input("Enter your place:", max_chars=25)
        if st.session_state.show_errors and not place:
            st.markdown('<p style="color:red;">Place is required.</p>', unsafe_allow_html=True)
    
    with col1:
        age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
        if st.session_state.show_errors and age == 0:
            st.markdown('<p style="color:red;">Age is required.</p>', unsafe_allow_html=True)

    if st.button("Next"):
        st.session_state.show_errors = True
        if name and phone and place and age:
            st.session_state.personal_info = {
                'name': name,
                'phone': phone,
                'place': place,
                'age': age
            }
            navigate_to_page('Symptoms')
        else:
            st.session_state.show_errors = True





# Page 2: Symptoms Page
elif st.session_state.page == 'Symptoms':
    st.title("Mpox Identification")

    # Initialize session state for form submission
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # General Symptoms Selection
    st.subheader("General Symptoms")
    selected_general_symptoms = st.multiselect("Select General Symptoms", general_symptoms)

    # Display error for general symptoms if form is submitted but no selection
    if st.session_state.submitted and not selected_general_symptoms:
        st.error("Please select at least one general symptom.")

    # Skin Symptoms Selection
    st.subheader("Skin Symptoms")
    selected_skin_symptoms = st.multiselect("Select Skin Symptoms", skin_symptoms)

    # Display error for skin symptoms if form is submitted but no selection
    if st.session_state.submitted and not selected_skin_symptoms:
        st.error("Please select at least one skin symptom.")

    # Image Upload
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Upload an image of the symptom...", type=["jpg", "jpeg", "png"])

    # Display error for image upload if form is submitted but no file uploaded
    if st.session_state.submitted and not uploaded_file:
        st.error("Please upload an image of the symptom.")

    # Submit button
    if st.button("Submit"):
        # Mark the form as submitted
        st.session_state.submitted = True

        # Validate the form
        if selected_general_symptoms and selected_skin_symptoms and uploaded_file:
            # If valid, proceed to the result page
            st.session_state.selected_general_symptoms = selected_general_symptoms
            st.session_state.selected_skin_symptoms = selected_skin_symptoms
            st.session_state.uploaded_file = uploaded_file
            navigate_to_page('Result')
        else:
            # If not valid, errors will be shown as set by session state
            st.session_state.submitted = True



# Page 3: Result Page
elif st.session_state.page == 'Result':
    st.title("Mpox Identification Results")

    # Retrieve selected symptoms and uploaded file from session state
    general_symptoms = st.session_state.get('selected_general_symptoms', [])
    skin_symptoms = st.session_state.get('selected_skin_symptoms', [])
    uploaded_file = st.session_state.get('uploaded_file', None)

    # Process the image (similar to the previous code)
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict with the model
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction[0])
    class_labels = ["Mpox", "Normal"]
    predicted_class = class_labels[class_index]

    # Generate a professional result message
    if predicted_class == "Mpox":
        result_message = f"""
        <div style="padding: 15px; border: 2px solid #FF4C4C; border-radius: 10px; background-color: #FFEDED;">
            <h3 style="color: #FF4C4C;">Mpox Detected</h3>
            <p>
                <strong>Symptoms Reported:</strong><br>
                <ul>
                    <li>General: {', '.join(general_symptoms)}</li>
                    <li>Skin: {', '.join(skin_symptoms)}</li>
                </ul>
            </p>
            <p>
                Based on the symptoms you reported and the analysis of the uploaded image, there is a strong indication 
                that you might be affected by <strong>Mpox</strong>.
            </p>
            <p>
                <strong>Next Steps:</strong><br>
                We highly recommend consulting a healthcare professional immediately for further diagnosis and treatment.
                Your health and safety are our top priorities.
            </p>
        </div>
        """
    else:
        result_message = f"""
        <div style="padding: 15px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #E8F5E9;">
            <h3 style="color: #4CAF50;">No Mpox Detected</h3>
            <p>
                <strong>Symptoms Reported:</strong><br>
                <ul>
                    <li>General: {', '.join(general_symptoms)}</li>
                    <li>Skin: {', '.join(skin_symptoms)}</li>
                </ul>
            </p>
            <p>
                Based on the symptoms you reported and the analysis of the uploaded image, there is no indication 
                of <strong>Mpox</strong> at this time.
            </p>
            <p>
                <strong>Next Steps:</strong><br>
                While Mpox has been ruled out, we advise you to consult a doctor to address the symptoms 
                you've been experiencing. Staying informed and seeking professional guidance is key to your well-being.
            </p>
        </div>
        """

    # Display the result message with Markdown
    st.markdown(result_message, unsafe_allow_html=True)
