import streamlit as st
import requests
import base64
import io
from PIL import Image
import time
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Nano Banana Studio",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mode-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .success-msg {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .error-msg {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .image-container {
        margin: 1rem 0;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = "unknown"

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API Configuration
    API_BASE = st.text_input(
        "FastAPI Backend URL", 
        value=os.getenv("API_BASE", "http://localhost:8000"),
        help="Enter the URL of your FastAPI backend server"
    )
    
    api_key = st.text_input(
        "Gemini API Key", 
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Your Google Gemini API key for image generation"
    )
    
    # API Status Check
    if st.button("🔍 Check API Status"):
        with st.spinner("Checking API connection..."):
            try:
                resp = requests.get(f"{API_BASE}/health", timeout=10)
                if resp.status_code == 200:
                    st.session_state.api_status = "connected"
                    st.success("✅ API Connected")
                else:
                    st.session_state.api_status = "error"
                    st.error("❌ API Error")
            except Exception as e:
                st.session_state.api_status = "error"
                st.error(f"❌ Connection failed: {str(e)}")
    
    # App Info
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **Nano Banana Studio** is a creative AI image generation and editing platform.
    
    **Features:**
    - Text-to-Image Generation
    - Image Editing & Restoration
    - Virtual Try-On
    - Ad Creation
    - Image Merging
    - Scene Generation
    """)
    
    # Download generated images
    if st.session_state.generated_images:
        st.markdown("---")
        st.markdown("### 💾 Downloads")
        for i, (img_b64, timestamp) in enumerate(st.session_state.generated_images):
            img_bytes = base64.b64decode(img_b64)
            st.download_button(
                label=f"📥 Download Image {i+1}",
                data=img_bytes,
                file_name=f"nano_banana_{timestamp}.png",
                mime="image/png",
                key=f"download_{i}"
            )

# Main header
st.markdown('<h1 class="main-header">🍌 Nano Banana Studio</h1>', unsafe_allow_html=True)
st.markdown("### Welcome to your creative AI image studio! Select a mode below to get started.")

# Mode selection with better organization
modes = {
    "Beginner": [
        "📝 Generate Image with Text",
        "✏️ Edit Image with Prompt"
    ],
    "Advanced": [
        "👗 Virtual Try On", 
        "🔧 Restore Old Image"
    ],
    "Professional": [
        "📢 Create Ads",
        "🔗 Merge Images", 
        "🎭 Generate Scenes"
    ]
}

# Mode selector with categories
selected_category = st.selectbox("Select Category", list(modes.keys()))
mode = st.selectbox("Select Mode", modes[selected_category])

# Helper functions
def validate_api_config():
    """Validate API configuration"""
    if not API_BASE:
        st.error("❌ Please enter FastAPI backend URL")
        return False
    if not api_key:
        st.error("❌ Please enter Gemini API Key")
        return False
    return True

def make_api_request(endpoint, data=None, files=None, success_msg="Operation completed successfully!"):
    """Generic API request handler with progress and error handling"""
    if not validate_api_config():
        return None
    
    try:
        with st.spinner("🔄 Processing your request... This may take a few moments."):
            progress_bar = st.progress(0)
            
            # Simulate progress for better UX
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            if files:
                response = requests.post(f"{API_BASE}/{endpoint}", data=data, files=files, timeout=60)
            else:
                response = requests.post(f"{API_BASE}/{endpoint}", data=data, timeout=60)
            
            progress_bar.empty()
            
            if response.status_code == 200:
                st.success(success_msg)
                return response.json()
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Please check the URL and ensure the server is running.")
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The server is taking too long to respond.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
    
    return None

def display_image(img_b64, caption="Generated Image", use_column_width=True):
    """Display base64 image and store in session"""
    try:
        img_bytes = base64.b64decode(img_b64)
        
        # Create a container for the image
        with st.container():
            st.markdown(f"**{caption}**")
            st.image(img_bytes, use_column_width=use_column_width)
            
            # Store in session for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.generated_images.append((img_b64, timestamp))
            
    except Exception as e:
        st.error(f"❌ Failed to display image: {str(e)}")

def display_multiple_images(images_data, captions=None, columns=2):
    """Display multiple images in a grid layout"""
    if not images_data:
        return
    
    if captions is None:
        captions = [f"Image {i+1}" for i in range(len(images_data))]
    
    cols = st.columns(min(columns, len(images_data)))
    
    for idx, (img_data, caption) in enumerate(zip(images_data, captions)):
        with cols[idx % columns]:
            if isinstance(img_data, dict) and 'image' in img_data:
                display_image(img_data['image'], caption)
            elif isinstance(img_data, str):
                display_image(img_data, caption)
            else:
                st.error(f"Invalid image data format for {caption}")

def image_upload_section(label, key, help_text="Upload an image file", accept_types=["png", "jpg", "jpeg"]):
    """Standardized image upload section"""
    st.markdown(f"**{label}**")
    uploaded_file = st.file_uploader(
        help_text, 
        key=key,
        type=accept_types,
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(uploaded_file, caption="Preview", use_column_width=True)
        with col2:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            st.write("File details:")
            st.json(file_details)
    
    return uploaded_file

# Mode-specific implementations
if "Generate Image with Text" in mode:
    st.markdown("### 📝 Generate Image from Text")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Describe your image",
            placeholder="A majestic dragon flying over a medieval castle at sunset, fantasy art style...",
            height=100
        )
        
        # Advanced options
        with st.expander("🎨 Advanced Options"):
            cola, colb = st.columns(2)
            with cola:
                size = st.selectbox("Image Size", ["512x512", "768x768", "1024x1024"], index=1)
                quality = st.slider("Quality", 1, 10, 7)
            with colb:
                style = st.selectbox("Style", ["realistic", "fantasy", "abstract", "cartoon", "photographic"])
                num_images = st.slider("Number of images", 1, 4, 1)
    
    with col2:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific and descriptive
        - Mention art style if desired
        - Include colors, mood, composition
        - Specify important details
        """)
    
    if st.button("🎨 Generate Image", type="primary", use_container_width=True):
        if not prompt:
            st.warning("⚠️ Please enter a prompt")
        else:
            data = {
                'api_key': api_key,
                'prompt': prompt,
                'size': size,
                'quality': quality,
                'style': style,
                'num_images': num_images
            }
            
            result = make_api_request("generate", data=data)
            if result:
                if 'image' in result:
                    display_image(result['image'])
                elif 'images' in result:
                    display_multiple_images(result['images'], [f"Generated Image {i+1}" for i in range(len(result['images']))])
                elif 'results' in result:
                    display_multiple_images(result['results'], [f"Generated Image {i+1}" for i in range(len(result['results']))])

elif "Edit Image with Prompt" in mode:
    st.markdown("### ✏️ Edit Image with Prompt")
    
    uploaded_file = image_upload_section("Upload Image to Edit", "edit_image_file")
    
    prompt = st.text_area(
        "Edit instructions",
        placeholder="Change the background to a beach, make it sunset, add some palm trees...",
        height=100
    )
    
    if st.button("✨ Edit Image", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("⚠️ Please upload an image")
        elif not prompt:
            st.warning("⚠️ Please enter edit instructions")
        else:
            result = make_api_request(
                "edit", 
                data={'api_key': api_key, 'prompt': prompt}, 
                files={'file': uploaded_file}
            )
            if result and 'image' in result:
                display_image(result['image'])

elif "Virtual Try On" in mode:
    st.markdown("### 👗 Virtual Try-On")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_img = image_upload_section("Product Image", "virtual_tryon_product", "Upload the clothing/item image")
    
    with col2:
        person_img = image_upload_section("Person Image", "virtual_tryon_person", "Upload the person image")
    
    prompt = st.text_area(
        "Additional instructions (optional)",
        placeholder="Ensure the clothing fits naturally, adjust lighting to match...",
        height=80
    )
    
    if st.button("👔 Try On", type="primary", use_container_width=True):
        if not product_img or not person_img:
            st.warning("⚠️ Please upload both product and person images")
        else:
            files = {'product': product_img, 'person': person_img}
            data = {'api_key': api_key, 'prompt': prompt}
            
            result = make_api_request("virtual_try_on", data=data, files=files)
            if result and 'image' in result:
                display_image(result['image'])

elif "Create Ads" in mode:
    st.markdown("### 📢 Create Advertising Content")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_img = image_upload_section("Model Image", "create_ads_model", "Upload model or background image")
    
    with col2:
        product_img = image_upload_section("Product Image", "create_ads_product", "Upload product image")
    
    prompt = st.text_area(
        "Ad creation instructions",
        placeholder="Create a modern, vibrant ad for social media. Add catchy text and professional lighting...",
        height=100
    )
    
    # Ad-specific options
    with st.expander("📋 Ad Specifications"):
        ad_type = st.selectbox("Ad Type", ["Social Media", "E-commerce", "Billboard", "Print"])
        platform = st.selectbox("Platform", ["Instagram", "Facebook", "Twitter", "General"])
        style = st.selectbox("Style", ["Modern", "Luxury", "Minimalist", "Bold", "Elegant"])
    
    if st.button("🚀 Create Ads", type="primary", use_container_width=True):
        if not model_img or not product_img:
            st.warning("⚠️ Please upload both model and product images")
        else:
            files = {'model': model_img, 'product': product_img}
            data = {
                'api_key': api_key, 
                'prompt': prompt,
                'ad_type': ad_type,
                'platform': platform,
                'style': style
            }
            
            result = make_api_request("create_ads", data=data, files=files)
            if result:
                if 'results' in result:
                    display_multiple_images(result['results'], [f"Ad Variation {i+1}" for i in range(len(result['results']))])
                elif 'image' in result:
                    display_image(result['image'])

elif "Merge Images" in mode:
    st.markdown("### 🔗 Merge Multiple Images")
    
    uploaded_files = st.file_uploader(
        "Upload images to merge (2-5 images recommended)",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"],
        key="merge_images_files"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} images selected")
        cols = st.columns(min(5, len(uploaded_files)))
        for idx, file in enumerate(uploaded_files[:5]):
            with cols[idx]:
                st.image(file, caption=f"Image {idx+1}", use_column_width=True)
    
    prompt = st.text_area(
        "Merging instructions",
        placeholder="Blend these images together seamlessly. Create a composite that combines elements from all images...",
        height=100
    )
    
    if st.button("🔄 Merge Images", type="primary", use_container_width=True):
        if not uploaded_files or len(uploaded_files) < 2:
            st.warning("⚠️ Please upload at least 2 images to merge")
        else:
            files = [('files', file) for file in uploaded_files[:5]]
            data = {'api_key': api_key, 'prompt': prompt}
            
            result = make_api_request("merge_images", data=data, files=files)
            if result and 'image' in result:
                display_image(result['image'])

elif "Generate Scenes" in mode:
    st.markdown("### 🎭 Generate Scene Variations")
    
    scene_img = image_upload_section("Upload Base Scene", "generate_scenes_scene", "Upload the scene image to generate variations from")
    
    prompt = st.text_area(
        "Scene variation instructions",
        placeholder="Generate different seasons: winter, spring, summer, autumn. Change time of day and weather conditions...",
        height=100
    )
    
    num_variations = st.slider("Number of variations", 1, 8, 4)
    
    if st.button("🌅 Generate Scenes", type="primary", use_container_width=True):
        if not scene_img:
            st.warning("⚠️ Please upload a scene image")
        else:
            data = {
                'api_key': api_key, 
                'prompt': prompt,
                'num_variations': num_variations
            }
            
            result = make_api_request("generate_scenes", data=data, files={'scene': scene_img})
            if result:
                if 'results' in result:
                    display_multiple_images(result['results'], [f"Scene Variation {i+1}" for i in range(len(result['results']))])
                elif 'image' in result:
                    display_image(result['image'])

elif "Restore Old Image" in mode:
    st.markdown("### 🔧 Restore Old/Damaged Images")
    
    old_img = image_upload_section("Upload Old Image", "restore_old_image_file", "Upload the old or damaged image to restore")
    
    prompt = st.text_area(
        "Restoration instructions (optional)",
        placeholder="Fix scratches, remove noise, enhance colors, improve sharpness...",
        height=80
    )
    
    # Restoration options
    with st.expander("🔧 Restoration Settings"):
        col1, col2 = st.columns(2)
        with col1:
            enhancement = st.selectbox("Enhancement Level", ["Light", "Medium", "Strong", "Maximum"])
            fix_scratches = st.checkbox("Fix scratches", value=True)
        with col2:
            colorize = st.checkbox("Colorize black & white", value=False)
            enhance_details = st.checkbox("Enhance details", value=True)
    
    if st.button("🛠️ Restore Image", type="primary", use_container_width=True):
        if not old_img:
            st.warning("⚠️ Please upload an image to restore")
        else:
            data = {
                'api_key': api_key, 
                'prompt': prompt,
                'enhancement': enhancement,
                'fix_scratches': fix_scratches,
                'colorize': colorize,
                'enhance_details': enhance_details
            }
            
            result = make_api_request("restore_old_image", data=data, files={'file': old_img})
            if result and 'image' in result:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(old_img, caption="Original Image", use_column_width=True)
                with col2:
                    display_image(result['image'], "Restored Image")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🍌 Nano Banana Studio | AI-Powered Creative Platform"
    "</div>",
    unsafe_allow_html=True
)