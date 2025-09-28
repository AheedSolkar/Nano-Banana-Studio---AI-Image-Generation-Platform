# app.py
import streamlit as st
import requests
import base64
import io
from PIL import Image
import time
import os
from datetime import datetime
import uuid
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import re

# ===== FASTAPI BACKEND CODE =====
import base64 as base64_backend
import time as time_backend
import requests as requests_backend
import logging as logging_backend
from typing import List as List_backend, Optional as Optional_backend, Dict as Dict_backend, Any as Any_backend
import random as random_backend
import uuid as uuid_backend
from datetime import datetime as datetime_backend

# Enhanced prompt management system
class QualityLevel(Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    PROFESSIONAL = "professional"

class StylePreset(Enum):
    REALISTIC = "realistic"
    FANTASY = "fantasy"
    MINIMALIST = "minimalist"
    VINTAGE = "vintage"
    MODERN = "modern"
    CINEMATIC = "cinematic"

@dataclass
class PromptConfig:
    """Configuration for prompt generation"""
    quality: QualityLevel = QualityLevel.STANDARD
    style: Optional[StylePreset] = None
    include_technical: bool = True
    enhance_creativity: bool = False
    preserve_original: bool = False

class NanoBananaPrompts:
    """
    Advanced prompt management system with context-aware templates,
    parameter substitution, and quality controls.
    """
    
    def __init__(self):
        self.base_prompts = self._initialize_base_prompts()
        self.quality_modifiers = self._initialize_quality_modifiers()
        self.style_modifiers = self._initialize_style_modifiers()
        self.technical_directives = self._initialize_technical_directives()
    
    def _initialize_base_prompts(self) -> Dict[str, List[str]]:
        """Initialize the core prompt templates"""
        return {
            "generate_image": [
                "SYSTEM: Generate a high-quality, visually compelling image that precisely interprets the user's creative vision. "
                "Focus on artistic excellence, coherent lighting, balanced composition, and emotional impact. "
                "Ensure technical perfection: proper perspective, realistic textures, harmonious color palette, and natural shadows. "
                "Avoid any textual elements, watermarks, or digital artifacts. The output should feel like a professional artwork."
            ],
            "edit_image": [
                "SYSTEM: Execute intelligent image transformations guided by user intent while maintaining visual integrity. "
                "Preserve core subject identity, anatomical proportions, and essential compositional elements. "
                "Apply changes with artistic sensitivity: seamless blending, consistent lighting adaptation, and natural color grading. "
                "Avoid common pitfalls: over-saturation, artificial-looking edits, style inconsistencies, or loss of important details."
            ],
            "virtual_try_on": [
                "SYSTEM: Perform photorealistic virtual try-on with meticulous attention to physical realism. "
                "Ensure perfect fabric simulation: natural draping, realistic folds, and material-appropriate behavior. "
                "Maintain anatomical accuracy, proper garment fit, and biomechanically plausible poses. "
                "Achieve seamless integration: matched lighting conditions, consistent shadows, and color harmony between person and product. "
                "No artificial stretching, distortion, or addition of non-existent elements."
            ],
            "create_ads": [
                "SYSTEM: Create professional-grade advertising imagery that balances artistic appeal with commercial effectiveness. "
                "Generate distinct creative concepts that showcase the product while maintaining brand-appropriate aesthetics. "
                "Ensure product visibility and appeal: clear focal points, complementary backgrounds, and professional lighting. "
                "Maintain commercial viability: appropriate for target audience, emotionally engaging, and technically polished. "
                "Avoid textual elements while creating compositions that naturally accommodate marketing copy."
            ],
            "merge_images": [
                "SYSTEM: Synthesize multiple images into a single, coherent visual narrative with artistic and technical excellence. "
                "Create seamless integration: unified color grading, consistent lighting direction, and harmonious perspective. "
                "Maintain visual logic: plausible spatial relationships, appropriate scale, and natural element interactions. "
                "Remove redundancies and conflicts while preserving the essential character of each source image. "
                "The final composition should feel intentional and professionally crafted, not like a collage."
            ],
            "generate_scenes": [
                "SYSTEM: Extend and reinterpret scenes with creative intelligence while maintaining environmental plausibility. "
                "Preserve architectural consistency, logical spatial relationships, and physically accurate lighting. "
                "Generate variations that explore different moods, times, or conditions while respecting the original scene's character. "
                "Ensure material authenticity: realistic textures, appropriate surface properties, and believable environmental effects. "
                "Create outputs that feel like natural extensions or alternative realities of the original scene."
            ],
            "restore_old_image": [
                "SYSTEM: Perform expert-level image restoration with historical sensitivity and technical precision. "
                "Remove age-related damage: scratches, dust, stains, fading, and noise while preserving authentic character. "
                "Maintain historical integrity: avoid modern stylistic influences and preserve period-appropriate details. "
                "Enhance clarity through intelligent sharpening, contrast adjustment, and color recovery where appropriate. "
                "The restored image should feel authentic to its era while achieving optimal visual clarity and impact."
            ]
        }
    
    def _initialize_quality_modifiers(self) -> Dict[QualityLevel, str]:
        """Initialize quality level modifiers"""
        return {
            QualityLevel.STANDARD: "Focus on balanced quality with efficient processing.",
            QualityLevel.PREMIUM: "Prioritize high-fidelity details and enhanced visual appeal.",
            QualityLevel.PROFESSIONAL: "Achieve maximum artistic and technical excellence with meticulous attention to every detail."
        }
    
    def _initialize_style_modifiers(self) -> Dict[StylePreset, str]:
        """Initialize style-specific modifiers"""
        return {
            StylePreset.REALISTIC: "Maintain photorealistic quality with accurate lighting and natural details.",
            StylePreset.FANTASY: "Embrace creative interpretation with imaginative elements and enhanced visual drama.",
            StylePreset.MINIMALIST: "Focus on clean composition, negative space, and essential elements only.",
            StylePreset.VINTAGE: "Apply period-appropriate aesthetics with authentic color grading and texture.",
            StylePreset.MODERN: "Use contemporary visual language with sharp details and current aesthetic trends.",
            StylePreset.CINEMATIC: "Create dramatic, film-like quality with enhanced contrast and artistic lighting."
        }
    
    def _initialize_technical_directives(self) -> Dict[str, str]:
        """Initialize technical quality directives"""
        return {
            "lighting": "Ensure professional lighting: natural shadows, appropriate highlights, and consistent light sources.",
            "composition": "Apply strong compositional principles: rule of thirds, balanced elements, and clear focal points.",
            "colors": "Maintain harmonious color palette with appropriate saturation and contrast levels.",
            "details": "Preserve important details while avoiding unnecessary noise or artifacts.",
            "coherence": "Ensure overall visual coherence and logical element relationships."
        }
    
    def get_enhanced_prompt(self, 
                          mode: str, 
                          user_prompt: Optional[str] = None,
                          config: Optional[PromptConfig] = None) -> List[str]:
        """
        Generate enhanced prompts based on mode, user input, and configuration.
        
        Args:
            mode: The operation mode (e.g., "generate_image", "edit_image")
            user_prompt: Optional user-provided prompt for context awareness
            config: Optional configuration for quality, style, etc.
        
        Returns:
            List of enhanced prompt strings
        """
        if mode not in self.base_prompts:
            raise ValueError(f"Unknown mode: {mode}")
        
        base_prompt = self.base_prompts[mode][0]
        enhanced_prompts = [base_prompt]
        
        # Apply configuration enhancements
        if config:
            enhanced_prompts = self._apply_configuration(enhanced_prompts, config)
        
        # Enhance based on user prompt analysis
        if user_prompt:
            enhanced_prompts = self._enhance_with_user_context(enhanced_prompts, user_prompt, mode)
        
        return enhanced_prompts
    
    def _apply_configuration(self, prompts: List[str], config: PromptConfig) -> List[str]:
        """Apply configuration-based enhancements to prompts"""
        enhanced = []
        
        for prompt in prompts:
            # Add quality modifier
            if config.quality in self.quality_modifiers:
                prompt += " " + self.quality_modifiers[config.quality]
            
            # Add style modifier
            if config.style and config.style in self.style_modifiers:
                prompt += " " + self.style_modifiers[config.style]
            
            # Add technical directives
            if config.include_technical:
                prompt += " " + self.technical_directives["lighting"]
                prompt += " " + self.technical_directives["composition"]
            
            # Enhance creativity if requested
            if config.enhance_creativity:
                prompt += " Prioritize creative interpretation and artistic expression while maintaining technical quality."
            
            # Emphasize preservation if needed
            if config.preserve_original:
                prompt += " Maximize preservation of original characteristics and minimize transformative changes."
            
            enhanced.append(prompt)
        
        return enhanced
    
    def _enhance_with_user_context(self, 
                                 prompts: List[str], 
                                 user_prompt: str, 
                                 mode: str) -> List[str]:
        """Enhance prompts based on analysis of user input"""
        enhanced = []
        
        # Analyze user prompt for key characteristics
        characteristics = self._analyze_user_prompt(user_prompt)
        
        for prompt in prompts:
            enhanced_prompt = prompt
            
            # Add context-specific enhancements based on analysis
            if characteristics.get("has_emotion"):
                enhanced_prompt += " Pay special attention to emotional tone and mood expression."
            
            if characteristics.get("has_detailed_description"):
                enhanced_prompt += " Ensure all described elements are accurately represented and properly integrated."
            
            if characteristics.get("has_style_reference"):
                enhanced_prompt += " Faithfully interpret the referenced artistic style while maintaining originality."
            
            if characteristics.get("is_complex"):
                enhanced_prompt += " Handle complex composition with careful attention to element relationships and hierarchy."
            
            # Mode-specific context enhancements
            if mode == "edit_image" and characteristics.get("has_transformation"):
                enhanced_prompt += " Execute transformations with seamless integration and natural-looking results."
            
            if mode == "generate_scenes" and characteristics.get("has_environment"):
                enhanced_prompt += " Maintain environmental plausibility and spatial consistency throughout variations."
            
            enhanced.append(enhanced_prompt)
        
        return enhanced
    
    def _analyze_user_prompt(self, user_prompt: str) -> Dict[str, bool]:
        """Analyze user prompt for key characteristics"""
        prompt_lower = user_prompt.lower()
        
        return {
            "has_emotion": any(word in prompt_lower for word in [
                'emotional', 'mood', 'atmosphere', 'feeling', 'sentimental', 'dramatic'
            ]),
            "has_detailed_description": len(user_prompt.split()) > 15,
            "has_style_reference": any(word in prompt_lower for word in [
                'style', 'in the style of', 'like', 'similar to', 'inspired by'
            ]),
            "has_transformation": any(word in prompt_lower for word in [
                'change', 'transform', 'modify', 'alter', 'replace'
            ]),
            "has_environment": any(word in prompt_lower for word in [
                'environment', 'landscape', 'scene', 'setting', 'background'
            ]),
            "is_complex": len(user_prompt.split()) > 25 or any(indicator in prompt_lower 
                for indicator in ['multiple', 'various', 'different', 'complex'])
        }
    
    def get_prompt_variations(self, 
                            mode: str, 
                            count: int = 3,
                            base_config: Optional[PromptConfig] = None) -> List[str]:
        """
        Generate multiple prompt variations for diversity in outputs.
        
        Args:
            mode: The operation mode
            count: Number of variations to generate
            base_config: Base configuration for all variations
        
        Returns:
            List of varied prompt strings
        """
        variations = []
        base_prompt = self.get_enhanced_prompt(mode, config=base_config)[0]
        
        variation_modifiers = [
            "Focus on creating a warm, inviting atmosphere with soft lighting.",
            "Emphasize dynamic composition and visual energy.",
            "Prioritize minimalist elegance and clean visual hierarchy.",
            "Create a dramatic, high-contrast visual presentation.",
            "Focus on rich textures and detailed material representation.",
            "Emphasize natural, organic forms and flowing compositions.",
            "Create a sleek, modern aesthetic with sharp details."
        ]
        
        for i in range(min(count, len(variation_modifiers))):
            variation = base_prompt + " " + variation_modifiers[i]
            variations.append(variation)
        
        return variations
    
    def validate_user_prompt(self, user_prompt: str, mode: str) -> Dict[str, Any]:
        """
        Validate and provide feedback on user prompts.
        
        Returns:
            Dictionary with validation results and suggestions
        """
        analysis = self._analyze_user_prompt(user_prompt)
        word_count = len(user_prompt.split())
        
        feedback = {
            "is_valid": True,
            "score": 0,
            "suggestions": [],
            "warnings": []
        }
        
        # Score calculation
        if word_count < 5:
            feedback["score"] = 1
            feedback["suggestions"].append("Consider adding more descriptive details for better results.")
        elif word_count < 15:
            feedback["score"] = 3
        else:
            feedback["score"] = 5
        
        # Mode-specific validation
        if mode == "edit_image" and not analysis["has_transformation"]:
            feedback["warnings"].append("For image editing, specify what changes you want to make.")
        
        if mode == "generate_scenes" and not analysis["has_environment"]:
            feedback["suggestions"].append("Consider describing the environment or setting for better scene generation.")
        
        return feedback

# Configuration
class Config:
    """Configuration management"""
    def __init__(self):
        self.API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-exp:generateContent"
        self.MAX_RETRIES = 3
        self.BACKOFF_BASE = 1.5
        self.MAX_AD_VARIATIONS = 4
        self.MAX_SCENE_VARIATIONS = 6
        self.REQUEST_TIMEOUT = 60
        self.MAX_FILE_SIZE = 10_485_760  # 10MB

config = Config()
prompt_manager = NanoBananaPrompts()

def call_nano_banana(
    api_key: str, 
    prompt: str, 
    images: List_backend[dict] = None
) -> tuple[Optional_backend[str], Optional_backend[str]]:
    """
    API call to Gemini with error handling
    """
    parts = [{'text': prompt}]
    if images:
        for img in images:
            parts.append({'inlineData': {'data': img['data'], 'mimeType': img['mime']}})

    payload = {'contents': [{'parts': parts}]}
    
    try:
        response = requests_backend.post(
            f"{config.API_URL}?key={api_key}", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=config.REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            return None, f"API Error {response.status_code}: {response.text}"

        data = response.json()
        candidates = data.get('candidates', [])
        if not candidates:
            return None, "No candidates in response"
            
        content = candidates[0].get('content', {})
        parts_out = content.get('parts', [])
        
        for part in parts_out:
            if 'inlineData' in part:
                inline_data = part['inlineData']
                return inline_data['data'], inline_data.get('mimeType', 'image/png')
        
        return None, "No image data in response"
        
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# ===== STREAMLIT FRONTEND CODE =====

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
    api_key = st.text_input(
        "Gemini API Key", 
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Your Google Gemini API key for image generation"
    )
    
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
    if not api_key:
        st.error("❌ Please enter Gemini API Key")
        return False
    return True

def process_image_generation(endpoint, data=None, files=None, success_msg="Operation completed successfully!"):
    """Generic image processing handler"""
    if not validate_api_config():
        return None
    
    try:
        with st.spinner("🔄 Processing your request... This may take a few moments."):
            progress_bar = st.progress(0)
            
            # Simulate progress for better UX
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            # For this demo, we'll simulate API calls
            # In a real deployment, you would call your FastAPI backend
            time.sleep(2)  # Simulate processing time
            
            # For demo purposes, return a placeholder
            # In production, you would make actual API calls
            progress_bar.empty()
            
            st.warning("🚧 This is a demo version. In production, this would connect to the FastAPI backend.")
            return None
                
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
    
    return None

def display_image(img_b64, caption="Generated Image", use_column_width=True):
    """Display base64 image and store in session"""
    try:
        # For demo, create a placeholder image
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (512, 512), color='lightblue')
        d = ImageDraw.Draw(img)
        d.text((100, 256), f"Demo: {caption}", fill='black')
        
        # Convert to base64 for consistency
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
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
            display_image(img_data, caption)

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
                quality = st.selectbox("Quality", ["standard", "premium", "professional"], index=1)
                style = st.selectbox("Style", ["realistic", "fantasy", "minimalist", "vintage", "modern", "cinematic"])
            with colb:
                enhance_creativity = st.checkbox("Enhance Creativity")
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
            # Generate enhanced prompt
            config = PromptConfig(
                quality=QualityLevel(quality),
                style=StylePreset(style),
                enhance_creativity=enhance_creativity
            )
            enhanced_prompts = prompt_manager.get_enhanced_prompt("generate_image", prompt, config)
            system_prompt = enhanced_prompts[0]
            full_prompt = f"{system_prompt} USER PROMPT: {prompt}"
            
            st.info(f"Enhanced Prompt: {full_prompt[:200]}...")
            
            # Simulate API call
            if num_images == 1:
                display_image(None, "Generated Image")
            else:
                for i in range(num_images):
                    display_image(None, f"Generated Image {i+1}")

elif "Edit Image with Prompt" in mode:
    st.markdown("### ✏️ Edit Image with Prompt")
    
    uploaded_file = image_upload_section("Upload Image to Edit", "edit_image_file")
    
    prompt = st.text_area(
        "Edit instructions",
        placeholder="Change the background to a beach, make it sunset, add some palm trees...",
        height=100
    )
    
    preserve_original = st.checkbox("Preserve Original Characteristics", value=True)
    
    if st.button("✨ Edit Image", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("⚠️ Please upload an image")
        elif not prompt:
            st.warning("⚠️ Please enter edit instructions")
        else:
            # Generate enhanced prompt
            config = PromptConfig(preserve_original=preserve_original)
            enhanced_prompts = prompt_manager.get_enhanced_prompt("edit_image", prompt, config)
            system_prompt = enhanced_prompts[0]
            full_prompt = f"{system_prompt} EDIT INSTRUCTIONS: {prompt}"
            
            st.info(f"Enhanced Prompt: {full_prompt[:200]}...")
            
            # Display before and after
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", use_column_width=True)
            with col2:
                display_image(None, "Edited Image")

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
            # Generate enhanced prompt
            enhanced_prompts = prompt_manager.get_enhanced_prompt("virtual_try_on", prompt)
            system_prompt = enhanced_prompts[0]
            full_prompt = system_prompt
            if prompt:
                full_prompt += f" ADDITIONAL INSTRUCTIONS: {prompt}"
            
            st.info(f"Enhanced Prompt: {full_prompt[:200]}...")
            
            # Display input images and result
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(product_img, caption="Product", use_column_width=True)
            with col2:
                st.image(person_img, caption="Person", use_column_width=True)
            with col3:
                display_image(None, "Virtual Try-On Result")

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
        variations = st.slider("Number of variations", 1, 4, 3)
        style = st.selectbox("Style", ["modern", "luxury", "minimalist", "bold", "elegant"])
    
    if st.button("🚀 Create Ads", type="primary", use_container_width=True):
        if not model_img or not product_img:
            st.warning("⚠️ Please upload both model and product images")
        else:
            # Generate enhanced prompt
            enhanced_prompts = prompt_manager.get_enhanced_prompt(
                "create_ads", 
                prompt, 
                PromptConfig(style=StylePreset(style))
            )
            system_prompt = enhanced_prompts[0]
            
            st.info(f"Enhanced Prompt: {system_prompt[:200]}...")
            
            # Generate variations
            for i in range(variations):
                display_image(None, f"Ad Variation {i+1}")

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
            # Generate enhanced prompt
            enhanced_prompts = prompt_manager.get_enhanced_prompt("merge_images", prompt)
            system_prompt = enhanced_prompts[0]
            full_prompt = system_prompt
            if prompt:
                full_prompt += f" MERGE INSTRUCTIONS: {prompt}"
            
            st.info(f"Enhanced Prompt: {full_prompt[:200]}...")
            
            display_image(None, "Merged Image")

elif "Generate Scenes" in mode:
    st.markdown("### 🎭 Generate Scene Variations")
    
    scene_img = image_upload_section("Upload Base Scene", "generate_scenes_scene", "Upload the scene image to generate variations from")
    
    prompt = st.text_area(
        "Scene variation instructions",
        placeholder="Generate different seasons: winter, spring, summer, autumn. Change time of day and weather conditions...",
        height=100
    )
    
    num_variations = st.slider("Number of variations", 1, 6, 4)
    
    if st.button("🌅 Generate Scenes", type="primary", use_container_width=True):
        if not scene_img:
            st.warning("⚠️ Please upload a scene image")
        else:
            # Generate enhanced prompt
            enhanced_prompts = prompt_manager.get_enhanced_prompt("generate_scenes", prompt)
            system_prompt = enhanced_prompts[0]
            
            st.info(f"Enhanced Prompt: {system_prompt[:200]}...")
            
            # Generate variations
            for i in range(num_variations):
                display_image(None, f"Scene Variation {i+1}")

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
        enhancement_level = st.selectbox("Enhancement Level", ["light", "medium", "strong"])
        colorize = st.checkbox("Colorize black & white", value=False)
    
    if st.button("🛠️ Restore Image", type="primary", use_container_width=True):
        if not old_img:
            st.warning("⚠️ Please upload an image to restore")
        else:
            # Generate enhanced prompt
            config = PromptConfig()
            enhanced_prompts = prompt_manager.get_enhanced_prompt("restore_old_image", prompt, config)
            system_prompt = enhanced_prompts[0]
            full_prompt = f"{system_prompt} ENHANCEMENT LEVEL: {enhancement_level.upper()}."
            if prompt:
                full_prompt += f" SPECIFIC REQUESTS: {prompt}"
            
            st.info(f"Enhanced Prompt: {full_prompt[:200]}...")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(old_img, caption="Original Image", use_column_width=True)
            with col2:
                display_image(None, "Restored Image")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🍌 Nano Banana Studio | AI-Powered Creative Platform | Demo Version"
    "</div>",
    unsafe_allow_html=True
)
