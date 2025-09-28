# 🍌 Nano Banana Studio - AI Image Generation Platform

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

A sophisticated AI-powered image generation and editing platform built with FastAPI and Streamlit. Transform your creative ideas into stunning visuals with multiple advanced modes and professional-grade image processing.

## 🚀 Features

### 🎨 **Creative Modes**
- **📝 Text-to-Image Generation** - Create images from text descriptions
- **✏️ Image Editing** - Transform existing images with AI-powered edits
- **👗 Virtual Try-On** - Realistic clothing try-on experience
- **📢 Ad Creation** - Generate professional advertising content
- **🔗 Image Merging** - Seamlessly blend multiple images
- **🎭 Scene Generation** - Create variations of existing scenes
- **🔧 Image Restoration** - Restore and enhance old/damaged photos

### ⚡ **Advanced Capabilities**
- **Quality Controls** - Standard, Premium, and Professional quality levels
- **Style Presets** - Cinematic, Vintage, Modern, Fantasy, and more
- **Batch Processing** - Generate multiple variations simultaneously
- **Real-time Previews** - Instant image previews and downloads
- **Smart Prompting** - Context-aware prompt engineering
- **Rate Limiting** - Fair usage and API protection

## 🛠️ Tech Stack

**Backend:**
- FastAPI - High-performance API framework
- Pydantic - Data validation and serialization
- Uvicorn - ASGI server
- CacheTools - Response caching
- Requests - HTTP client

**Frontend:**
- Streamlit - Interactive web application
- Base64 - Image encoding/decoding
- PIL - Image processing

**AI/ML:**
- Google Gemini API - Advanced image generation
- Custom Prompt Engineering - Enhanced output quality

## 📋 Prerequisites

- Python 3.9 or higher
- Google Gemini API key
- 4GB+ RAM recommended

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/nano-banana-studio.git
cd nano-banana-studio
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
API_BASE=http://localhost:8000
MAX_AD_VARIATIONS=4
MAX_SCENE_VARIATIONS=6
MAX_RETRIES=3
REQUEST_TIMEOUT=60
MAX_FILE_SIZE=10485760
CACHE_TTL=300
```

## 🎯 Quick Start

### 1. Start the Backend Server
```bash
python backend/main.py
```
The API will be available at `http://localhost:8000`

### 2. Start the Frontend (New Terminal)
```bash
streamlit run frontend/app.py
```
Access the application at `http://localhost:8501`

### 3. Configure API Settings
1. Open the Streamlit app in your browser
2. Navigate to the sidebar settings
3. Enter your Gemini API key
4. Verify backend URL (`http://localhost:8000`)
5. Click "Check API Status" to confirm connection


## 🔧 API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/generate` | POST | Generate image from text | `prompt`, `quality`, `style` |
| `/edit` | POST | Edit existing image | `prompt`, `file`, `preserve_original` |
| `/virtual_try_on` | POST | Virtual clothing try-on | `product`, `person`, `prompt` |
| `/create_ads` | POST | Generate ad variations | `model`, `product`, `variations`, `style` |
| `/merge_images` | POST | Merge multiple images | `files[]`, `prompt` |
| `/generate_scenes` | POST | Generate scene variations | `scene`, `variations`, `prompt` |
| `/restore_old_image` | POST | Restore old images | `file`, `prompt`, `enhancement_level` |
| `/health` | GET | API health check | - |
| `/validate_prompt` | POST | Validate user prompts | `prompt`, `mode` |

## 🎨 Usage Examples

### Basic Image Generation
```python
# Using the API directly
import requests

response = requests.post("http://localhost:8000/generate", 
    data={
        "api_key": "your_gemini_key",
        "prompt": "A majestic dragon flying over a medieval castle at sunset",
        "quality": "premium",
        "style": "fantasy"
    }
)
```

### Image Editing
```python
with open("image.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/edit",
        data={
            "api_key": "your_gemini_key",
            "prompt": "Change background to beach at sunset"
        },
        files={"file": f}
    )
```

## ⚙️ Configuration

### Quality Levels
- **Standard**: Balanced quality with efficient processing
- **Premium**: High-fidelity details and enhanced visual appeal  
- **Professional**: Maximum artistic and technical excellence

### Style Presets
- `realistic` - Photorealistic quality
- `fantasy` - Creative and imaginative
- `cinematic` - Dramatic, film-like quality
- `vintage` - Period-appropriate aesthetics
- `modern` - Contemporary visual language
- `minimalist` - Clean and essential elements

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Manual Docker Build
```bash
docker build -t nano-banana-studio .
docker run -p 8000:8000 -p 8501:8501 nano-banana-studio
```

## 🧪 Testing

Run the test suite:
```bash
# Backend tests
python -m pytest tests/test_backend.py -v

# Frontend tests  
python -m pytest tests/test_frontend.py -v

# All tests
python -m pytest tests/ -v
```

## 📊 Performance

- **Response Time**: 2-15 seconds depending on complexity
- **Concurrent Users**: Supports 50+ simultaneous users
- **Cache Efficiency**: 80%+ cache hit rate for repeated requests
- **File Handling**: Supports up to 10MB images
- **Rate Limiting**: 100 requests per hour per API key

## 🔒 Security Features

- API key validation and rate limiting
- File type and size validation
- CORS protection
- Request sanitization
- Secure file upload handling
- Error message sanitization


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 **Email**: aheed.study@gmail.com

## 🙏 Acknowledgments

- Google Gemini for advanced AI capabilities
- FastAPI and Streamlit teams for excellent frameworks

---

<div align="center">

**🍌 Transform your creativity with Nano Banana Studio**

*Professional AI image generation made accessible*

</div>



**⭐ Star us on GitHub if you find this project helpful!**

click : https://nano-banana-studio---ai-image-generation-platform-lvwvmoevuqfq.streamlit.app/
