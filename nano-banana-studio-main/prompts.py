# nano_banana_prompts.py
"""
Enhanced prompt management system for Nano Banana Studio.
Features context-aware templates, parameterized prompts, quality controls,
and adaptive prompting based on user input and image characteristics.
"""

from typing import Dict, List, Optional, Any
import re
from dataclasses import dataclass
from enum import Enum

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

# Legacy interface for backward compatibility
PROMPTS = {
    "generate_image": [
        "SYSTEM: Generate a high-quality image based on the appended user prompt. Maintain clarity, coherent lighting, clean composition, and omit all textual overlays or watermarks."
    ],
    "edit_image": [
        "SYSTEM: Apply non-destructive visual transformations guided by the appended user prompt while preserving subject identity, proportions, and core composition. Avoid artifacts, over-saturation, or unintended style drift."
    ],
    "virtual_try_on": [
        "SYSTEM: Perform realistic virtual try-on by blending the product image onto the person image. Maintain anatomical correctness, natural fabric behavior, consistent lighting, and seamless color integration. No distortions or added accessories."
    ],
    "create_ads": [
        "SYSTEM: Produce professional advertisement imagery combining the model and product. Each generation should feel like a distinct ad concept while keeping the product clearly legible, composition balanced, and free of textual elements or logos."
    ],
    "merge_images": [
        "SYSTEM: Merge all provided images into a single coherent output guided by the user prompt. Unify perspective, color temperature, exposure, and shadow logic; remove redundancies; avoid frames, borders, or extraneous artifacts."
    ],
    "generate_scenes": [
        "SYSTEM: Generate extended or reinterpreted scene outputs derived from the uploaded image and optional user prompt. Preserve spatial coherence, plausible lighting, and material consistency while allowing creative environmental variation."
    ],
    "restore_old_image": [
        "SYSTEM: Restore the uploaded aged or damaged image. Remove scratches, noise, stains, and fading while preserving authentic detail, texture, and historical integrity. No stylistic modernization beyond faithful clarity recovery."
    ]
}

# Global instance for easy access
prompt_manager = NanoBananaPrompts()

def get_enhanced_prompt(mode: str, user_prompt: str = "", **kwargs) -> List[str]:
    """Convenience function to get enhanced prompts"""
    config = PromptConfig(**kwargs) if kwargs else None
    return prompt_manager.get_enhanced_prompt(mode, user_prompt, config)

if __name__ == "__main__":
    from pprint import pprint
    
    # Demo the enhanced prompt system
    manager = NanoBananaPrompts()
    
    print("=== Enhanced Prompt System Demo ===")
    
    # Test enhanced prompt generation
    config = PromptConfig(
        quality=QualityLevel.PROFESSIONAL,
        style=StylePreset.CINEMATIC,
        include_technical=True,
        enhance_creativity=True
    )
    
    enhanced = manager.get_enhanced_prompt(
        "generate_image",
        "A mysterious forest at twilight with glowing mushrooms and ancient ruins",
        config
    )
    
    print("\nEnhanced Prompt:")
    print(enhanced[0])
    
    # Test prompt validation
    validation = manager.validate_user_prompt(
        "Make it brighter",
        "edit_image"
    )
    
    print(f"\nPrompt Validation: {validation}")
    
    # Test variations
    variations = manager.get_prompt_variations("create_ads", 3)
    print(f"\nPrompt Variations ({len(variations)}):")
    for i, var in enumerate(variations, 1):
        print(f"{i}. {var[:100]}...")