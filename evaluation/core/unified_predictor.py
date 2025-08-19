#!/usr/bin/env python3
"""
Unified Predictor - OpenRouter based vision language model predictor
"""

import asyncio
import base64
import json
import time
import logging
import re
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image
import aiohttp
import yaml


class UnifiedPredictor:
    """
    Unified predictor using OpenRouter API for multiple vision language models
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 models_config_path: str = "config/models.yaml",
                 prompts_config_path: str = "config/prompts.yaml"):
        """
        Initialize unified predictor
        
        Args:
            model_name: Model name (e.g., "gemini-2.5-flash")
            api_key: OpenRouter API key
            models_config_path: Path to models configuration file
            prompts_config_path: Path to prompts configuration file
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Load configurations
        self.models_config = self._load_yaml(models_config_path)
        self.prompts_config = self._load_yaml(prompts_config_path)
        
        # Get model configuration
        self.model_config = self._get_model_config(model_name)
        
        # OpenRouter settings
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/user/KnowDR-REC",
            "X-Title": "KVQA Evaluation Framework"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_interval = 60.0 / 200  # 200 requests per minute
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Add provider and model_name attributes for compatibility
        self.provider = "openrouter"
        
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config {path}: {e}")
            return {}
    
    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        all_models = (self.models_config.get("default_models", []) + 
                     self.models_config.get("user_models", []))
        
        for model in all_models:
            if model.get("name") == model_name:
                return model
        
        # Fallback configuration
        return {
            "name": model_name,
            "openrouter_id": model_name,
            "max_tokens": 2048,
            "temperature": 0.1
        }
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    
    def _format_prompt(self, template: str, question: str, image: Image.Image) -> str:
        """Format prompt template with variables"""
        if not template:
            return question
            
        width, height = image.size
        return template.format(
            question=question,
            image_width=width,
            image_height=height
        )
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, messages: list) -> str:
        """Make API request to OpenRouter"""
        await self._rate_limit()
        
        payload = {
            "model": self.model_config.get("openrouter_id", self.model_name),
            "messages": messages,
            "max_tokens": self.model_config.get("max_tokens", 2048),
            "temperature": self.model_config.get("temperature", 0.1)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                self.stats["total_requests"] += 1
                
                if response.status == 200:
                    result = await response.json()
                    self.stats["successful_requests"] += 1
                    
                    # Update token usage if available
                    if "usage" in result:
                        self.stats["total_tokens"] += result["usage"].get("total_tokens", 0)
                    
                    return result["choices"][0]["message"]["content"]
                else:
                    self.stats["failed_requests"] += 1
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
    
    def _parse_response(self, response_text: str, sample_type: str = "positive") -> Dict[str, Any]:
        """Parse model response"""
        try:
            response_text = response_text.strip()
            
            if sample_type == "negative":
                # Simple yes/no parsing for negative samples
                response_lower = response_text.lower()
                if "yes" in response_lower:
                    return {
                        "target_description": "found_but_no_bbox",
                        "bounding_box": [0, 0, 100, 100],
                        "confidence": 0.9,
                        "success": True
                    }
                elif "no" in response_lower:
                    return {
                        "target_description": "not_found", 
                        "bounding_box": None,
                        "confidence": 0.9,
                        "success": True
                    }
                else:
                    return {
                        "target_description": "parsing_failed",
                        "bounding_box": None,
                        "confidence": 0.5,
                        "success": False
                    }
            
            # Parse positive sample response
            # Extract target description
            target_pattern = r'Target:\s*([^\n]+)'
            target_match = re.search(target_pattern, response_text, re.IGNORECASE)
            target_description = target_match.group(1).strip() if target_match else "unknown"
            
            # Extract bounding box
            bbox_pattern = r'Bounding Box:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
            bbox_match = re.search(bbox_pattern, response_text, re.IGNORECASE)
            
            if bbox_match:
                bbox = [float(coord) for coord in bbox_match.groups()]
                return {
                    "target_description": target_description,
                    "bounding_box": bbox,
                    "confidence": 0.8,
                    "success": True
                }
            
            # Fallback: try to extract any 4 numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', response_text)
            if len(numbers) >= 4:
                bbox = [float(n) for n in numbers[:4]]
                return {
                    "target_description": target_description,
                    "bounding_box": bbox, 
                    "confidence": 0.5,
                    "success": True
                }
            
            return {
                "target_description": target_description,
                "bounding_box": [0, 0, 100, 100],
                "confidence": 0.0,
                "success": False,
                "raw_response": response_text
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            return {
                "target_description": "parsing_failed",
                "bounding_box": [0, 0, 100, 100],
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "raw_response": response_text
            }
    
    async def predict_single(self, 
                           image: Image.Image,
                           question: str,
                           sample_type: str = "positive") -> Dict[str, Any]:
        """
        Predict single sample
        
        Args:
            image: PIL Image object
            question: Question text
            sample_type: "positive" or "negative"
            
        Returns:
            Prediction result dictionary
        """
        try:
            # Get appropriate prompt template
            if sample_type == "positive":
                template = self.prompts_config.get("positive_prompt", {}).get("template", "")
            else:
                template = self.prompts_config.get("negative_prompt", {}).get("template", "")
            
            # Format prompt
            prompt = self._format_prompt(template, question, image)
            
            # Convert image to base64
            image_b64 = self._image_to_base64(image)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_b64}}
                    ]
                }
            ]
            
            # Make API request
            response_text = await self._make_request(messages)
            
            # Parse response
            result = self._parse_response(response_text, sample_type)
            
            # Add metadata
            result.update({
                "model_used": self.model_name,
                "openrouter_id": self.model_config.get("openrouter_id"),
                "prompt_used": prompt[:100] + "..." if len(prompt) > 100 else prompt
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                "target_description": "prediction_failed",
                "bounding_box": [0, 0, 100, 100],
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "model_used": self.model_name
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        return {
            "model_name": self.model_name,
            "openrouter_id": self.model_config.get("openrouter_id"),
            "provider": "openrouter",
            **self.stats
        }
