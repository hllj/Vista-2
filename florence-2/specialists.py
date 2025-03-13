import os
from typing import List, Dict, Any, Tuple
import json
import google.generativeai as genai
from PIL import Image
import numpy as np
import re
from config import *

class SpecialistBase:
    def __init__(self):
        # Initialize Gemini API
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError

class TextSpecialist(SpecialistBase):
    def __init__(self):
        super().__init__()
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Generate text annotations for an image at different granularities."""
        image = Image.open(image_path)
        
        # Convert image to format expected by genai
        image_data = self._prepare_image(image)
        
        results = {}
        for text_type, settings in TEXT_TYPES.items():
            prompt = self._get_text_prompt(text_type, settings)
            response = self.model.generate_content([prompt, image_data])
            results[text_type] = response.text
            
        return results
    
    def _prepare_image(self, image: Image.Image) -> Dict[str, Any]:
        """Prepare image for Gemini API."""
        # Convert PIL Image to bytes
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return {"mime_type": "image/png", "data": image_bytes}
    
    def _get_text_prompt(self, text_type: str, settings: Dict[str, Any]) -> str:
        """Get prompt for text generation based on type."""
        if text_type == "brief":
            return f"""Provide a brief caption for this image in at most {settings['max_length']} characters.
            Focus on the main subjects and actions visible in the image.
            Use clear, concise language without unnecessary details."""
            
        elif text_type == "detailed":
            return f"""Describe this image in detail using at most {settings['max_length']} characters.
            Include information about:
            - Main subjects and their attributes
            - Actions or events taking place
            - Spatial relationships between objects
            - Notable visual characteristics
            Use descriptive language that captures the essence of the scene."""
            
        elif text_type == "more_detailed":
            return f"""Provide a comprehensive description of this image in at most {settings['max_length']} characters.
            Include detailed information about:
            - All visible subjects and their specific attributes (color, size, position)
            - Actions, expressions, and interactions
            - Environmental context and setting
            - Visual composition and notable elements
            - Implied narrative or context if evident
            Use rich, descriptive language that thoroughly captures all aspects of the image."""
            
        else:
            return f"Describe this image {text_type}ly in at most {settings['max_length']} characters."

class RegionSpecialist(SpecialistBase):
    def __init__(self):
        super().__init__()
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Detect and annotate regions in the image."""
        image = Image.open(image_path)
        
        # Convert image to format expected by genai
        image_data = self._prepare_image(image)
        
        # Ask Gemini to identify objects and their locations
        prompt = """Analyze this image and identify distinct objects with their locations.

For each object detected, provide:
1. Object name/category
2. Bounding box coordinates in normalized format [x1, y1, x2, y2] where:
   - x1, y1: top-left corner coordinates (values between 0-1)
   - x2, y2: bottom-right corner coordinates (values between 0-1)
3. Confidence score (between 0-1) indicating detection certainty

Format your response as a valid JSON object with this structure:
{
  "objects": [
    {
      "name": "object_name",
      "box": [x1, y1, x2, y2],
      "confidence": 0.95
    },
    ...
  ]
}

Be precise with coordinates and ensure the JSON is properly formatted."""
        
        response = self.model.generate_content([prompt, image_data])
        
        # Process and filter results
        results = self._process_response(response.text)
        return self._filter_results(results)
    
    def _prepare_image(self, image: Image.Image) -> Dict[str, Any]:
        """Prepare image for Gemini API."""
        # Convert PIL Image to bytes
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return {"mime_type": "image/png", "data": image_bytes}
    
    def _process_response(self, response: str) -> Dict[str, Any]:
        """Process the JSON response from Gemini."""
        # Extract JSON from response
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without markdown formatting
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback to using the entire response
                json_str = response
        
        try:
            data = json.loads(json_str)
            # Convert to our internal format
            results = {}
            for i, obj in enumerate(data.get("objects", [])):
                results[f"object_{i}"] = {
                    "name": obj.get("name", "unknown"),
                    "box": obj.get("box", [0, 0, 1, 1]),
                    "confidence": obj.get("confidence", 0.5)
                }
            return results
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty results
            print(f"Failed to parse JSON response: {response}")
            return {}
    
    def _filter_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter results based on confidence and NMS."""
        # Remove low confidence detections
        filtered = {k: v for k, v in results.items() 
                   if v.get('confidence', 0) > BOX_CONFIDENCE_THRESHOLD}
        
        if not filtered:
            return {}
            
        # Apply NMS
        boxes = np.array([region['box'] for region in filtered.values()])
        scores = np.array([region['confidence'] for region in filtered.values()])
        
        # Get indices of boxes to keep after NMS
        keep_indices = self._non_max_suppression(boxes, scores, NMS_THRESHOLD)
        
        # Create filtered result
        keys = list(filtered.keys())
        filtered_results = {
            keys[i]: filtered[keys[i]]
            for i in keep_indices if i < len(keys)
        }
        
        return filtered_results
        
    def _non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, 
                           threshold: float) -> List[int]:
        """Implement Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
            
        return keep

class TextPhraseRegionSpecialist(SpecialistBase):
    def __init__(self):
        super().__init__()
        self.text_specialist = TextSpecialist()
        self.region_specialist = RegionSpecialist()
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Generate text-phrase-region triplets."""
        # Get text annotations
        text_results = self.text_specialist.process_image(image_path)
        
        # Get region annotations
        region_results = self.region_specialist.process_image(image_path)
        
        # Link phrases to regions
        triplets = self._create_triplets(image_path, text_results, region_results)
        
        return triplets
    
    def _prepare_image(self, image: Image.Image) -> Dict[str, Any]:
        """Prepare image for Gemini API."""
        # Convert PIL Image to bytes
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return {"mime_type": "image/png", "data": image_bytes}
    
    def _create_triplets(self, image_path: str, text_results: Dict[str, Any], 
                        region_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create text-phrase-region triplets by linking components."""
        if not text_results or not region_results:
            return []
            
        image = Image.open(image_path)
        image_data = self._prepare_image(image)
        
        # Combine all text descriptions
        all_text = ""
        for text_type in ["brief", "detailed", "more_detailed"]:
            if text_type in text_results:
                all_text += f"{text_type.upper()}: {text_results[text_type]}\n\n"
        
        # Format region information
        regions_info = []
        for region_id, region in region_results.items():
            regions_info.append(f"Region {region_id}: {region['name']} at box {region['box']}")
        regions_text = "\n".join(regions_info)
        
        # Create prompt for Gemini to link phrases to regions
        prompt = f"""Given an image with the following descriptions and detected regions, 
create text-phrase-region triplets by linking specific phrases from the descriptions to the regions.

IMAGE DESCRIPTIONS:
{all_text}

DETECTED REGIONS:
{regions_text}

For each region, identify the most relevant noun phrases from the descriptions that refer to it.
Format your response as a valid JSON array with this structure:
[
  {{
    "region_id": "object_0",
    "phrase": "a red car",
    "text_source": "brief",
    "confidence": 0.95
  }},
  ...
]

Ensure each triplet has:
1. region_id: matching one of the provided region IDs
2. phrase: the exact noun phrase from the description that refers to the region
3. text_source: which description type the phrase came from (brief, detailed, or more_detailed)
4. confidence: a score (0-1) indicating how confident you are about this match

Only include phrases that clearly refer to specific regions. Ensure the JSON is properly formatted."""
        
        response = self.model.generate_content([prompt, image_data])
        
        # Process the response
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown formatting
                json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback to using the entire response
                    json_str = response.text
            
            triplets = json.loads(json_str)
            return triplets
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Failed to parse triplets response: {e}")
            return [] 