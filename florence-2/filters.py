import spacy
from typing import Dict, List, Any, Tuple
import numpy as np
from config import *

class AnnotationFilter:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def filter_text_annotations(self, text_annotations: Dict[str, str]) -> Dict[str, str]:
        """Filter text annotations based on complexity and quality criteria."""
        filtered_annotations = {}
        
        for text_type, text in text_annotations.items():
            doc = self.nlp(text)
            
            # Check object count
            objects = [token for token in doc if token.dep_ in ["dobj", "pobj"]]
            if len(objects) > MAX_OBJECTS_PER_IMAGE:
                continue
                
            # Check action complexity
            root = [token for token in doc if token.dep_ == "ROOT"][0]
            action_complexity = len(list(root.children))
            if action_complexity < MIN_ACTION_COMPLEXITY:
                continue
                
            # Check object complexity
            object_complexity = max([len(list(obj.children)) for obj in objects]) if objects else 0
            if object_complexity < MIN_OBJECT_COMPLEXITY:
                continue
                
            filtered_annotations[text_type] = text
            
        return filtered_annotations
    
    def filter_region_annotations(self, regions: Dict[str, Any]) -> Dict[str, Any]:
        """Filter region annotations based on confidence and overlap."""
        # Filter by confidence
        confident_regions = {
            k: v for k, v in regions.items()
            if v['confidence'] > BOX_CONFIDENCE_THRESHOLD
        }
        
        # Apply NMS to remove overlapping boxes
        boxes = np.array([region['box'] for region in confident_regions.values()])
        scores = np.array([region['confidence'] for region in confident_regions.values()])
        
        keep_indices = self._non_max_suppression(boxes, scores, NMS_THRESHOLD)
        
        # Create filtered result
        filtered_regions = {
            k: confident_regions[k]
            for k in list(confident_regions.keys())[keep_indices]
        }
        
        return filtered_regions
    
    def filter_triplets(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter text-phrase-region triplets."""
        filtered_triplets = []
        
        for triplet in triplets:
            # Skip if phrase is in blacklist
            if triplet['phrase'].lower() in PHRASE_BLACKLIST:
                continue
                
            # Skip if confidence is too low
            if triplet['confidence'] < CONFIDENCE_THRESHOLD:
                continue
                
            # Add to filtered results
            filtered_triplets.append(triplet)
            
        return filtered_triplets
    
    def _non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray,
                           threshold: float) -> List[int]:
        """Implement Non-Maximum Suppression."""
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