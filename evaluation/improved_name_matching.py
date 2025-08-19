#!/usr/bin/env python3
"""
Improved Name Matching - Simple name similarity calculation
"""

import re
from typing import Dict, Any
from difflib import SequenceMatcher


class ImprovedNameMatcher:
    """
    Simple name matcher for KVQA evaluation
    """
    
    def __init__(self):
        self.similarity_threshold = 0.7
    
    def normalize_name(self, name: str) -> str:
        """Normalize name for comparison"""
        if not name:
            return ""
        
        # Convert to lowercase and remove extra spaces
        name = name.lower().strip()
        
        # Remove common prefixes/suffixes
        name = re.sub(r'\b(the|actor|person|man|woman|guy|girl)\b', '', name)
        
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def calculate_name_similarity(self, predicted_name: str, expected_name: str) -> Dict[str, Any]:
        """Calculate name similarity scores"""
        pred_normalized = self.normalize_name(predicted_name)
        exp_normalized = self.normalize_name(expected_name)
        
        if not pred_normalized or not exp_normalized:
            return {
                "exact_match": False,
                "sequence_similarity": 0.0,
                "word_overlap": 0.0,
                "overall_score": 0.0,
                "weighted_score": 0.0
            }
        
        # Exact match after normalization
        exact_match = pred_normalized == exp_normalized
        
        # Sequence similarity
        sequence_similarity = SequenceMatcher(None, pred_normalized, exp_normalized).ratio()
        
        # Word overlap
        pred_words = set(pred_normalized.split())
        exp_words = set(exp_normalized.split())
        
        if len(exp_words) > 0:
            word_overlap = len(pred_words.intersection(exp_words)) / len(exp_words)
        else:
            word_overlap = 0.0
        
        # Overall score (average of different metrics)
        overall_score = (sequence_similarity + word_overlap) / 2
        
        # Weighted score (emphasize word overlap for names)
        weighted_score = 0.3 * sequence_similarity + 0.7 * word_overlap
        
        return {
            "exact_match": exact_match,
            "sequence_similarity": sequence_similarity,
            "word_overlap": word_overlap,
            "overall_score": overall_score,
            "weighted_score": weighted_score
        }
    
    def is_name_match(self, predicted_name: str, expected_name: str) -> bool:
        """Determine if two names match"""
        similarity = self.calculate_name_similarity(predicted_name, expected_name)
        return similarity["weighted_score"] >= self.similarity_threshold
