#!/usr/bin/env python3
"""
Improved Name Matching - Dictionary-based name matching with similarity fallback
"""

import re
import json
import os
from typing import Dict, Any, Set, Optional
from difflib import SequenceMatcher
from pathlib import Path


class ImprovedNameMatcher:
    """
    Dictionary-based name matcher for KVQA evaluation with similarity fallback
    """
    
    def __init__(self, variants_dict_path: Optional[str] = None):
        self.similarity_threshold = 0.7
        self.variants_dict = {}
        self.name_to_key_lookup = {}  # Fast lookup table: variant name -> dictionary key
        
        # Default dictionary path
        default_dict_path = ""
        dict_path = variants_dict_path or default_dict_path
        
        # Load variants dictionary
        if dict_path and os.path.exists(dict_path):
            self.load_variants_dict(dict_path)
        else:
            print(f"Warning: Name variants dictionary not found at {dict_path}")
            print("Falling back to similarity-based matching only")
    
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
        
        # Check dictionary matching
        dict_match = False
        if self.variants_dict:
            dict_match = self.is_name_match_dict(predicted_name, expected_name)
        
        # If dictionary matching succeeds, give highest score
        if dict_match:
            weighted_score = 1.0
            overall_score = 1.0
        
        return {
            "exact_match": exact_match,
            "sequence_similarity": sequence_similarity,
            "word_overlap": word_overlap,
            "overall_score": overall_score,
            "weighted_score": weighted_score,
            "dict_match": dict_match
        }
    
    def load_variants_dict(self, dict_path: str):
        """Load name variants dictionary"""
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.variants_dict = json.load(f)
            
            # Build fast lookup table
            self._build_lookup_table()
            print(f"✅ Successfully loaded name variants dictionary: {len(self.variants_dict)} persons, {len(self.name_to_key_lookup)} variants")
            
        except Exception as e:
            print(f"❌ Failed to load name variants dictionary {dict_path}: {e}")
            self.variants_dict = {}
            self.name_to_key_lookup = {}
    
    def _build_lookup_table(self):
        """Build fast lookup table"""
        self.name_to_key_lookup = {}
        
        for person_key, person_data in self.variants_dict.items():
            # Add canonical name
            canonical_name = person_data.get("canonical_name", "")
            if canonical_name:
                self._add_to_lookup(canonical_name, person_key)
            
            # Add all variants
            variants = person_data.get("variants", [])
            for variant in variants:
                if variant:
                    self._add_to_lookup(variant, person_key)
            
            # Add aliases (if any)
            aliases = person_data.get("aliases", [])
            for alias in aliases:
                if alias:
                    self._add_to_lookup(alias, person_key)
    
    def _add_to_lookup(self, name: str, person_key: str):
        """Add name and its normalized version to lookup table"""
        # Original name
        self.name_to_key_lookup[name.lower().strip()] = person_key
        
        # Normalized version
        normalized = self.normalize_name(name)
        if normalized:
            self.name_to_key_lookup[normalized] = person_key
    
    def _find_person_key(self, name: str) -> Optional[str]:
        """Find person key corresponding to name"""
        if not name:
            return None
        
        # Try direct matching
        key = self.name_to_key_lookup.get(name.lower().strip())
        if key:
            return key
        
        # Try normalized matching
        normalized = self.normalize_name(name)
        if normalized:
            key = self.name_to_key_lookup.get(normalized)
            if key:
                return key
        
        return None
    
    def is_name_match_dict(self, predicted_name: str, expected_name: str) -> bool:
        """Use dictionary for name matching"""
        if not predicted_name or not expected_name:
            return False
        
        # Find person keys corresponding to both names
        pred_key = self._find_person_key(predicted_name)
        exp_key = self._find_person_key(expected_name)
        
        # If both keys are found and identical, matching succeeds
        if pred_key and exp_key and pred_key == exp_key:
            return True
        
        return False
    
    def is_name_match_similarity(self, predicted_name: str, expected_name: str) -> bool:
        """Use similarity for name matching (original logic)"""
        similarity = self.calculate_name_similarity(predicted_name, expected_name)
        return similarity["weighted_score"] >= self.similarity_threshold
    
    def is_name_match(self, predicted_name: str, expected_name: str) -> bool:
        """Unified name matching interface"""
        # Prioritize dictionary matching
        if self.variants_dict and self.is_name_match_dict(predicted_name, expected_name):
            return True
        
        # Fall back to similarity matching when dictionary matching fails
        return self.is_name_match_similarity(predicted_name, expected_name)
    
    def get_canonical_name(self, name: str) -> Optional[str]:
        """Get canonical name"""
        person_key = self._find_person_key(name)
        if person_key and person_key in self.variants_dict:
            return self.variants_dict[person_key].get("canonical_name")
        return None
    
    def get_all_variants(self, name: str) -> list:
        """Get all variants of a name"""
        person_key = self._find_person_key(name)
        if person_key and person_key in self.variants_dict:
            return self.variants_dict[person_key].get("variants", [])
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dictionary statistics"""
        if not self.variants_dict:
            return {
                "dictionary_loaded": False,
                "total_persons": 0,
                "total_variants": 0,
                "lookup_table_size": 0
            }
        
        total_variants = 0
        total_aliases = 0
        
        for person_data in self.variants_dict.values():
            total_variants += len(person_data.get("variants", []))
            total_aliases += len(person_data.get("aliases", []))
        
        return {
            "dictionary_loaded": True,
            "total_persons": len(self.variants_dict),
            "total_variants": total_variants,
            "total_aliases": total_aliases,
            "lookup_table_size": len(self.name_to_key_lookup),
            "average_variants_per_person": total_variants / len(self.variants_dict) if self.variants_dict else 0
        }
    
    def debug_match(self, predicted_name: str, expected_name: str) -> Dict[str, Any]:
        """Debug matching process, return detailed information"""
        pred_key = self._find_person_key(predicted_name)
        exp_key = self._find_person_key(expected_name)
        dict_match = self.is_name_match_dict(predicted_name, expected_name)
        similarity_match = self.is_name_match_similarity(predicted_name, expected_name)
        final_match = self.is_name_match(predicted_name, expected_name)
        
        return {
            "predicted_name": predicted_name,
            "expected_name": expected_name,
            "predicted_normalized": self.normalize_name(predicted_name),
            "expected_normalized": self.normalize_name(expected_name),
            "predicted_key": pred_key,
            "expected_key": exp_key,
            "dict_match": dict_match,
            "similarity_match": similarity_match,
            "final_match": final_match,
            "predicted_canonical": self.get_canonical_name(predicted_name),
            "expected_canonical": self.get_canonical_name(expected_name),
            "similarity_scores": self.calculate_name_similarity(predicted_name, expected_name)
        }
