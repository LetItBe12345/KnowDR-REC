#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KVQA Data Loader - Unified dataset loading and preprocessing management
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
import logging
import time


class KVQADataLoader:
    """
    KVQA Dataset Loader
    
    Features:
    1. Load various formats of KVQA datasets
    2. Unify data structures
    3. Image preprocessing and validation
    4. Support positive and negative sample datasets
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize data loader
        
        Args:
            logger: Logger instance, creates default logger if None
        """
        self.logger = logger or self._setup_default_logger()
        self.image_cache = {}  # Image cache
        self.dataset_stats = {}  # Dataset statistics
        
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_dataset(self, 
                    dataset_path: str, 
                    max_samples: Optional[int] = None,
                    validate_images: bool = True,
                    cache_images: bool = False) -> List[Dict[str, Any]]:
        """
        Load dataset
        
        Args:
            dataset_path: Dataset file path
            max_samples: Maximum number of samples limit
            validate_images: Whether to validate image file existence
            cache_images: Whether to cache images to memory
            
        Returns:
            Unified format dataset list
        """
        start_time = time.time()
        
        # Check file existence
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
        
        # Load JSON data
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.logger.info(f"Raw dataset loaded: {len(raw_data)} samples")
        
        # Limit sample count
        if max_samples and max_samples < len(raw_data):
            raw_data = raw_data[:max_samples]
            self.logger.info(f"Limited sample count to: {max_samples}")
        
        # Process data
        dataset = []
        failed_samples = 0
        
        for i, sample in enumerate(raw_data):
            try:
                processed_sample = self._process_sample(
                    sample, 
                    validate_images=validate_images,
                    cache_images=cache_images
                )
                
                if processed_sample:
                    dataset.append(processed_sample)
                else:
                    failed_samples += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to process sample {i}: {e}")
                failed_samples += 1
                continue
        
        load_time = time.time() - start_time
        
        # Update statistics
        self.dataset_stats = {
            'dataset_path': dataset_path,
            'total_samples': len(dataset),
            'failed_samples': failed_samples,
            'success_rate': len(dataset) / (len(dataset) + failed_samples) if (len(dataset) + failed_samples) > 0 else 0,
            'load_time': load_time,
            'validate_images': validate_images,
            'cache_images': cache_images
        }
        
        self.logger.info(f"Dataset loading completed: success {len(dataset)}, failed {failed_samples}, time {load_time:.2f}s")
        
        return dataset
    
    def _process_sample(self, 
                       sample: Dict[str, Any], 
                       validate_images: bool = True,
                       cache_images: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process single sample, unify data format
        
        Args:
            sample: Raw sample data
            validate_images: Whether to validate images
            cache_images: Whether to cache images
            
        Returns:
            Processed sample data, returns None on failure
        """
        try:
            # Determine sample type
            sample_type = self._determine_sample_type(sample)
            
            # Extract basic information
            processed_sample = {
                'type': sample_type,
                'question_id': self._extract_question_id(sample),
                'question_text': self._extract_question_text(sample),
                'rewritten_question': self._extract_rewritten_question(sample),
                'image_path': self._extract_image_path(sample),
                'person_name': self._extract_person_name(sample),
                'bounding_box': self._extract_bounding_box(sample),
                'original_sample': sample  # Keep original data for debugging
            }
            
            # Special handling for negative samples
            if sample_type == 'negative':
                processed_sample['gt_answer'] = sample.get('gt', 'No')
                processed_sample['negative_type'] = sample.get('negative_type', 'unknown')
            
            # Validate image file
            if validate_images:
                if not self._validate_image(processed_sample['image_path']):
                    self.logger.warning(f"Image file validation failed: {processed_sample['image_path']}")
                    return None
            
            # Cache image
            if cache_images:
                processed_sample['image'] = self._load_and_cache_image(processed_sample['image_path'])
            
            return processed_sample
            
        except Exception as e:
            self.logger.error(f"Sample processing failed: {e}")
            return None
    
    def _determine_sample_type(self, sample: Dict[str, Any]) -> str:
        """Determine sample type"""
        # Check if there's explicit type field
        if 'type' in sample:
            return sample['type']
        
        # Check if there's gt field (negative sample feature)
        if 'gt' in sample:
            return 'negative'
        
        # Check if there's negative_type field
        if 'negative_type' in sample:
            return 'negative'
        
        # Default to positive sample
        return 'positive'
    
    def _extract_question_id(self, sample: Dict[str, Any]) -> str:
        """Extract question ID"""
        return sample.get('question_id', sample.get('id', f"sample_{hash(str(sample))}"[:8]))
    
    def _extract_question_text(self, sample: Dict[str, Any]) -> str:
        """Extract original question text"""
        return sample.get('question', sample.get('question_text', ''))
    
    def _extract_rewritten_question(self, sample: Dict[str, Any]) -> str:
        """Extract rewritten question text"""
        return sample.get('rewritten_question', self._extract_question_text(sample))
    
    def _extract_image_path(self, sample: Dict[str, Any]) -> str:
        """Extract image path"""
        return sample.get('image_path', sample.get('image', ''))
    
    def _extract_person_name(self, sample: Dict[str, Any]) -> str:
        """Extract person name"""
        return sample.get('person_name', sample.get('target_person', sample.get('answer', '')))
    
    def _extract_bounding_box(self, sample: Dict[str, Any]) -> List[float]:
        """Extract bounding box coordinates"""
        bbox = sample.get('bounding_box', sample.get('bbox', [0, 0, 100, 100]))
        
        # Ensure it's a list of floats
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [float(coord) for coord in bbox]
        
        # Default bounding box
        return [0.0, 0.0, 100.0, 100.0]
    
    def _validate_image(self, image_path: str) -> bool:
        """Validate image file"""
        if not image_path or not os.path.exists(image_path):
            return False
        
        try:
            # Try to open image file
            with Image.open(image_path) as img:
                # Validate image format
                img.verify()
            return True
        except Exception:
            return False
    
    def _load_and_cache_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and cache image"""
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        
        try:
            image = Image.open(image_path).convert('RGB')
            self.image_cache[image_path] = image
            return image
        except Exception as e:
            self.logger.warning(f"Image loading failed {image_path}: {e}")
            return None
    
    def get_batch_samples(self, 
                         dataset: List[Dict[str, Any]], 
                         batch_size: int = 32,
                         shuffle: bool = False) -> List[List[Dict[str, Any]]]:
        """
        Batch dataset
        
        Args:
            dataset: Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle order
            
        Returns:
            Batched dataset list
        """
        if shuffle:
            import random
            dataset = dataset.copy()
            random.shuffle(dataset)
        
        batches = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batches.append(batch)
        
        self.logger.info(f"Dataset batching completed: {len(batches)} batches, {batch_size} samples per batch")
        return batches
    
    def filter_by_type(self, 
                      dataset: List[Dict[str, Any]], 
                      sample_type: str) -> List[Dict[str, Any]]:
        """
        Filter dataset by sample type
        
        Args:
            dataset: Dataset
            sample_type: Sample type ('positive' or 'negative')
            
        Returns:
            Filtered dataset
        """
        filtered = [sample for sample in dataset if sample.get('type') == sample_type]
        self.logger.info(f"Filtered by type '{sample_type}': {len(filtered)} samples")
        return filtered
    
    def load_multiple_datasets(self, 
                              dataset_paths: List[str],
                              merge: bool = True,
                              max_samples_per_dataset: Optional[int] = None) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Load multiple datasets
        
        Args:
            dataset_paths: List of dataset paths
            merge: Whether to merge all datasets
            max_samples_per_dataset: Maximum samples per dataset
            
        Returns:
            Merged dataset or dataset dictionary
        """
        datasets = {}
        
        for path in dataset_paths:
            try:
                dataset_name = Path(path).stem
                dataset = self.load_dataset(
                    path, 
                    max_samples=max_samples_per_dataset,
                    validate_images=False  # Skip validation during batch loading for speed
                )
                datasets[dataset_name] = dataset
                self.logger.info(f"Loaded dataset '{dataset_name}': {len(dataset)} samples")
            except Exception as e:
                self.logger.error(f"Dataset loading failed {path}: {e}")
                continue
        
        if merge:
            merged_dataset = []
            for dataset_name, dataset in datasets.items():
                # Add dataset source identifier for each sample
                for sample in dataset:
                    sample['dataset_source'] = dataset_name
                merged_dataset.extend(dataset)
            
            self.logger.info(f"Merged {len(datasets)} datasets, total {len(merged_dataset)} samples")
            return merged_dataset
        
        return datasets
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data loading statistics"""
        stats = self.dataset_stats.copy()
        stats['image_cache_size'] = len(self.image_cache)
        return stats
    
    def clear_cache(self):
        """Clear image cache"""
        self.image_cache.clear()
        self.logger.info("Image cache cleared")
    
    def validate_dataset_integrity(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate dataset integrity
        
        Args:
            dataset: Dataset
            
        Returns:
            Validation report
        """
        report = {
            'total_samples': len(dataset),
            'missing_fields': {},
            'invalid_images': [],
            'invalid_bboxes': [],
            'sample_types': {'positive': 0, 'negative': 0, 'unknown': 0}
        }
        
        required_fields = ['question_id', 'question_text', 'image_path', 'bounding_box']
        
        for i, sample in enumerate(dataset):
            # Check required fields
            for field in required_fields:
                if field not in sample or not sample[field]:
                    if field not in report['missing_fields']:
                        report['missing_fields'][field] = []
                    report['missing_fields'][field].append(i)
            
            # Validate image
            if not self._validate_image(sample.get('image_path', '')):
                report['invalid_images'].append(i)
            
            # Validate bounding box
            bbox = sample.get('bounding_box', [])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                report['invalid_bboxes'].append(i)
            
            # Count sample types
            sample_type = sample.get('type', 'unknown')
            if sample_type in report['sample_types']:
                report['sample_types'][sample_type] += 1
            else:
                report['sample_types']['unknown'] += 1
        
        # Calculate integrity score
        total_issues = (
            sum(len(issues) for issues in report['missing_fields'].values()) +
            len(report['invalid_images']) +
            len(report['invalid_bboxes'])
        )
        
        report['integrity_score'] = 1.0 - (total_issues / (len(dataset) * len(required_fields) + len(dataset) * 2))
        report['integrity_score'] = max(0.0, report['integrity_score'])
        
        self.logger.info(f"Dataset integrity validation completed, integrity score: {report['integrity_score']:.2%}")
        
        return report


def test_dataloader():
    """Test dataloader functionality"""
    print("🔥 Testing KVQA Data Loader")
    
    # Create data loader
    loader = KVQADataLoader()
    
    # Test dataset paths (modify according to actual situation)
    test_datasets = [
        "/home/jin/KVQA/改写/back_up/rewritten_questions_1_more_hop.json",
        "/home/jin/KVQA/改写/back_up/negative_final.json"
    ]
    
    for dataset_path in test_datasets:
        if os.path.exists(dataset_path):
            print(f"\nTesting dataset: {dataset_path}")
            
            # Load dataset
            dataset = loader.load_dataset(dataset_path, max_samples=5, validate_images=False)
            
            print(f"Loading result: {len(dataset)} samples")
            
            # Show first sample
            if dataset:
                print("First sample:")
                sample = dataset[0]
                for key, value in sample.items():
                    if key != 'original_sample':  # Skip original data
                        print(f"  {key}: {value}")
            
            # Validate dataset integrity
            report = loader.validate_dataset_integrity(dataset)
            print(f"Integrity score: {report['integrity_score']:.2%}")
            
            # Show statistics
            stats = loader.get_statistics()
            print(f"Loading statistics: {stats}")
        else:
            print(f"Dataset does not exist: {dataset_path}")


if __name__ == "__main__":
    test_dataloader()