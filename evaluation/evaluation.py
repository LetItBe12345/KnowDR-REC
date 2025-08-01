#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KVQA Evaluator - Unified evaluation framework
"""

import asyncio
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from tqdm import tqdm
import logging

from dataloader import KVQADataLoader
from improved_name_matching import ImprovedNameMatcher


class BoundingBoxEvaluator:
    """Bounding box evaluator - Calculate IoU, accuracy and other geometric metrics"""
    
    @staticmethod
    def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU of two bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection area
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
            
        except Exception as e:
            logging.warning(f"IoU calculation error: {e}")
            return 0.0
    
    @staticmethod
    def calculate_accuracy_at_threshold(iou: float, threshold: float = 0.5) -> bool:
        """Calculate accuracy at specified threshold"""
        return iou >= threshold
    
    @staticmethod
    def calculate_center_distance(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between centers of two bounding boxes"""
        try:
            cx1, cy1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
            cx2, cy2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2
            return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        except Exception as e:
            logging.warning(f"Center distance calculation error: {e}")
            return float('inf')


class KVQAEvaluationMetrics:
    """KVQA evaluation metrics calculator"""
    
    def __init__(self):
        self.bbox_evaluator = BoundingBoxEvaluator()
        self.name_matcher = ImprovedNameMatcher()
    
    def evaluate_positive_sample(self, 
                                prediction: Dict[str, Any], 
                                ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate positive sample
        
        Args:
            prediction: Prediction result
            ground_truth: Ground truth
            
        Returns:
            Evaluation result dictionary
        """
        # Extract predicted and ground truth bounding boxes
        pred_bbox = prediction.get('bounding_box')
        gt_bbox = ground_truth.get('bounding_box', [0, 0, 100, 100])
        
        # Use default invalid bounding box if prediction is None
        effective_bbox = pred_bbox if pred_bbox is not None else [0, 0, 0, 0]
        
        # Calculate geometric metrics
        iou = self.bbox_evaluator.calculate_iou(effective_bbox, gt_bbox)
        accuracy_at_50 = self.bbox_evaluator.calculate_accuracy_at_threshold(iou, 0.5)
        accuracy_at_75 = self.bbox_evaluator.calculate_accuracy_at_threshold(iou, 0.75)
        accuracy_at_90 = self.bbox_evaluator.calculate_accuracy_at_threshold(iou, 0.9)
        center_distance = self.bbox_evaluator.calculate_center_distance(effective_bbox, gt_bbox)
        
        # Calculate name matching metrics
        predicted_name = prediction.get('target_description', '')
        expected_name = ground_truth.get('person_name', '')
        
        name_similarity = self.name_matcher.calculate_name_similarity(predicted_name, expected_name)
        name_match = self.name_matcher.is_name_match(predicted_name, expected_name)
        
        # Calculate combined metrics (bounding box + name)
        combined_accuracy_at_50 = 1 if accuracy_at_50 and name_match else 0
        combined_accuracy_at_75 = 1 if accuracy_at_75 and name_match else 0
        combined_accuracy_at_90 = 1 if accuracy_at_90 and name_match else 0
        
        return {
            'iou': iou,
            'accuracy_at_50': accuracy_at_50,
            'accuracy_at_75': accuracy_at_75,
            'accuracy_at_90': accuracy_at_90,
            'center_distance': center_distance,
            'name_match': name_match,
            'name_similarity_scores': name_similarity,
            'overall_name_score': name_similarity['overall_score'],
            'weighted_name_score': name_similarity['weighted_score'],
            'combined_accuracy_at_50': combined_accuracy_at_50,
            'combined_accuracy_at_75': combined_accuracy_at_75,
            'combined_accuracy_at_90': combined_accuracy_at_90,
        }
    
    def evaluate_negative_sample(self, 
                                prediction: Dict[str, Any], 
                                ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate negative sample
        
        Args:
            prediction: Prediction result
            ground_truth: Ground truth
            
        Returns:
            Evaluation result dictionary
        """
        # Get ground truth answer
        gt_answer = ground_truth.get('gt_answer', 'No')
        
        # Determine prediction result
        predicted_bbox = prediction.get('bounding_box')
        target_description = prediction.get('target_description', 'unknown')
        
        if target_description == 'not_found':
            pred_answer = 'No'
            pred_bool = False
        elif target_description == 'found_but_no_bbox':
            pred_answer = 'Yes'
            pred_bool = True
        elif predicted_bbox is None:
            pred_answer = 'No'
            pred_bool = False
        else:
            pred_answer = 'Yes'
            pred_bool = True
        
        # Parse ground truth answer
        if gt_answer.lower() in ['no', 'false', '0']:
            gt_bool = False
        elif gt_answer.lower() in ['yes', 'true', '1']:
            gt_bool = True
        else:
            gt_bool = False  # Default to False
        
        # Calculate accuracy
        is_correct = (pred_bool == gt_bool)
        
        result = {
            'answer': pred_answer,
            'gt_answer': gt_answer,
            'is_correct': is_correct,
            'predicted_existence': pred_bool,
            'ground_truth_existence': gt_bool
        }
        
        # If incorrectly predicted bounding box, calculate IoU and other metrics
        if not is_correct and predicted_bbox is not None:
            gt_bbox = ground_truth.get('bounding_box', [0, 0, 100, 100])
            iou = self.bbox_evaluator.calculate_iou(predicted_bbox, gt_bbox)
            result.update({
                'iou': iou,
                'accuracy_at_50': self.bbox_evaluator.calculate_accuracy_at_threshold(iou, 0.5),
                'accuracy_at_75': self.bbox_evaluator.calculate_accuracy_at_threshold(iou, 0.75),
                'accuracy_at_90': self.bbox_evaluator.calculate_accuracy_at_threshold(iou, 0.9),
            })
        else:
            result.update({
                'iou': 0.0,
                'accuracy_at_50': False,
                'accuracy_at_75': False,
                'accuracy_at_90': False,
            })
        
        return result


class KVQAEvaluator:
    """
    KVQA unified evaluator
    
    Features:
    1. Support multiple predictors (Qwen/Gemini/Grok etc.)
    2. Handle positive and negative sample evaluation
    3. Generate detailed evaluation reports
    4. Support batch and single sample evaluation
    """
    
    def __init__(self, 
                 predictor: Any,
                 output_dir: str = "evaluation_results",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize evaluator
        
        Args:
            predictor: Predictor instance (supports async predict_single interface)
            output_dir: Output directory
            logger: Logger instance
        """
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.dataloader = KVQADataLoader(logger)
        self.metrics = KVQAEvaluationMetrics()
        
        # Setup logger
        self.logger = logger or self._setup_logger()
        
        # Statistics
        self.evaluation_stats = {
            'start_time': None,
            'end_time': None,
            'total_samples': 0,
            'successful_samples': 0,
            'failed_samples': 0,
            'positive_samples': 0,
            'negative_samples': 0
        }
        
        self.logger.info(f"KVQA evaluator initialized, output directory: {output_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.FileHandler(self.output_dir / 'evaluation.log', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate single sample"""
        try:
            sample_type = sample.get('type', 'positive')
            question_id = sample.get('question_id', 'unknown')
            
            # Load image
            from PIL import Image
            image_path = sample['image_path']
            
            if not os.path.exists(image_path):
                return self._create_error_result(sample, 'Image file not found')
            
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                return self._create_error_result(sample, f'Image loading failed: {e}')
            
            # Execute prediction
            prediction_result = await self.predictor.predict_single(
                image, 
                sample['rewritten_question'], 
                sample_type=sample_type
            )
            
            if not prediction_result.get('success', False):
                return self._create_error_result(sample, 'Prediction failed')
            
            # Extract prediction data
            if sample_type == 'negative' and 'existence_prediction' in prediction_result:
                # Handle dual prediction results
                prediction = self._extract_dual_prediction(prediction_result)
            else:
                # Handle single prediction results
                prediction = prediction_result.get('parsed_result', {})
            
            # Calculate evaluation metrics
            if sample_type == 'positive':
                metrics_result = self.metrics.evaluate_positive_sample(prediction, sample)
            else:
                metrics_result = self.metrics.evaluate_negative_sample(prediction, sample)
            
            # Assemble final result
            result = {
                'question_id': question_id,
                'type': sample_type,
                'question_text': sample['rewritten_question'],
                'image_path': image_path,
                'ground_truth': {
                    'bounding_box': sample['bounding_box'],
                    'person_name': sample.get('person_name', ''),
                    'gt_answer': sample.get('gt_answer')
                },
                'prediction': prediction,
                'metrics': metrics_result,
                'model_info': {
                    'provider': getattr(self.predictor, 'provider', 'unknown'),
                    'model': getattr(self.predictor, 'model_name', 'unknown')
                },
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sample evaluation failed {sample.get('question_id', 'unknown')}: {e}")
            return self._create_error_result(sample, str(e))
    
    def _extract_dual_prediction(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dual prediction results"""
        existence_pred = prediction_result.get('existence_prediction', {})
        location_pred = prediction_result.get('location_prediction', {})
        
        existence_result = existence_pred.get('parsed_result', {})
        location_result = location_pred.get('parsed_result', {})
        
        # Combine results
        combined_result = {
            'target_description': 'dual_prediction',
            'bounding_box': location_result.get('bounding_box'),
            'existence_answer': existence_result.get('answer', 'No'),
            'existence_confidence': existence_result.get('confidence', 0.0),
            'location_bbox': location_result.get('bounding_box'),
            'raw_existence_response': existence_pred.get('response', ''),
            'raw_location_response': location_pred.get('response', '')
        }
        
        # Determine final target_description
        if existence_result.get('existence', False):
            if location_result.get('bounding_box') is not None:
                combined_result['target_description'] = 'found_with_bbox'
            else:
                combined_result['target_description'] = 'found_but_no_bbox'
        else:
            combined_result['target_description'] = 'not_found'
        
        return combined_result
    
    def _create_error_result(self, sample: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'question_id': sample.get('question_id', 'unknown'),
            'type': sample.get('type', 'unknown'),
            'question_text': sample.get('rewritten_question', ''),
            'image_path': sample.get('image_path', ''),
            'error': error_message,
            'success': False,
            'prediction': None,
            'metrics': None
        }
    
    async def evaluate_dataset(self, 
                              dataset_path: str, 
                              max_samples: Optional[int] = None,
                              batch_size: int = None) -> Dict[str, Any]:
        """
        Evaluate entire dataset
        
        Args:
            dataset_path: Dataset path
            max_samples: Maximum number of samples
            batch_size: Batch processing size (if None, all concurrent)
            
        Returns:
            Evaluation result dictionary
        """
        self.evaluation_stats['start_time'] = time.time()
        
        # Load dataset
        dataset = self.dataloader.load_dataset(
            dataset_path, 
            max_samples=max_samples,
            validate_images=False  # Validate during evaluation
        )
        
        if not dataset:
            return {'error': 'Failed to load dataset', 'statistics': self.evaluation_stats}
        
        self.evaluation_stats['total_samples'] = len(dataset)
        
        # Execute evaluation
        if batch_size:
            results = await self._evaluate_in_batches(dataset, batch_size)
        else:
            results = await self._evaluate_all_concurrent(dataset)
        
        self.evaluation_stats['end_time'] = time.time()
        
        # Calculate statistics
        statistics = self._calculate_statistics(results, dataset_path)
        
        # Save results
        evaluation_data = {
            'statistics': statistics,
            'results': results,
            'dataset_info': self.dataloader.get_statistics(),
            'predictor_stats': getattr(self.predictor, 'get_statistics', lambda: {})()
        }
        
        output_file = self._save_results(evaluation_data)
        
        self.logger.info(f"Evaluation completed! Results saved to: {output_file}")
        self._log_summary(statistics)
        
        return evaluation_data
    
    async def _evaluate_all_concurrent(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Full concurrent evaluation"""
        self.logger.info(f"Starting concurrent evaluation of {len(dataset)} samples")
        
        tasks = [self.evaluate_single_sample(sample) for sample in dataset]
        
        results = []
        with tqdm(total=len(tasks), desc="Evaluation progress") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
        
        return results
    
    async def _evaluate_in_batches(self, 
                                   dataset: List[Dict[str, Any]], 
                                   batch_size: int) -> List[Dict[str, Any]]:
        """Batch evaluation"""
        self.logger.info(f"Starting batch evaluation of {len(dataset)} samples, batch size: {batch_size}")
        
        all_results = []
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        
        with tqdm(total=len(dataset), desc="Evaluation progress") as pbar:
            for batch_idx in range(0, len(dataset), batch_size):
                batch = dataset[batch_idx:batch_idx + batch_size]
                
                # Process current batch concurrently
                tasks = [self.evaluate_single_sample(sample) for sample in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        error_result = self._create_error_result(batch[i], str(result))
                        all_results.append(error_result)
                    else:
                        all_results.append(result)
                
                pbar.update(len(batch))
        
        return all_results
    
    def _calculate_statistics(self, 
                             results: List[Dict[str, Any]], 
                             dataset_path: str) -> Dict[str, Any]:
        """Calculate statistics"""
        # Separate successful and failed results
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        # Separate by type
        positive_results = [r for r in successful_results if r.get('type') == 'positive']
        negative_results = [r for r in successful_results if r.get('type') == 'negative']
        
        # Update statistics
        self.evaluation_stats.update({
            'successful_samples': len(successful_results),
            'failed_samples': len(failed_results),
            'positive_samples': len(positive_results),
            'negative_samples': len(negative_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'evaluation_time': self.evaluation_stats['end_time'] - self.evaluation_stats['start_time'],
            'samples_per_second': len(results) / (self.evaluation_stats['end_time'] - self.evaluation_stats['start_time'])
        })
        
        statistics = {
            'dataset_path': dataset_path,
            'timestamp': datetime.now().isoformat(),
            'overall': self.evaluation_stats.copy()
        }
        
        # Calculate positive sample statistics
        if positive_results:
            statistics['positive'] = self._calculate_positive_statistics(positive_results)
        
        # Calculate negative sample statistics
        if negative_results:
            statistics['negative'] = self._calculate_negative_statistics(negative_results)
        
        return statistics
    
    def _calculate_positive_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate positive sample statistics"""
        metrics_list = [r['metrics'] for r in results]
        
        # Extract various metrics
        ious = [m['iou'] for m in metrics_list]
        acc_50 = [m['accuracy_at_50'] for m in metrics_list]
        acc_75 = [m['accuracy_at_75'] for m in metrics_list]
        acc_90 = [m['accuracy_at_90'] for m in metrics_list]
        name_matches = [m['name_match'] for m in metrics_list]
        combined_acc_50 = [m['combined_accuracy_at_50'] for m in metrics_list]
        combined_acc_75 = [m['combined_accuracy_at_75'] for m in metrics_list]
        combined_acc_90 = [m['combined_accuracy_at_90'] for m in metrics_list]
        
        return {
            'total_samples': len(results),
            'mean_iou': float(np.mean(ious)),
            'median_iou': float(np.median(ious)),
            'std_iou': float(np.std(ious)),
            'accuracy_at_50': float(np.mean(acc_50)),
            'accuracy_at_75': float(np.mean(acc_75)),
            'accuracy_at_90': float(np.mean(acc_90)),
            'macc': float(np.mean([np.mean(acc_50), np.mean(acc_75), np.mean(acc_90)])),
            'name_match_accuracy': float(np.mean(name_matches)),
            'combined_accuracy_at_50': float(np.mean(combined_acc_50)),
            'combined_accuracy_at_75': float(np.mean(combined_acc_75)),
            'combined_accuracy_at_90': float(np.mean(combined_acc_90)),
            'combined_macc': float(np.mean([np.mean(combined_acc_50), np.mean(combined_acc_75), np.mean(combined_acc_90)])),
        }
    
    def _calculate_negative_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate negative sample statistics"""
        metrics_list = [r['metrics'] for r in results]
        
        correct_predictions = [m['is_correct'] for m in metrics_list]
        incorrect_results = [m for m in metrics_list if not m['is_correct']]
        
        stats = {
            'total_samples': len(results),
            'accuracy': float(np.mean(correct_predictions)),
            'total_errors': len(incorrect_results)
        }
        
        # Calculate IoU metrics for incorrect predictions
        if incorrect_results:
            error_ious = [m.get('iou', 0.0) for m in incorrect_results]
            error_acc_50 = [m.get('accuracy_at_50', False) for m in incorrect_results]
            error_acc_75 = [m.get('accuracy_at_75', False) for m in incorrect_results]
            error_acc_90 = [m.get('accuracy_at_90', False) for m in incorrect_results]
            
            stats['error_metrics'] = {
                'mean_iou_on_errors': float(np.mean(error_ious)),
                'accuracy_at_50_on_errors': float(np.mean(error_acc_50)),
                'accuracy_at_75_on_errors': float(np.mean(error_acc_75)),
                'accuracy_at_90_on_errors': float(np.mean(error_acc_90)),
                'macc_on_errors': float(np.mean([np.mean(error_acc_50), np.mean(error_acc_75), np.mean(error_acc_90)]))
            }
        
        return stats
    
    def _save_results(self, evaluation_data: Dict[str, Any]) -> Path:
        """Save evaluation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        provider = getattr(self.predictor, 'provider', 'unknown')
        model = getattr(self.predictor, 'model_name', 'unknown')
        
        filename = f"evaluation_{provider}_{model}_{timestamp}.json"
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def _log_summary(self, statistics: Dict[str, Any]):
        """Log evaluation summary"""
        overall = statistics['overall']
        
        self.logger.info("=== Evaluation Summary ===")
        self.logger.info(f"Total samples: {overall['total_samples']}")
        self.logger.info(f"Success rate: {overall['success_rate']:.2%}")
        self.logger.info(f"Evaluation time: {overall['evaluation_time']:.1f}s")
        self.logger.info(f"Processing speed: {overall['samples_per_second']:.1f} samples/s")
        
        if 'positive' in statistics:
            pos_stats = statistics['positive']
            self.logger.info(f"=== Positive samples ({pos_stats['total_samples']}) ===")
            self.logger.info(f"Mean IoU: {pos_stats['mean_iou']:.3f}")
            self.logger.info(f"Accuracy@0.5: {pos_stats['accuracy_at_50']:.2%}")
            self.logger.info(f"Name match accuracy: {pos_stats['name_match_accuracy']:.2%}")
            self.logger.info(f"Combined accuracy@0.5: {pos_stats['combined_accuracy_at_50']:.2%}")
        
        if 'negative' in statistics:
            neg_stats = statistics['negative']
            self.logger.info(f"=== Negative samples ({neg_stats['total_samples']}) ===")
            self.logger.info(f"Accuracy: {neg_stats['accuracy']:.2%}")
            if neg_stats['total_errors'] > 0:
                error_metrics = neg_stats.get('error_metrics', {})
                self.logger.info(f"Error prediction IoU: {error_metrics.get('mean_iou_on_errors', 0.0):.3f}")


async def test_evaluator():
    """Test evaluator"""
    print("🔥 Testing KVQA Evaluator")
    
    # Create mock predictor
    class MockPredictor:
        def __init__(self):
            self.provider = "mock"
            self.model_name = "test-model"
        
        async def predict_single(self, image, question, sample_type='positive'):
            import random
            
            if sample_type == 'positive':
                return {
                    'success': True,
                    'parsed_result': {
                        'target_description': 'Test Person',
                        'bounding_box': [100 + random.randint(-20, 20), 
                                       150 + random.randint(-20, 20), 
                                       300 + random.randint(-20, 20), 
                                       450 + random.randint(-20, 20)]
                    }
                }
            else:
                return {
                    'success': True,
                    'parsed_result': {
                        'target_description': 'not_found',
                        'bounding_box': None
                    }
                }
        
        def get_statistics(self):
            return {'mock_stats': True}
    
    # Create evaluator
    predictor = MockPredictor()
    evaluator = KVQAEvaluator(predictor, output_dir="test_evaluation_results")
    
    # Test dataset
    test_dataset_path = "/home/jin/KVQA/改写/back_up/rewritten_questions_1_more_hop.json"
    
    if os.path.exists(test_dataset_path):
        print(f"Testing dataset: {test_dataset_path}")
        
        # Run evaluation
        results = await evaluator.evaluate_dataset(test_dataset_path, max_samples=5)
        
        print("Evaluation completed!")
        print("Statistics:")
        print(json.dumps(results['statistics'], indent=2, ensure_ascii=False))
    else:
        print(f"Test dataset does not exist: {test_dataset_path}")


if __name__ == "__main__":
    asyncio.run(test_evaluator())