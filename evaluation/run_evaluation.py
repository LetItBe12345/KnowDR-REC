#!/usr/bin/env python3
"""
KVQA Evaluation Framework - Main Entry Point
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from core.unified_predictor import UnifiedPredictor
from evaluation import KVQAEvaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables
        api_key = config.get('api', {}).get('openrouter_api_key', '')
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            config['api']['openrouter_api_key'] = os.getenv(env_var, '')
        
        return config
    except Exception as e:
        print(f"Failed to load config {config_path}: {e}")
        sys.exit(1)


def setup_logging(output_dir: str, verbose: bool = False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, 'evaluation.log'), 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration"""
    # Check API key
    api_key = config.get('api', {}).get('openrouter_api_key', '')
    if not api_key:
        print("Error: OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
        return False
    
    # Check models
    models = config.get('models', [])
    if not models:
        print("Error: No models specified in configuration.")
        return False
    
    return True


async def run_single_model_evaluation(
    model_name: str,
    config: Dict[str, Any],
    dataset_path: str,
    output_dir: str) -> Dict[str, Any]:
    """Run evaluation for a single model"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation for model: {model_name}")
    
    try:
        # Create predictor
        predictor = UnifiedPredictor(
            model_name=model_name,
            api_key=config['api']['openrouter_api_key'],
            models_config_path="config/models.yaml",
            prompts_config_path="config/prompts.yaml"
        )
        
        # Create evaluator
        model_output_dir = os.path.join(output_dir, f"model_{model_name}")
        evaluator = KVQAEvaluator(
            predictor=predictor,
            output_dir=model_output_dir
        )
        
        # Run evaluation
        eval_config = config.get('evaluation', {})
        results = await evaluator.evaluate_dataset(
            dataset_path=dataset_path,
            max_samples=eval_config.get('max_samples'),
            batch_size=eval_config.get('batch_size', 32)
        )
        
        logger.info(f"Completed evaluation for model: {model_name}")
        return {
            'model': model_name,
            'success': True,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed for model {model_name}: {e}")
        return {
            'model': model_name,
            'success': False,
            'error': str(e)
        }


async def run_evaluation(
    config_path: str,
    dataset_path: str,
    models: List[str] = None,
    output_dir: str = None,
    max_samples: int = None) -> Dict[str, Any]:
    """Run complete evaluation"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Validate configuration
    if not validate_config(config):
        return {'error': 'Configuration validation failed'}
    
    # Override configuration with command line arguments
    if models:
        config['models'] = models
    if output_dir:
        config['evaluation']['output_dir'] = output_dir
    if max_samples:
        config['evaluation']['max_samples'] = max_samples
    
    # Setup logging
    final_output_dir = config.get('evaluation', {}).get('output_dir', 'evaluation_results')
    setup_logging(final_output_dir, verbose=False)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting KVQA evaluation framework")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Models: {config['models']}")
    logger.info(f"Output directory: {final_output_dir}")
    
    # Check dataset file
    if not os.path.exists(dataset_path):
        error_msg = f"Dataset file not found: {dataset_path}"
        logger.error(error_msg)
        return {'error': error_msg}
    
    # Run evaluation for each model
    all_results = []
    
    for model_name in config['models']:
        result = await run_single_model_evaluation(
            model_name=model_name,
            config=config,
            dataset_path=dataset_path,
            output_dir=final_output_dir
        )
        all_results.append(result)
    
    # Generate summary
    successful_models = [r for r in all_results if r['success']]
    failed_models = [r for r in all_results if not r['success']]
    
    summary = {
        'total_models': len(config['models']),
        'successful_models': len(successful_models),
        'failed_models': len(failed_models),
        'models_results': all_results,
        'output_directory': final_output_dir
    }
    
    logger.info(f"Evaluation completed. Success: {len(successful_models)}/{len(config['models'])} models")
    
    return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='KVQA Evaluation Framework')
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config_template.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset file path'
    )
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        nargs='+',
        help='Models to evaluate (override config)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (override config)'
    )
    
    parser.add_argument(
        '--max-samples', '-s',
        type=int,
        help='Maximum number of samples to evaluate'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Please copy config/config_template.yaml and customize it.")
        sys.exit(1)
    
    # Run evaluation
    try:
        summary = asyncio.run(run_evaluation(
            config_path=args.config,
            dataset_path=args.dataset,
            models=args.models,
            output_dir=args.output,
            max_samples=args.max_samples
        ))
        
        if 'error' in summary:
            print(f"Error: {summary['error']}")
            sys.exit(1)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total models: {summary['total_models']}")
        print(f"Successful: {summary['successful_models']}")
        print(f"Failed: {summary['failed_models']}")
        print(f"Output directory: {summary['output_directory']}")
        
        if summary['failed_models'] > 0:
            print("\nFailed models:")
            for result in summary['models_results']:
                if not result['success']:
                    print(f"  - {result['model']}: {result['error']}")
        
        print("\nEvaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
