"""Utility modules for the Trittention-Transformer project."""

from utils.data_utils import (
    ArithmeticDataset, 
    SequenceDataset, 
    load_dataset, 
    create_dataloaders,
    load_toy_dataset
)

from utils.evaluation_utils import (
    EvaluationResult,
    calculate_metrics,
    evaluate_model,
    compare_models,
    save_results,
    visualize_attention
)

from utils.visualization_utils import (
    visualize_attention_matrix,
    visualize_attention_comparisons,
    plot_training_history,
    plot_model_comparisons,
    plot_complexity_analysis,
    visualize_embeddings
)

__all__ = [
    # Data utils
    'ArithmeticDataset',
    'SequenceDataset',
    'load_dataset',
    'create_dataloaders',
    'load_toy_dataset',
    
    # Evaluation utils
    'EvaluationResult',
    'calculate_metrics',
    'evaluate_model',
    'compare_models',
    'save_results',
    'visualize_attention',
    
    # Visualization utils
    'visualize_attention_matrix',
    'visualize_attention_comparisons',
    'plot_training_history',
    'plot_model_comparisons',
    'plot_complexity_analysis',
    'visualize_embeddings'
]
