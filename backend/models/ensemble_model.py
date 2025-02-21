import numpy as np

def ensemble_predictions(predictions: dict, weights: dict = None) -> float:
    """
    Combines predictions from multiple models using a weighted average.

    Parameters:
        predictions (dict): A dictionary where keys are model names and values are their predictions.
                            For example:
                              {
                                  "score_model": 110.0,
                                  "player_projection": 112.5,
                                  "quarter_analysis": 109.0
                              }
        weights (dict, optional): A dictionary with weights for each model.
                                  If not provided, equal weights are used.

    Returns:
        float: The final ensemble prediction as a weighted average.
    """
    if not predictions:
        raise ValueError("No predictions provided for ensemble modeling.")

    # Use equal weights if none are provided
    if weights is None:
        num_models = len(predictions)
        weights = {model: 1/num_models for model in predictions}

    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {model: weight / total_weight for model, weight in weights.items()}

    final_prediction = sum(normalized_weights[model] * pred for model, pred in predictions.items())
    return final_prediction

if __name__ == '__main__':
    # Example usage:
    # Assume we have three model predictions for the final home score.
    model_preds = {
        "score_model": 110.0,
        "player_projection": 112.5,
        "quarter_analysis": 109.0
    }
    
    # Optionally, define custom weights (if you want to favor a particular model)
    custom_weights = {
        "score_model": 0.5,
        "player_projection": 0.3,
        "quarter_analysis": 0.2
    }
    
    # Ensemble with equal weights
    ensemble_eq = ensemble_predictions(model_preds)
    print(f"Ensemble prediction with equal weights: {ensemble_eq:.2f}")
    
    # Ensemble with custom weights
    ensemble_custom = ensemble_predictions(model_preds, weights=custom_weights)
    print(f"Ensemble prediction with custom weights: {ensemble_custom:.2f}")
