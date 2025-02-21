import os
import requests

# Retrieve the Cloudflare Workers AI endpoint from environment variables.
CLOUDFLARE_WORKER_AI_URL = os.getenv("CLOUDFLARE_WORKER_AI_URL")
if not CLOUDFLARE_WORKER_AI_URL:
    raise Exception("CLOUDFLARE_WORKER_AI_URL environment variable is not set")

def generate_game_summary(model_outputs: dict) -> str:
    """
    Generates a narrative game summary by calling the Cloudflare Workers AI endpoint.
    
    Parameters:
        model_outputs (dict): Outputs from various models (win probability, momentum shift, etc.)
    
    Returns:
        str: A generated narrative game summary.
    """
    # Construct the prompt using the model outputs.
    prompt = (
        "Generate a concise narrative summary for a basketball game with the following details:\n"
        f"Win Probability: {model_outputs.get('win_probability', 'N/A'):.2f}\n"
        f"Momentum Shift: {model_outputs.get('momentum_shift', 'N/A')}\n"
        f"Projected Margin: {model_outputs.get('projected_margin', 'N/A')}\n"
        f"Player Projection: {model_outputs.get('player_projection', 'N/A')}\n"
        "Narrative Summary:"
    )
    
    payload = {
        "prompt": prompt,
        "max_length": 150,  # Adjust parameters as needed
        "temperature": 0.7
    }
    
    try:
        response = requests.post(CLOUDFLARE_WORKER_AI_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        # Assume the response contains a key 'generated_text'
        summary_text = data.get("generated_text", "No summary generated.")
        return summary_text
    except Exception as e:
        return f"Error generating summary: {e}"

if __name__ == "__main__":
    # Example usage:
    example_model_outputs = {
        "win_probability": 0.82,
        "momentum_shift": 0.25,
        "projected_margin": 8,
        "player_projection": "Player X is expected to score 28 points and grab 7 rebounds."
    }
    summary = generate_game_summary(example_model_outputs)
    print("Generated Game Summary:\n", summary)
