import os
from dotenv import load_dotenv
from together import Together
from src.utils.decorators import timer_decorator, log_decorator, error_handler


class TogetherAIIntegration:
    def __init__(self):
        load_dotenv()
        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.models = {
            "feature_generation": "togethercomputer/llama-2-70b-chat",
            "result_interpretation": "togethercomputer/llama-2-70b-chat"
        }

    def generate_features(self, data):
        """Genera nuevas características usando LLM."""
        prompt = f"Given the following customer data, suggest 3 new relevant features:\n{data.to_dict()}"
        response = self.client.chat.completions.create(
            model=self.models["feature_generation"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        # Aquí procesarías la respuesta para extraer las nuevas características
        return response.choices[0].message.content

    def interpret_results(self, prediction, probabilities, customer_data):
        """Interpreta los resultados de la predicción usando LLM."""
        prompt = f"""
        Given the following prediction and customer data, provide a detailed explanation:
        Prediction: {'Churn' if prediction == 1 else 'No Churn'}
        Churn Probability: {probabilities[1]:.2f}
        Customer Data: {customer_data.to_dict()}
        """
        response = self.client.chat.completions.create(
            model=self.models["result_interpretation"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content