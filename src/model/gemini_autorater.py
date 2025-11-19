import os
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class GeminiAutorater:
    """
    Gemini-based autorater for hallucination detection.
    Uses Google's Gemini 2.0 Flash (or Pro) to evaluate if a generated answer is correct.
    """
    
    # Evaluation prompt inspired by SimpleQA-verified paper
    EVALUATION_PROMPT = """You are an expert factual accuracy evaluator. Your task is to determine if a model's answer to a question is correct.

Question: {question}

Gold Answer (correct answer): {gold_answer}

Model's Answer: {generated_answer}

Instructions:
1. Compare the model's answer with the gold answer
2. The model's answer is CORRECT if it contains the essential information from the gold answer (even if phrased differently)
3. The model's answer is INCORRECT (hallucination) if:
   - It contradicts the gold answer
   - It provides different factual information
   - It doesn't answer the question
   - It says "I don't know" or refuses to answer

Respond with ONLY one word:
- "CORRECT" if the model's answer matches the gold answer
- "INCORRECT" if the model's answer is wrong or hallucinates

Your evaluation:"""

    def __init__(self, model_name: str = "Gemini 2.0 Flash-Lite", api_key: Optional[str] = None):
        """
        Initialize the Gemini autorater.
        
        Args:
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
                       Options: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env variable
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
                )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        print(f"Initialized GeminiAutorater with model: {model_name}")
        # Generation config for deterministic output
        self.generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 10,  # We only need "CORRECT" or "INCORRECT"
        }
    
    
    def evaluate(self, question: str, gold_answer: str, generated_answer: str) -> dict:
        """
        Evaluate if the generated answer is correct using Gemini.
        
        Args:
            question: The question asked
            gold_answer: The correct answer
            generated_answer: The model's generated answer
            
        Returns:
            dict with keys:
                - is_hallucination: bool (True if incorrect, False if correct)
                - gemini_response: str (raw response from Gemini)
                - confidence: str (CORRECT or INCORRECT)
        """
        prompt = self.EVALUATION_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            gemini_response = response.text.strip().upper()
            
            # Parse response
            is_hallucination = "INCORRECT" in gemini_response
            confidence = "INCORRECT" if is_hallucination else "CORRECT"
            
            return {
                "is_hallucination": is_hallucination,
                "gemini_response": gemini_response,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Fallback to simple substring matching
            is_hallucination = gold_answer.lower().strip() not in generated_answer.lower().strip()
            return {
                "is_hallucination": is_hallucination,
                "gemini_response": f"ERROR: {str(e)}",
                "confidence": "FALLBACK"
            }
    
    
    def batch_evaluate(self, examples: list) -> list:
        """
        Evaluate multiple examples.
        
        Args:
            examples: List of dicts with keys: question, gold_answer, generated_answer
            
        Returns:
            List of evaluation results
        """
        results = []
        for example in examples:
            result = self.evaluate(
                question=example["question"],
                gold_answer=example["gold_answer"],
                generated_answer=example["generated_answer"]
            )
            result["instance_id"] = example.get("instance_id", None)
            results.append(result)
        
        return results
