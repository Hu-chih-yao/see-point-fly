import os
import base64
from google import genai
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from dotenv import load_dotenv
from typing import List

class VLMClient:
    """
    Unified VLM API client that supports both Gemini and OpenAI providers
    Handles API initialization, configuration, and response processing
    """

    def __init__(self, api_provider: str, model_name: str):
        """
        Initialize VLM client with specified provider and model

        Args:
            api_provider (str): Either "gemini" or "openai"
            model_name (str): Specific model name to use
        """
        self.api_provider = api_provider.lower()
        self.model_name = model_name

        # Load environment variables
        load_dotenv()

        # Initialize the appropriate client
        if self.api_provider == "openai":
            self._init_openai_client()
        elif self.api_provider == "gemini":
            self._init_gemini_client()
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")

    def _init_gemini_client(self):
        """Initialize Gemini client"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=api_key)

        # Store generation config for later use
        self.generation_config = genai.types.GenerateContentConfig(
            temperature=0.4,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )

    def _init_openai_client(self):
        """Initialize OpenAI client with configurable base URL"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        base_url = os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')

        self.openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate_response(self, prompt: str, image) -> str:
        """
        Generate response from VLM with image and text prompt

        Args:
            prompt (str): Text prompt for the model
            image: Image data (numpy array)

        Returns:
            str: Raw response text from the model
        """
        # Encode image to base64
        import cv2

        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        if self.api_provider == "openai":
            return self._get_openai_response(prompt, encoded_image)
        else:
            return self._get_gemini_response(prompt, encoded_image)

    def _get_gemini_response(self, prompt: str, encoded_image: str) -> str:
        """Get response from Gemini API"""
        import base64
        import io
        from PIL import Image

        # Decode base64 string to bytes and convert to PIL Image
        image_bytes = base64.b64decode(encoded_image)
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Use PIL Image directly as the migration guide suggests
        contents = [prompt, pil_image]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self.generation_config
        )
        return response.text or ""

    def _get_openai_response(self, prompt: str, encoded_image: str) -> str:
        """Get response from OpenAI API with image"""
        messages: List[ChatCompletionUserMessageParam] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]

        response = self.openai_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://spf-web.pages.dev",
                "X-Title": "See, Point, Fly"
            },
            model=self.model_name,
            messages=messages,
            temperature=0.4,
            max_tokens=8192
        )

        return response.choices[0].message.content or ""

    @staticmethod
    def clean_response_text(response_text: str) -> str:
        """Clean response text from markdown formatting"""
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        return response_text
