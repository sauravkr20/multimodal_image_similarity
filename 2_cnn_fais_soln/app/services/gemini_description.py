from google import genai
from google.genai.types import Blob, Part, Content, GenerateContentConfig
from typing import Optional

class GeminiDescriptionService:
    def __init__(self, gemini_api_key: Optional[str] = None):
        if gemini_api_key:
            self.client = genai.Client(api_key=gemini_api_key)
        else:
            self.client = genai.Client()
        self.model_name = "gemini-2.0-flash-exp"

    async def generate_description(self, image_bytes: bytes) -> Optional[str]:
        try:
            prompt_text = """
                Describe the jewelry in the image. Provide the description in a structured format using the following attributes, if present:

                - Category: (e.g., Pendant, Earrings, Ring)
                - Sub Category: (e.g., Hearts, Statement, Solitaire)
                - Style: (e.g., Office, Traditional, Modern)
                - Stone: (e.g., Zircon, Diamond, Pearl)
                - Stone Color: (e.g., Transparent, White, Green)
                - Stone Shape: (e.g., Round, Oval, Square)
                - Stone Setting: (e.g., Prong, Bezel, Pave)

                Format the description as a comma-separated string of "Attribute: Value" pairs. If an attribute is not clearly identifiable in the image, omit it. Do not include other properties.
                """

            parts = [
                Part(text=prompt_text),
                Part(inline_data=Blob(mime_type="image/jpeg", data=image_bytes))
            ]

            content = Content(parts=parts)
            

            response = self.client.models.generate_content(
                model = self.model_name,
                contents=[content]
            )

            # print(response)

            if response.candidates and len(response.candidates) > 0:
                # Extract text from the first part of the content of the first candidate
                generated_text = response.candidates[0].content.parts[0].text
                return generated_text.strip()

            print("Gemini: no description returned")
            return None


        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
