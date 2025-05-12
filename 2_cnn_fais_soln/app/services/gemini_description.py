from google import genai
from google.genai.types import Blob, Part, Content, GenerateContentConfig
from typing import Optional, List

class GeminiDescriptionService:
    def __init__(self, gemini_api_key: Optional[str] = None):
        if gemini_api_key:
            self.client = genai.Client(api_key=gemini_api_key)
        else:
            self.client = genai.Client()
        self.model_name = "gemini-2.0-flash-exp"

    async def generate_description(self, image_bytes_list: List[bytes]) -> Optional[str]:
        try:
            prompt_text = """
                Describe the jewelry in the image. Provide the description in a structured format using the following attributes, if present:

                - Category: (e.g., Anklets, Bracelet, Brooch, Chains, Coin, Cufflinks, Earrings, Haath Phool, Idol, Maang Tika, Mens Earring, Pendant, Rings, Toe Rings)
                - Sub Category: (e.g., Bangles, Beads, Black/ Evil Eye, Black/ Evil eye, Braidal, Braidals, Bridal, Cocktails, Bridals, Chains, Chains, Evil Eye, Chains, Solitaire, Charms, Classics, Clip Ons, Couple, Cuffs, Danglers, Double Layered Necklace, Drops, Evil Eye, Evil Eye, Mangalsutra, Evil Eye, Slider, Evil Eyes, Evil eye, Fashion, Flexible, Flexibles, Hearts, Hoops, Huggies, J Studs, Jhumkis, Lariat Necklace, Mangalsutra, Mangalsutras, Mens, Mismatch, Necklace, Not Found, Open Rings, Pendant, Religious, Slider, Solitaire, Spiritual, Statement, Station Necklace, Studs, Sui dhaga, Tennis, Thread, Triple Layered Necklace, Venkys, bracelet, danglers, women ring)
                - Style: (e.g., Everyday, Not Found, Office, Party, Party , Traditional, Wedding)
                - Stone: (e.g., Bead, Chalcedony, Colored Stone, Colored Stone, Zircon, Colored Zircon, Cultured Polki, Enamel, MOP, Marcasite, Moon Stone, Moonstone, Morganite, No Stone, None, Zircon, Not Found, Pearl, Pearl, Zircon, Rose Cut Polki, Turquoise, Zircon, Zircon, Colored Stone, Zircon, Colored Zircon, Zircon, Pearl, Zircon, Rose Cut Polki, colored zircon, enamel, pearl, zircon, zircons)
                - Color: (e.g., Black Rhodium, Dual Color, Gold, Gold, Rose Gold, Mulit - Color, Multi Color, Multi-color, Multicolor, Oxidised Silver, Rose Gold, SIlver, Silver, Silver, Gold, Silver, Oxidised Silver, Silver, Rose Gold)
                - Stone Shape: (e.g., Round, Oval, Square)
                - Stone Setting: (e.g., Prong, Bezel, Pave)
                - detailed design description: (e.g., Design description)
                - motif: Animal, Animals, Bird, Bow, Butterfly, CLASSIC, Chain, Chakra, Couple ring, Crown, Drink, Drop, Evil eye, FLORAL, Filigree, Fish, Geometric, HEART, Infinity, Insect, Interlocking, Knot, LEAF, Lotus, Mandala, Mesh, Moon, Mushroon, Music, Nature, Not Found, Paisley, Paw, Peacock, Santa, Shell, Snake, Snowflake, Snowflakes, Snowman, Solitaire, Spiritual, Star, Sun, Tree of Life, Tree of life, Unicorn, Vanki, Vehicle, Wave, Weave, Wedding, Wing, Wings, architecture/temple, bird, classic, crown, fish, fruit, nature, peacock

                If an attribute has more than one applicable value, list the values separated by commas (e.g., "Category: Pendant, Earrings").

                Format the description as a semicolon-separated list of "Attribute: Value" pairs, in the order listed above. 
                
                Example output:
                Category: Pendant, Earrings; Sub Category: Hearts, Statement; Style: Traditional; Stone: Diamond; Stone Color: White; Stone Shape: Round; Stone Setting: Prong; Detailed design description: Delicate heart-shaped pendant with intricate filigree.

                Do not include any other properties or extra text.

                """

            parts = [
                Part(text=prompt_text)
                # Part(inline_data=Blob(mime_type="image/jpeg", data=image_bytes))

            ]


            parts.extend([Part(inline_data=Blob(mime_type="image/jpeg", data=image_bytes)) for image_bytes in image_bytes_list])


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
