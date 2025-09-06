import gradio as gr
import requests
import json
import re
from typing import List, Tuple, Optional
import random

class GoogleImageSearcher:
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search_character_image(self, character_name: str) -> Optional[str]:
        """Search for character image and return the first result URL with resolution > 500x500"""
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': f'{character_name} -text -logo -watermark',
                'searchType': 'image',
                'num': 1,
                'imgSize': 'large',
                'imgColorType': 'color',
                'safe': 'active',
                'rights': 'cc_publicdomain,cc_attribute,cc_sharealike'
            }

            print(params)
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                # Filter images with resolution > 500x500
                valid_images = []
                for item in data['items']:
                    image_info = item.get('image', {})
                    width = image_info.get('width')
                    height = image_info.get('height')
                    if width is not None and height is not None:
                        try:
                            if int(width) > 250 and int(height) > 250:
                                valid_images.append(item)
                        except Exception:
                            continue
                if valid_images:
                    random_item = random.choice(valid_images)
                    return random_item['link']
            return None
            
        except Exception as e:
            print(f"Error searching for {character_name}: {e}")
            return None

class CharacterExtractor:
    """Extract character names from user messages"""
    
    @staticmethod
    def extract_characters(message: str) -> List[str]:
        """Extract character names from user message"""
        # Common patterns for character requests
        patterns = [
            r"(.+?)\s+(?:vs|versus|against|fights?)\s+(.+?)(?:$|!|\?)",
            r"(.+?)\s+and\s+(.+?)(?:\s+fighting|battle|fight|$|!|\?)",
            r"battle\s+between\s+(.+?)\s+and\s+(.+?)(?:$|!|\?)",
            r"fight\s+with\s+(.+?)\s+and\s+(.+?)(?:$|!|\?)",
            r"(.+?)\s*,\s*(.+?)(?:\s+battle|fight|$|!|\?)"
        ]
        
        message = message.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                char1 = match.group(1).strip()
                char2 = match.group(2).strip()
                return [char1, char2]
        
        if " vs " in message:
            parts = message.split(" vs ")
            if len(parts) == 2:
                return [parts[0].strip(), parts[1].strip()]
        
        if " versus " in message:
            parts = message.split(" versus ")
            if len(parts) == 2:
                return [parts[0].strip(), parts[1].strip()]
        
        return []

class ClashOfMemesBot:
    def __init__(self, google_searcher: GoogleImageSearcher):
        self.searcher = google_searcher
        self.extractor = CharacterExtractor()
        self.conversation_state = {"waiting_for_characters": True, "characters": []}
    
    def process_message(self, message: str, history: List) -> Tuple[str, List, Optional[str], Optional[str]]:
        """Process user message and return response with images"""
        
        # Extract characters from message
        characters = self.extractor.extract_characters(message)
        
        if len(characters) == 2:
            # Found two characters, search for images
            char1, char2 = characters
            
            response = f"🥊 **Battle Setup Detected!**\n\n"
            response += f"**Fighter 1:** {char1.title()}\n"
            response += f"**Fighter 2:** {char2.title()}\n\n"
            response += "Searching for character images... 🔍"
            
            # Update history
            history.append([message, response])
            
            # Search for images
            img1_url = self.searcher.search_character_image(char1)
            img2_url = self.searcher.search_character_image(char2)
            
            # Update response based on search results
            final_response = f"🥊 **Battle Ready!**\n\n"
            final_response += f"**Fighter 1:** {char1.title()}\n"
            final_response += f"**Fighter 2:** {char2.title()}\n\n"
            
            if img1_url and img2_url:
                final_response += "✅ Found images for both fighters!\n"
                final_response += "Ready to generate epic battle? (Feature coming soon...)"
            elif img1_url or img2_url:
                final_response += "⚠️ Found image for one fighter, using placeholder for the other.\n"
                final_response += "Ready to generate battle with available images?"
            else:
                final_response += "❌ Couldn't find images for these characters.\n"
                final_response += "Try different character names or more specific descriptions."
            
            # Update the last message in history
            history[-1][1] = final_response
            
            return "", history, img1_url, img2_url
            
        elif len(characters) == 1:
            # Only found one character
            response = f"I found **{characters[0].title()}**, but I need TWO fighters!\n\n"
            response += "Try something like:\n"
            response += f"• '{characters[0]} vs Goku'\n"
            response += f"• '{characters[0]} fights Batman'\n"
            response += f"• 'Battle between {characters[0]} and Superman'"
            
            history.append([message, response])
            return "", history, None, None
            
        else:
            # No clear characters found
            response = self.get_help_message()
            history.append([message, response])
            return "", history, None, None
    
    def get_help_message(self) -> str:
        """Return help message for character input"""
        return """🤖 **Welcome to Clash of Memes!**

I need you to tell me which two characters should fight! 

**Try formats like:**
• "Goku vs Superman"
• "Batman fights Spider-Man"
• "Battle between Pikachu and Sonic"
• "Terminator versus PowerPuff Girls"

**Popular characters:**
• Superheroes: Superman, Batman, Spider-Man, Iron Man
• Anime: Goku, Naruto, Luffy, Saitama
• Games: Mario, Sonic, Link, Master Chief
• Movies: Terminator, John Wick, Neo, Darth Vader

Just tell me who should fight! 🥊"""

# Initialize components outside of NO_RELOAD block for faster development
# This will only run once when using gradio reload mode
if gr.NO_RELOAD:
    API_KEY = "AIzaSyCE22Nkh6T2cVJ7RBRlCiQLcgw3Bl_pJDY"
    SEARCH_ENGINE_ID = "b6861e6ec73d74167"
    searcher = GoogleImageSearcher(API_KEY, SEARCH_ENGINE_ID)
    bot = ClashOfMemesBot(searcher)
else:
    # For regular Python execution
    API_KEY = "AIzaSyCE22Nkh6T2cVJ7RBRlCiQLcgw3Bl_pJDY"
    SEARCH_ENGINE_ID = "b6861e6ec73d74167"
    searcher = GoogleImageSearcher(API_KEY, SEARCH_ENGINE_ID)
    bot = ClashOfMemesBot(searcher)

def chat_interface(message, history):
    if not message.strip():
        return "", history, None, None
        
    return bot.process_message(message, history)

# Create Gradio interface
with gr.Blocks(title="Clash of Memes", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🥊 Clash of Memes
    ### Tell me which two characters should fight, and I'll find their images!
    **🔥 Auto-reload enabled** - run with `gradio app.py` for instant updates!
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=400,
                show_label=False,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type: 'Goku vs Superman' or 'Batman fights Spider-Man'...",
                    container=False,
                    scale=7
                )
                submit = gr.Button("Send", scale=1, variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Fighter Images")
            with gr.Row():
                img1 = gr.Image(label="Fighter 1", height=200)
                img2 = gr.Image(label="Fighter 2", height=200)
    
    # Event handlers
    def submit_message(message, history):
        response, new_history, image1, image2 = chat_interface(message, history)
        return "", new_history, image1, image2
    
    submit.click(
        submit_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, img1, img2]
    )
    
    msg.submit(
        submit_message,
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, img1, img2]
    )
    
    # Initialize with help message
    demo.load(
        lambda: ([(None, bot.get_help_message())], None, None),
        outputs=[chatbot, img1, img2]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)