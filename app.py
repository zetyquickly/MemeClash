import gradio as gr
import requests
import json
import re
from typing import List, Tuple, Optional
import random
import fal_client
import PIL
from dotenv import load_dotenv
import os

load_dotenv()

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
        self.conversation_state = {
            "waiting_for_characters": True, 
            "characters": [],
            "images_loaded": False,
            "waiting_for_confirmation": False
        }

    def generate_battle_scene(self, characters: List[str]) -> str:
        """Generate a battle scene between two characters"""
    

        FAL_KEY = "2c39734b-d428-401a-8f20-18d44ec7d7d7:e7cf9e41ead942f5314c048554639f64"

        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(log["message"])

        result = fal_client.subscribe(
            "fal-ai/nano-banana-edit",
            arguments={
                {
                "prompt": "imagine the characters in the first image and the chracters in second image in a fighting game scene",
                "image_urls": [
                    characters[0],
                    characters[1]
                ],
                "num_images": 1,
                "output_format": "jpeg"
}
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        return result.images[0]["url"]
    
    def process_message(self, message: str, history: List) -> Tuple[str, List, Optional[str], Optional[str]]:
        """Process user message and return response with images"""
        
        message_lower = message.lower().strip()
        
        # Check if user is responding to confirmation prompt
        if self.conversation_state["waiting_for_confirmation"]:
            if any(word in message_lower for word in ["yes", "continue", "fight", "generate", "go", "proceed", "start"]):
                response = "🚀 **Generating Epic Battle Scene!**\n\n"
                response += f"**{self.conversation_state['characters'][0].title()} VS {self.conversation_state['characters'][1].title()}**\n\n"
                response += "⚔️ The battle is about to begin!\n"
                response += "🎬 Creating cinematic fight sequence...\n"
                response += "💥 Adding special effects...\n\n"
                response += "🔧 **Battle generation feature coming soon!**\n"
                response += "This will create an epic animated battle between your chosen fighters!"
                
                # Reset state for new battle
                self.conversation_state = {
                    "waiting_for_characters": True,
                    "characters": [],
                    "images_loaded": False,
                    "waiting_for_confirmation": False
                }
                
                history.append([message, response])
                return "", history, None, None
                
            elif any(word in message_lower for word in ["no", "change", "different", "other", "new"]):
                response = "🔄 **Choose New Fighters!**\n\n"
                response += "No problem! Let's pick different characters.\n\n"
                response += self.get_help_message()
                
                # Reset state for new character selection
                self.conversation_state = {
                    "waiting_for_characters": True,
                    "characters": [],
                    "images_loaded": False,
                    "waiting_for_confirmation": False
                }
                
                history.append([message, response])
                return "", history, None, None
            else:
                # User didn't give clear yes/no, ask again
                response = "🤔 **Please choose:**\n\n"
                response += "• Type **'YES'** or **'CONTINUE'** to generate the fight\n"
                response += "• Type **'NO'** or **'CHANGE'** to pick different characters\n\n"
                response += f"Current fighters: **{self.conversation_state['characters'][0].title()}** vs **{self.conversation_state['characters'][1].title()}**"
                
                history.append([message, response])
                return "", history, None, None
        
        # Extract characters from message
        characters = self.extractor.extract_characters(message)
        
        if len(characters) == 2:
            # Found two characters, search for images
            char1, char2 = characters
            self.conversation_state["characters"] = [char1, char2]
            
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
            if img1_url and img2_url:
                final_response = f"✅ **Images Found!**\n\n"
                final_response += f"**Fighter 1:** {char1.title()}\n"
                final_response += f"**Fighter 2:** {char2.title()}\n\n"
                final_response += "🖼️ Both character images loaded successfully!\n\n"
                final_response += "**What would you like to do?**\n"
                final_response += "• Type **'YES'** or **'CONTINUE'** to generate the epic battle!\n"
                final_response += "• Type **'NO'** or **'CHANGE'** to choose different characters"
                
                # Set state to wait for confirmation
                self.conversation_state["waiting_for_confirmation"] = True
                self.conversation_state["images_loaded"] = True
                
            elif img1_url or img2_url:
                final_response = f"⚠️ **Partial Success**\n\n"
                final_response += f"**Fighter 1:** {char1.title()}\n"
                final_response += f"**Fighter 2:** {char2.title()}\n\n"
                final_response += "Found image for one fighter, using placeholder for the other.\n\n"
                final_response += "**What would you like to do?**\n"
                final_response += "• Type **'YES'** or **'CONTINUE'** to proceed with available images\n"
                final_response += "• Type **'NO'** or **'CHANGE'** to try different characters"
                
                # Set state to wait for confirmation
                self.conversation_state["waiting_for_confirmation"] = True
                self.conversation_state["images_loaded"] = True
                
            else:
                final_response = f"❌ **No Images Found**\n\n"
                final_response += f"Couldn't find images for **{char1.title()}** or **{char2.title()}**.\n\n"
                final_response += "**Suggestions:**\n"
                final_response += "• Try more popular characters (Superman, Goku, Batman)\n"
                final_response += "• Use full character names (Spider-Man instead of Spider)\n"
                final_response += "• Try different character combinations\n\n"
                final_response += "Choose new fighters or try again with different names!"
                
                # Reset state since no images found
                self.conversation_state = {
                    "waiting_for_characters": True,
                    "characters": [],
                    "images_loaded": False,
                    "waiting_for_confirmation": False
                }
            
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
    API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    searcher = GoogleImageSearcher(API_KEY, SEARCH_ENGINE_ID)
    bot = ClashOfMemesBot(searcher)
else:
    # For regular Python execution
    API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
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