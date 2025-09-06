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
        """Search for character image and return the first result URL with resolution > 250x250"""
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

            print(f"Searching for: {character_name}")
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                # Filter images with resolution > 250x250
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
                    else:
                        # If no size info, include anyway
                        valid_images.append(item)
                        
                if valid_images:
                    random_item = random.choice(valid_images)
                    return random_item['link']
                elif data['items']:
                    # Fallback to any image if no size filtering worked
                    return random.choice(data['items'])['link']
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
        self.reset_conversation_state()
        # Set up FAL client
        self.fal_key = os.getenv("FAL_KEY")
        if not self.fal_key:
            raise ValueError("FAL_KEY not found in environment variables")

    def reset_conversation_state(self):
        """Reset the conversation state to initial values"""
        self.conversation_state = {
            "waiting_for_characters": True, 
            "characters": [],
            "character_images": [],
            "images_loaded": False,
            "waiting_for_confirmation": False
        }

    def generate_battle_scene(self, image_urls: List[str], character_names: List[str]) -> str:
        """Generate a battle scene between two characters using their images"""
        try:
            print(f"Generating battle scene for {character_names}")
            print(f"Using image URLs: {image_urls}")
            
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(log["message"])

            # Create a dynamic prompt based on character names
            prompt = f"Epic battle scene between {character_names[0]} and {character_names[1]} in a fighting game arena. Dynamic action poses, special effects, energy blasts, dramatic lighting, cinematic composition, high quality anime/comic book style"
            
            result = fal_client.subscribe(
                "fal-ai/nano-banana/edit",
                arguments={
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "num_images": 1,
                    "output_format": "jpeg",
                    "seed": random.randint(1, 10000)
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            
            if result and result["images"] and len(result["images"]) > 0:
                return result["images"][0]["url"]
            else:
                print("No images returned from FAL API")
                return None
                
        except Exception as e:
            print(f"Error generating battle scene: {e}")
            return None
    
    def process_message(self, message: str, history: List) -> Tuple[str, List, Optional[str], Optional[str], Optional[str]]:
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
                response += "⏳ **Please wait while the AI generates your battle scene...**"
                
                # Add to history first
                history.append([message, response])
                
                # Generate battle scene
                battle_image_url = None
                if len(self.conversation_state["character_images"]) >= 2:
                    battle_image_url = self.generate_battle_scene(
                        self.conversation_state["character_images"],
                        self.conversation_state["characters"]
                    )
                
                if battle_image_url:
                    final_response = "🎉 **EPIC BATTLE GENERATED!**\n\n"
                    final_response += f"**{self.conversation_state['characters'][0].title()} VS {self.conversation_state['characters'][1].title()}**\n\n"
                    final_response += "✨ Your cinematic battle scene is ready!\n"
                    final_response += "🔥 Witness the ultimate clash!\n\n"
                    final_response += "Want another battle? Just tell me two new characters!"
                else:
                    final_response = "❌ **Battle Generation Failed**\n\n"
                    final_response += "Sorry, there was an issue generating the battle scene.\n"
                    final_response += "This could be due to:\n"
                    final_response += "• API limitations\n"
                    final_response += "• Image processing issues\n"
                    final_response += "• Network connectivity\n\n"
                    final_response += "Try again with the same or different characters!"
                
                # Update the last message in history
                history[-1][1] = final_response
                
                # Reset state for new battle
                self.conversation_state = {
                    "waiting_for_characters": True,
                    "characters": [],
                    "character_images": [],
                    "images_loaded": False,
                    "waiting_for_confirmation": False
                }
                
                # Safely return character images - they should be None after reset
                return "", history, None, None, battle_image_url
                
            elif any(word in message_lower for word in ["no", "change", "different", "other", "new"]):
                response = "🔄 **Choose New Fighters!**\n\n"
                response += "No problem! Let's pick different characters.\n\n"
                response += self.get_help_message()
                
                # Reset state for new character selection
                self.conversation_state = {
                    "waiting_for_characters": True,
                    "characters": [],
                    "character_images": [],
                    "images_loaded": False,
                    "waiting_for_confirmation": False
                }
                
                history.append([message, response])
                return "", history, None, None, None
            else:
                # User didn't give clear yes/no, ask again
                response = "🤔 **Please choose:**\n\n"
                response += "• Type **'YES'** or **'CONTINUE'** to generate the fight\n"
                response += "• Type **'NO'** or **'CHANGE'** to pick different characters\n\n"
                response += f"Current fighters: **{self.conversation_state['characters'][0].title()}** vs **{self.conversation_state['characters'][1].title()}**"
                
                history.append([message, response])
                return "", history, None, None, None
        
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
            
            # Store image URLs
            self.conversation_state["character_images"] = [img1_url, img2_url]
            
            # Update response based on search results
            if img1_url and img2_url:
                final_response = f"✅ **Images Found!**\n\n"
                final_response += f"**Fighter 1:** {char1.title()}\n"
                final_response += f"**Fighter 2:** {char2.title()}\n\n"
                final_response += "🖼️ Both character images loaded successfully!\n\n"
                final_response += "**Ready to generate the epic battle scene?**\n"
                final_response += "• Type **'YES'** or **'CONTINUE'** to create the battle!\n"
                final_response += "• Type **'NO'** or **'CHANGE'** to choose different characters"
                
                # Set state to wait for confirmation
                self.conversation_state["waiting_for_confirmation"] = True
                self.conversation_state["images_loaded"] = True
                
            elif img1_url or img2_url:
                final_response = f"⚠️ **Partial Success**\n\n"
                final_response += f"**Fighter 1:** {char1.title()}\n"
                final_response += f"**Fighter 2:** {char2.title()}\n\n"
                final_response += "Found image for one fighter, but couldn't find the other.\n"
                final_response += "The battle scene may not work well with only one image.\n\n"
                final_response += "**What would you like to do?**\n"
                final_response += "• Type **'YES'** or **'CONTINUE'** to try anyway\n"
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
                    "character_images": [],
                    "images_loaded": False,
                    "waiting_for_confirmation": False
                }
            
            # Update the last message in history
            history[-1][1] = final_response
            
            return "", history, img1_url, img2_url, None
            
        elif len(characters) == 1:
            # Only found one character
            response = f"I found **{characters[0].title()}**, but I need TWO fighters!\n\n"
            response += "Try something like:\n"
            response += f"• '{characters[0]} vs Goku'\n"
            response += f"• '{characters[0]} fights Batman'\n"
            response += f"• 'Battle between {characters[0]} and Superman'"
            
            history.append([message, response])
            return "", history, None, None, None
            
        else:
            # No clear characters found
            response = self.get_help_message()
            history.append([message, response])
            return "", history, None, None, None
    
    def get_help_message(self) -> str:
        """Return help message for character input"""
        return """🤖 **Welcome to Clash of Memes!**

I need you to tell me which two characters should fight! 

**Try formats like:**
• "Goku vs Superman"
• "Batman fights Spider-Man"
• "Battle between Pikachu and Sonic"
• "Terminator versus PowerPuff Girls"

**Popular characters that work well:**
• Superheroes: Superman, Batman, Spider-Man, Iron Man
• Anime: Goku, Naruto, Luffy, Saitama
• Games: Mario, Sonic, Link, Master Chief
• Movies: Terminator, John Wick, Neo, Darth Vader

Just tell me who should fight! 🥊"""

# Initialize components
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

if not API_KEY or not SEARCH_ENGINE_ID:
    print("Warning: Google API credentials not found in environment variables")

searcher = GoogleImageSearcher(API_KEY, SEARCH_ENGINE_ID) if API_KEY and SEARCH_ENGINE_ID else None
bot = ClashOfMemesBot(searcher) if searcher else None

def chat_interface(message, history):
    if not bot:
        return "", history + [[message, "❌ Bot not initialized. Please check your API credentials."]], None, None, None
        
    if not message.strip():
        return "", history, None, None, None
        
    return bot.process_message(message, history)

# Create Gradio interface
with gr.Blocks(title="Clash of Memes", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🥊 Clash of Memes - Battle Scene Generator
    ### Tell me which two characters should fight, and I'll create an epic battle scene!
    
    **How it works:**
    1. Tell me two characters (e.g., "Goku vs Superman")
    2. I'll find their images
    3. Generate an epic AI battle scene!
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
            gr.Markdown("### Character Images")
            with gr.Row():
                img1 = gr.Image(label="Fighter 1", height=200)
                img2 = gr.Image(label="Fighter 2", height=200)
            
            gr.Markdown("### Battle Scene")
            battle_img = gr.Image(label="Epic Battle", height=300)
    
    # Event handlers
    def submit_message(message, history):
        response, new_history, image1, image2, battle_image = chat_interface(message, history)
        return "", new_history, image1, image2, battle_image
    
    submit.click(
        submit_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, img1, img2, battle_img]
    )
    
    msg.submit(
        submit_message,
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, img1, img2, battle_img]
    )
    
    # Initialize with help message
    demo.load(
        lambda: (
            bot.reset_conversation_state() or [(None, bot.get_help_message())] if bot else [(None, "❌ Bot not initialized")], 
            None, None, None
        ),
        outputs=[chatbot, img1, img2, battle_img]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)