import gradio as gr
import requests
import json
import re
from typing import List, Tuple, Optional
import random
import fal_client
from PIL import Image, ImageOps
from dotenv import load_dotenv
import os
import pickle
from pathlib import Path
import base64
import io

load_dotenv()

import requests
import random
from typing import Optional


import requests
import random
from typing import Optional

class ImageProcessor:
    """Handle image downloading, processing and base64 conversion"""
    
    @staticmethod
    def download_and_process_image(url: str, target_size: Tuple[int, int] = (1280, 720)) -> Optional[str]:
        """Download image, resize/pad to target size, and return as base64"""
        try:
            print(f"Downloading and processing image: {url}")
            
            # Headers to mimic a real browser and avoid 403 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Try multiple times with different approaches
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # First attempt: normal request with headers
                        response = requests.get(url, headers=headers, timeout=15)
                    elif attempt == 1:
                        # Second attempt: without custom headers
                        response = requests.get(url, timeout=15)
                    else:
                        # Third attempt: with session
                        session = requests.Session()
                        session.headers.update(headers)
                        response = session.get(url, timeout=15)
                    
                    response.raise_for_status()
                    break
                    
                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # Last attempt
                        raise e
                    continue
            
            # Check if we got actual image data
            if len(response.content) < 1024:  # Less than 1KB probably not a real image
                print(f"Response too small ({len(response.content)} bytes), probably not an image")
                return None
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                print(f"Content-Type '{content_type}' doesn't appear to be an image")
                # Still try to process it, might be a false negative
            
            # Open image with PIL
            try:
                image = Image.open(io.BytesIO(response.content))
            except Exception as e:
                print(f"Failed to open image with PIL: {e}")
                return None
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if len(image.split()) > 3 else None)
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Calculate scaling to fit within target size while maintaining aspect ratio
            original_width, original_height = image.size
            target_width, target_height = target_size
            
            # Calculate scale factor to fit image within target dimensions
            scale_factor = min(target_width / original_width, target_height / original_height)
            
            # Calculate new dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize image using high-quality resampling
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and black background
            final_image = Image.new('RGB', target_size, (0, 0, 0))
            
            # Calculate position to center the resized image
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Paste resized image onto black background
            final_image.paste(resized_image, (x_offset, y_offset))
            
            # Convert to base64
            buffer = io.BytesIO()
            final_image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            print(f"Successfully processed image: {original_width}x{original_height} -> {new_width}x{new_height} -> {target_width}x{target_height}")
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            print(f"Error processing image {url}: {e}")
            return None

class BattleCache:
    """Cache system for battle scenes, videos and character images"""
    
    def __init__(self, cache_file: str = "battle_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Handle backward compatibility with old cache format
                    if cache_data and isinstance(list(cache_data.values())[0], tuple):
                        print("Converting old cache format to new format...")
                        new_cache = {}
                        for key, value in cache_data.items():
                            if isinstance(value, tuple) and len(value) == 2:
                                battle_img, battle_video = value
                                new_cache[key] = {
                                    "character_images": [None, None],
                                    "battle_image": battle_img,
                                    "battle_video": battle_video
                                }
                        return new_cache
                    return cache_data
        except Exception as e:
            print(f"Error loading cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _make_key(self, characters: List[str]) -> str:
        """Create consistent cache key from character pair"""
        # Sort characters to ensure consistent key regardless of order
        sorted_chars = sorted([char.lower().strip() for char in characters])
        return tuple(sorted_chars)
    
    def get(self, characters: List[str]) -> Optional[dict]:
        """Get cached character images, battle scene and video URLs"""
        key = self._make_key(characters)
        cached_data = self.cache.get(key)
        if cached_data:
            # Ensure we return the expected structure
            return {
                "character_images": cached_data.get("character_images", [None, None]),
                "battle_image": cached_data.get("battle_image"),
                "battle_video": cached_data.get("battle_video")
            }
        return None
    
    def set_image(self, characters: List[str], character_images: List[Optional[str]], battle_img_url: str):
        """Cache character images and battle image"""
        key = self._make_key(characters)
        current = self.cache.get(key, {
            "character_images": [None, None],
            "battle_image": None,
            "battle_video": None
        })
        
        self.cache[key] = {
            "character_images": character_images[:2] if character_images else [None, None],  # Ensure we only store 2 images
            "battle_image": battle_img_url,
            "battle_video": current.get("battle_video")  # Keep existing video if any
        }
        self._save_cache()
        print(f"Cached character images and battle image for {characters}")
    
    def set_video(self, characters: List[str], character_images: List[Optional[str]], battle_img_url: str, battle_video_url: str):
        """Cache character images, battle image and video"""
        key = self._make_key(characters)
        self.cache[key] = {
            "character_images": character_images[:2] if character_images else [None, None],  # Ensure we only store 2 images
            "battle_image": battle_img_url,
            "battle_video": battle_video_url
        }
        self._save_cache()
        print(f"Cached character images, battle image and video for {characters}")
    
    def clear_cache(self):
        """Clear all cache"""
        self.cache = {}
        self._save_cache()
        print("Cache cleared")

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
                'q': f'{character_name} -text -watermark',
                'searchType': 'image',
                'num': 1,
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
        self.cache = BattleCache()
        self.image_processor = ImageProcessor()
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
            "waiting_for_confirmation": False,
            "battle_image_generated": False,
            "battle_image_url": None,
            "waiting_for_video_confirmation": False,
            "force_regenerate": False
        }

    def animate_battle_scene(self, image_url: str, character_names: List[str]) -> str:
        """Animate a battle scene between two characters using their images"""
        try:
            print(f"Generating battle video for {character_names}")
            
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(log["message"])

            # Create a dynamic prompt based on character names
            coin_flip = random.randint(0, 1)
            winner = character_names[0] if coin_flip == 0 else character_names[1]
            loser = character_names[1] if coin_flip == 0 else character_names[0]
            styles = ["mortal kombat style", "super smash bros style", "street fighter style", "tekken style", "comic book style", "pixel art style"]
            prompt = f"Epic battle scene between {winner} is defeating {loser} in a fighting game arena. In style of {random.choice(styles)}. Dynamic action poses, special effects, energy blasts, dramatic lighting, cinematic composition, high quality"

            result = fal_client.subscribe(
                "fal-ai/veo3/fast/image-to-video",
                arguments={
                    "prompt": prompt,
                    "image_url": image_url,
                    "seed": random.randint(1, 10000),
                    "generate_audio": False
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            
            if result and result.get("video") and result["video"].get("url"):
                return result["video"]["url"]
            else:
                print("No video returned from FAL API")
                return None
                
        except Exception as e:
            print(f"Error generating battle scene: {e}")
            return None

    def generate_battle_scene(self, image_urls: List[str], character_names: List[str]) -> str:
        """Generate a battle scene between two characters using their images"""
        try:
            print(f"Generating battle scene for {character_names}")
            print(f"Using image URLs: {image_urls}")
            
            # Download and process images to base64
            processed_images = []
            failed_urls = []
            
            for i, url in enumerate(image_urls):
                if url:
                    print(f"Processing image {i+1}/{len(image_urls)}: {character_names[i] if i < len(character_names) else 'Unknown'}")
                    base64_image = self.image_processor.download_and_process_image(url)
                    if base64_image:
                        processed_images.append(base64_image)
                        print(f"✅ Successfully processed image {i+1}/{len(image_urls)} for {character_names[i] if i < len(character_names) else 'Unknown'}")
                    else:
                        failed_urls.append((i, url))
                        print(f"❌ Failed to process image {i+1}/{len(image_urls)} for {character_names[i] if i < len(character_names) else 'Unknown'}")
                else:
                    print(f"❌ No URL provided for image {i+1}/{len(image_urls)}")
            
            # If we don't have enough images, try to proceed with what we have
            if len(processed_images) == 0:
                print("No images processed successfully")
                return None
            elif len(processed_images) == 1:
                # Duplicate the single image so we have 2
                print("Only 1 image processed, duplicating for nano-banana")
                processed_images.append(processed_images[0])
            
            print(f"Proceeding with {len(processed_images)} processed images")
            
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(log["message"])

            # Create a dynamic prompt based on character names
            styles = ["mortal kombat style", "super smash bros style", "street fighter style", "tekken style", "comic book style", "pixel art style"]
            prompt = f"Epic battle scene between {character_names[0]} and {character_names[1]} in a fighting game arena. In style of {random.choice(styles)}. Dynamic action poses, special effects, energy blasts, dramatic lighting, cinematic composition, high quality"
            
            result = fal_client.subscribe(
                "fal-ai/nano-banana/edit",
                arguments={
                    "prompt": prompt,
                    "image_urls": processed_images,  # Now using base64 images
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
    
    def process_message(self, message: str, history: List) -> Tuple[str, List, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Process user message and return response with images and video"""
        
        message_lower = message.lower().strip()
        
        # Check if user is responding to video confirmation prompt
        if self.conversation_state["waiting_for_video_confirmation"]:
            if any(word in message_lower for word in ["yes", "video", "animate", "generate", "go", "proceed", "start"]):
                response = "🎬 **Generating Battle Video!**\n\n"
                response += f"**{self.conversation_state['characters'][0].title()} VS {self.conversation_state['characters'][1].title()}**\n\n"
                response += "🎥 Creating animated battle sequence...\n"
                response += "✨ Adding motion and effects...\n"
                response += "🔥 Bringing the fight to life...\n\n"
                response += "⏳ **Please wait while the AI generates your battle video... (this may take a minute)**"
                
                # Store current images before resetting state
                current_battle_image = self.conversation_state.get("battle_image_url")
                current_char_images = self.conversation_state.get("character_images", [None, None])
                char1_img = current_char_images[0] if len(current_char_images) > 0 else None
                char2_img = current_char_images[1] if len(current_char_images) > 1 else None
                
                # Add to history first
                history.append([message, response])
                
                # Generate battle video
                battle_video_url = None
                if self.conversation_state["battle_image_url"]:
                    battle_video_url = self.animate_battle_scene(
                        self.conversation_state["battle_image_url"],
                        self.conversation_state["characters"]
                    )
                
                if battle_video_url:
                    # Cache both image and video with character images
                    self.cache.set_video(
                        self.conversation_state["characters"],
                        self.conversation_state["character_images"],
                        self.conversation_state["battle_image_url"],
                        battle_video_url
                    )
                    
                    final_response = "🎉 **EPIC BATTLE VIDEO READY!**\n\n"
                    final_response += f"**{self.conversation_state['characters'][0].title()} VS {self.conversation_state['characters'][1].title()}**\n\n"
                    final_response += "🎬 Your animated battle sequence is complete!\n"
                    final_response += "🔥 Watch the ultimate clash come to life!\n\n"
                    final_response += "Want another battle? Just tell me two new characters!"
                else:
                    final_response = "❌ **Video Generation Failed**\n\n"
                    final_response += "Sorry, there was an issue generating the battle video.\n"
                    final_response += "This could be due to:\n"
                    final_response += "• API limitations\n"
                    final_response += "• Video processing issues\n"
                    final_response += "• Network connectivity\n\n"
                    final_response += "The battle image is still available above!\n"
                    final_response += "Try again or choose different characters!"
                
                # Update the last message in history
                history[-1][1] = final_response
                
                # Reset state for new battle
                self.reset_conversation_state()
                
                return "", history, char1_img, char2_img, current_battle_image, battle_video_url
                
            elif any(word in message_lower for word in ["no", "skip", "new", "different", "next"]):
                response = "👍 **No problem!**\n\n"
                response += "Your epic battle image is ready above!\n\n"
                response += "Ready for a new battle? Just tell me two new characters to fight! 🥊"
                
                # Store current images before resetting state
                current_battle_image = self.conversation_state.get("battle_image_url")
                current_char_images = self.conversation_state.get("character_images", [None, None])
                char1_img = current_char_images[0] if len(current_char_images) > 0 else None
                char2_img = current_char_images[1] if len(current_char_images) > 1 else None
                
                # Reset state for new character selection
                self.reset_conversation_state()
                
                history.append([message, response])
                return "", history, char1_img, char2_img, current_battle_image, None
            else:
                # User didn't give clear yes/no, ask again
                response = "🤔 **Please choose:**\n\n"
                response += "• Type **'YES'** or **'VIDEO'** to generate an animated battle video\n"
                response += "• Type **'NO'** or **'SKIP'** to keep just the image and start a new battle\n\n"
                response += f"Current battle image: **{self.conversation_state['characters'][0].title()}** vs **{self.conversation_state['characters'][1].title()}**"
                
                # Preserve current images
                current_battle_image = self.conversation_state.get("battle_image_url")
                current_char_images = self.conversation_state.get("character_images", [None, None])
                char1_img = current_char_images[0] if len(current_char_images) > 0 else None
                char2_img = current_char_images[1] if len(current_char_images) > 1 else None
                
                history.append([message, response])
                return "", history, char1_img, char2_img, current_battle_image, None
        
        # Check if user is responding to confirmation prompt
        if self.conversation_state["waiting_for_confirmation"]:
            if any(word in message_lower for word in ["yes", "continue", "fight", "generate", "go", "proceed", "start"]):
                response = "🚀 **Generating Epic Battle Scene!**\n\n"
                response += f"**{self.conversation_state['characters'][0].title()} VS {self.conversation_state['characters'][1].title()}**\n\n"
                response += "⚔️ The battle is about to begin!\n"
                response += "📥 Downloading and processing character images...\n"
                response += "🔄 Trying multiple methods to bypass image blocks...\n"
                response += "🎬 Creating cinematic fight sequence...\n"
                response += "💥 Adding special effects...\n\n"
                response += "⏳ **Please wait while the AI generates your battle scene...**"
                
                # Add to history first
                history.append([message, response])
                
                # Generate battle scene
                battle_image_url = None
                if len(self.conversation_state["character_images"]) >= 1:  # Changed from >= 2
                    battle_image_url = self.generate_battle_scene(
                        self.conversation_state["character_images"],
                        self.conversation_state["characters"]
                    )
                
                if battle_image_url:
                    # Cache the image with character images
                    self.cache.set_image(
                        self.conversation_state["characters"], 
                        self.conversation_state["character_images"],
                        battle_image_url
                    )
                    
                    # Store the battle image URL for potential video generation
                    self.conversation_state["battle_image_url"] = battle_image_url
                    self.conversation_state["battle_image_generated"] = True
                    
                    final_response = "🎉 **EPIC BATTLE GENERATED!**\n\n"
                    final_response += f"**{self.conversation_state['characters'][0].title()} VS {self.conversation_state['characters'][1].title()}**\n\n"
                    final_response += "✨ Your cinematic battle scene is ready!\n"
                    final_response += "🔥 Witness the ultimate clash!\n\n"
                    
                    # Check if some images failed
                    if not all(self.conversation_state["character_images"]):
                        final_response += "⚠️ Note: Some character images couldn't be downloaded due to website restrictions, but we still created an epic battle!\n\n"
                    
                    final_response += "🎬 **Want to make it even more epic?**\n"
                    final_response += "• Type **'YES'** or **'VIDEO'** to generate an animated battle video!\n"
                    final_response += "• Type **'NO'** or **'NEW'** to start a new battle with different characters"
                    
                    # Set state to wait for video confirmation
                    self.conversation_state["waiting_for_confirmation"] = False
                    self.conversation_state["waiting_for_video_confirmation"] = True
                else:
                    final_response = "❌ **Battle Generation Failed**\n\n"
                    final_response += "Sorry, there was an issue generating the battle scene.\n"
                    final_response += "This could be due to:\n"
                    final_response += "• Image download failures (403 Forbidden errors)\n"
                    final_response += "• API limitations\n"
                    final_response += "• Image processing issues\n"
                    final_response += "• Network connectivity\n\n"
                    final_response += "💡 **Try:**\n"
                    final_response += "• Different character names\n"
                    final_response += "• More popular characters (they usually have better accessible images)\n"
                    final_response += "• Trying again (sometimes it works on retry)"
                    
                    # Reset state for new battle
                    self.reset_conversation_state()
                
                # Update the last message in history
                history[-1][1] = final_response
                
                return "", history, None, None, battle_image_url, None
                
            elif any(word in message_lower for word in ["force", "regenerate", "new", "fresh"]):
                # Force regenerate - skip cache
                self.conversation_state["force_regenerate"] = True
                
                response = "🔄 **Force Regenerating Epic Battle Scene!**\n\n"
                response += f"**{self.conversation_state['characters'][0].title()} VS {self.conversation_state['characters'][1].title()}**\n\n"
                response += "⚔️ Creating a fresh new battle scene!\n"
                response += "📥 Downloading and processing character images...\n"
                response += "🎬 Ignoring cached version...\n"
                response += "💥 Adding special effects...\n\n"
                response += "⏳ **Please wait while the AI generates your battle scene...**"
                
                # Add to history first
                history.append([message, response])
                
                # Generate battle scene (skip cache)
                battle_image_url = None
                if len(self.conversation_state["character_images"]) >= 1:  # Changed from >= 2
                    battle_image_url = self.generate_battle_scene(
                        self.conversation_state["character_images"],
                        self.conversation_state["characters"]
                    )
                
                if battle_image_url:
                    # Cache the new image
                    self.cache.set_image(self.conversation_state["characters"], self.conversation_state["character_images"], battle_image_url)
                    
                    # Store the battle image URL for potential video generation
                    self.conversation_state["battle_image_url"] = battle_image_url
                    self.conversation_state["battle_image_generated"] = True
                    
                    final_response = "🎉 **FRESH EPIC BATTLE GENERATED!**\n\n"
                    final_response += f"**{self.conversation_state['characters'][0].title()} VS {self.conversation_state['characters'][1].title()}**\n\n"
                    final_response += "✨ Your brand new cinematic battle scene is ready!\n"
                    final_response += "🔥 Witness the ultimate clash!\n\n"
                    final_response += "🎬 **Want to make it even more epic?**\n"
                    final_response += "• Type **'YES'** or **'VIDEO'** to generate an animated battle video!\n"
                    final_response += "• Type **'NO'** or **'NEW'** to start a new battle with different characters"
                    
                    # Set state to wait for video confirmation
                    self.conversation_state["waiting_for_confirmation"] = False
                    self.conversation_state["waiting_for_video_confirmation"] = True
                else:
                    final_response = "❌ **Battle Generation Failed**\n\n"
                    final_response += "Sorry, there was an issue generating the battle scene.\n"
                    final_response += "This could be due to:\n"
                    final_response += "• API limitations\n"
                    final_response += "• Image processing issues\n"
                    final_response += "• Network connectivity\n\n"
                    final_response += "Try again with the same or different characters!"
                    
                    # Reset state for new battle
                    self.reset_conversation_state()
                
                # Update the last message in history
                history[-1][1] = final_response
                
                return "", history, None, None, battle_image_url, None
                
            elif any(word in message_lower for word in ["no", "change", "different", "other"]):
                response = "🔄 **Choose New Fighters!**\n\n"
                response += "No problem! Let's pick different characters.\n\n"
                response += self.get_help_message()
                
                # Reset state for new character selection
                self.reset_conversation_state()
                
                history.append([message, response])
                return "", history, None, None, None, None
            else:
                # User didn't give clear yes/no, ask again
                response = "🤔 **Please choose:**\n\n"
                response += "• Type **'YES'** or **'CONTINUE'** to generate the fight\n"
                response += "• Type **'FORCE'** or **'REGENERATE'** to create a fresh new version\n"
                response += "• Type **'NO'** or **'CHANGE'** to pick different characters\n\n"
                response += f"Current fighters: **{self.conversation_state['characters'][0].title()}** vs **{self.conversation_state['characters'][1].title()}**"
                
                history.append([message, response])
                return "", history, None, None, None, None
        
        # Extract characters from message
        characters = self.extractor.extract_characters(message)
        
        if len(characters) == 2:
            # Found two characters, check cache first
            char1, char2 = characters
            self.conversation_state["characters"] = [char1, char2]
            
            # Check cache unless force regenerate is requested
            cached_result = None
            if not any(word in message_lower for word in ["force", "regenerate", "fresh", "new version"]):
                cached_result = self.cache.get([char1, char2])
            
            if cached_result:
                cached_char_images = cached_result.get("character_images", [None, None])
                cached_battle_img = cached_result.get("battle_image")
                cached_battle_video = cached_result.get("battle_video")
                
                if cached_battle_img and cached_battle_video:
                    # Both image and video cached
                    response = f"💾 **Found in Cache!**\n\n"
                    response += f"**{char1.title()} VS {char2.title()}**\n\n"
                    response += "🎉 Epic battle image AND video ready from cache!\n"
                    response += "🔥 Your complete battle experience awaits!\n\n"
                    response += "Want another battle? Just tell me two new characters!"
                    
                    # Set up state
                    self.conversation_state["battle_image_url"] = cached_battle_img
                    self.conversation_state["battle_image_generated"] = True
                    
                    history.append([message, response])
                    # Reset for next battle
                    self.reset_conversation_state()
                    return "", history, cached_char_images[0], cached_char_images[1], cached_battle_img, cached_battle_video
                    
                elif cached_battle_img:
                    # Only image cached
                    response = f"💾 **Found Battle Image in Cache!**\n\n"
                    response += f"**{char1.title()} VS {char2.title()}**\n\n"
                    response += "🎉 Epic battle image ready from cache!\n"
                    response += "🔥 Witness the ultimate clash!\n\n"
                    response += "🎬 **Want to make it even more epic?**\n"
                    response += "• Type **'YES'** or **'VIDEO'** to generate an animated battle video!\n"
                    response += "• Type **'FORCE'** to regenerate a fresh battle image\n"
                    response += "• Type **'NO'** or **'NEW'** to start a new battle with different characters"
                    
                    # Set up state for video confirmation
                    self.conversation_state["battle_image_url"] = cached_battle_img
                    self.conversation_state["battle_image_generated"] = True
                    self.conversation_state["waiting_for_video_confirmation"] = True
                    self.conversation_state["character_images"] = cached_char_images
                    
                    history.append([message, response])
                    return "", history, cached_char_images[0], cached_char_images[1], cached_battle_img, None
            
            # No cache hit or force regenerate requested
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
                final_response += "🖼️ Both character images loaded successfully!\n"
                final_response += "📐 Images will be resized to 1280x720 with centered content\n"
                final_response += "🔄 Advanced download methods will be used to bypass restrictions\n\n"
                final_response += "**Ready to generate the epic battle scene?**\n"
                final_response += "• Type **'YES'** or **'CONTINUE'** to create the battle!\n"
                final_response += "• Type **'FORCE'** or **'REGENERATE'** to ensure a fresh new battle\n"
                final_response += "• Type **'NO'** or **'CHANGE'** to choose different characters"
                
                # Set state to wait for confirmation
                self.conversation_state["waiting_for_confirmation"] = True
                self.conversation_state["images_loaded"] = True
                
            elif img1_url or img2_url:
                final_response = f"⚠️ **Partial Success**\n\n"
                final_response += f"**Fighter 1:** {char1.title()}\n"
                final_response += f"**Fighter 2:** {char2.title()}\n\n"
                final_response += "Found image for one fighter, but couldn't find the other.\n"
                final_response += "We can still try to create a battle scene with the available image!\n\n"
                final_response += "**What would you like to do?**\n"
                final_response += "• Type **'YES'** or **'CONTINUE'** to try with available images\n"
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
                final_response += "• Try different character combinations\n"
                final_response += "• Some characters have more accessible images than others\n\n"
                final_response += "Choose new fighters or try again with different names!"
                
                # Reset state since no images found
                self.reset_conversation_state()
            
            # Update the last message in history
            history[-1][1] = final_response
            
            return "", history, img1_url, img2_url, None, None
            
        elif len(characters) == 1:
            # Only found one character
            response = f"I found **{characters[0].title()}**, but I need TWO fighters!\n\n"
            response += "Try something like:\n"
            response += f"• '{characters[0]} vs Goku'\n"
            response += f"• '{characters[0]} fights Batman'\n"
            response += f"• 'Battle between {characters[0]} and Superman'"
            
            history.append([message, response])
            return "", history, None, None, None, None
            
        else:
            # No clear characters found
            response = self.get_help_message()
            history.append([message, response])
            return "", history, None, None, None, None
    
    def get_help_message(self) -> str:
        """Return help message for character input"""
        return """🤖 **Welcome to Meme Clash!**

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

**Special Commands:**
• Add "force" or "regenerate" to skip cache and create fresh battles

**Enhanced Features:**
• Advanced image downloading with multiple retry methods
• Character images automatically resized to 1280x720 with centered content
• Can work even if some images fail to download
• Higher quality battle scenes with proper image formatting

Just tell me who should fight! 🥊"""

# Initialize components
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
SERP_API_KEY = os.getenv("SERP_API_KEY")

if not API_KEY or not SEARCH_ENGINE_ID:
    print("Warning: Google API credentials not found in environment variables")

searcher = GoogleImageSearcher(API_KEY, SEARCH_ENGINE_ID) if API_KEY and SEARCH_ENGINE_ID else None
bot = ClashOfMemesBot(searcher) if searcher else None

def chat_interface(message, history):
    if not bot:
        return "", history + [[message, "❌ Bot not initialized. Please check your API credentials."]], None, None, None, None
        
    if not message.strip():
        return "", history, None, None, None, None
        
    return bot.process_message(message, history)

# Create Gradio interface
with gr.Blocks(title="Meme Clash", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🥊 Meme Clash - Battle Scene Generator
    ### Tell me which two characters should fight, and I'll create an epic battle scene AND video!
    
    **How it works:**
    1. Tell me two characters (e.g., "Goku vs Superman")
    2. I'll find their images
    3. Generate an epic AI battle scene!
    4. Optionally create an animated battle video!
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
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Battle Scene")
                    battle_img = gr.Image(label="Epic Battle", height=400)
                with gr.Column(scale=1):
                    gr.Markdown("### Battle Video")
                    battle_video = gr.Video(label="Animated Battle", height=400)
        
        with gr.Column(scale=1):
            gr.Markdown("### Character Images")
            with gr.Row():
                img1 = gr.Image(label="Fighter 1", height=200)
                img2 = gr.Image(label="Fighter 2", height=200)
    
    # Event handlers
    def submit_message(message, history):
        response, new_history, image1, image2, battle_image, video = chat_interface(message, history)
        return "", new_history, image1, image2, battle_image, video
    
    submit.click(
        submit_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, img1, img2, battle_img, battle_video]
    )
    
    msg.submit(
        submit_message,
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, img1, img2, battle_img, battle_video]
    )
    
    # Initialize with help message
    demo.load(
        lambda: (
            bot.reset_conversation_state() or [(None, bot.get_help_message())] if bot else [(None, "❌ Bot not initialized")], 
            None, None, None, None
        ),
        outputs=[chatbot, img1, img2, battle_img, battle_video]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)