"""
Label Map Parser for TensorFlow format
Extracts word labels from label_map.pbtxt file
"""
import re
from pathlib import Path
from typing import List, Dict
from simplified_config import config, logger

class LabelMapParser:
    """Parse TensorFlow label map format"""
    
    def __init__(self, label_map_path: Path = None):
        self.label_map_path = label_map_path or Path("label_map.pbtxt")
        self.words = []
        self.id_to_word = {}
        self.word_to_id = {}
    
    def parse(self) -> List[str]:
        """
        Parse label map file and extract words
        
        Returns:
            List of word labels
        """
        if not self.label_map_path.exists():
            logger.error(f"Label map not found: {self.label_map_path}")  # Removed emoji
            return []
        
        logger.info(f"Parsing label map: {self.label_map_path}")  # Removed emoji
        
        try:
            with open(self.label_map_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse TensorFlow label map format
            # Example format:
            # item {
            #   id: 1
            #   name: 'hello'
            #   display_name: 'Hello'
            # }
            
            items = re.findall(r'item\s*\{[^}]+\}', content, re.MULTILINE | re.DOTALL)
            
            for item in items:
                # Extract id
                id_match = re.search(r'id:\s*(\d+)', item)
                
                # Extract name or display_name
                name_match = re.search(r'(?:display_name|name):\s*["\']([^"\']+)["\']', item)
                
                if id_match and name_match:
                    item_id = int(id_match.group(1))
                    word = name_match.group(1).lower().strip()
                    
                    if word:
                        self.words.append(word)
                        self.id_to_word[item_id] = word
                        self.word_to_id[word] = item_id
                        
                        logger.debug(f"Found: {item_id} -> '{word}'")
            
            # Sort words by ID to maintain order
            sorted_items = sorted(self.id_to_word.items())
            self.words = [word for _, word in sorted_items]
            
            logger.info(f"Parsed {len(self.words)} words from label map")  # Removed emoji
            
            # Log first few words
            if self.words:
                preview = ', '.join(self.words[:10])
                if len(self.words) > 10:
                    preview += f", ... and {len(self.words) - 10} more"
                logger.info(f"   Words: {preview}")
            
            return self.words
            
        except Exception as e:
            logger.error(f"Error parsing label map: {e}")  # Removed emoji
            return []
    
    def get_words_from_videos(self, videos_dir: Path = None) -> List[str]:
        """
        Get word list from video filenames as fallback
        
        Args:
            videos_dir: Directory containing video files
            
        Returns:
            List of words from video filenames
        """
        if videos_dir is None:
            videos_dir = config.VIDEOS_DIR
        
        if not videos_dir.exists():
            logger.error(f"Videos directory not found: {videos_dir}")  # Removed emoji
            return []
        
        video_files = list(videos_dir.glob("*.mp4"))
        words = []
        
        for video_file in video_files:
            word = video_file.stem.lower().strip()
            if word and word not in words:
                words.append(word)
        
        words.sort()
        logger.info(f"Found {len(words)} words from video files")  # Removed emoji
        
        return words
    
    def get_all_available_words(self) -> List[str]:
        """
        Get comprehensive list of available words from both label map and videos
        
        Returns:
            Combined list of unique words
        """
        words_from_label_map = self.parse()
        words_from_videos = self.get_words_from_videos()
        
        # Combine and deduplicate
        all_words = set(words_from_label_map + words_from_videos)
        
        # Filter out empty/invalid words
        valid_words = [word for word in all_words if word and len(word.strip()) > 0]
        valid_words.sort()
        
        logger.info(f"Total available words: {len(valid_words)}")  # Removed emoji
        
        return valid_words
    
    def save_word_list(self, output_path: Path = None) -> None:
        """Save word list to file"""
        if output_path is None:
            output_path = config.DATA_DIR / "available_words.txt"
        
        words = self.get_all_available_words()
        
        with open(output_path, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
            for word in words:
                f.write(f"{word}\n")
        
        logger.info(f"Saved {len(words)} words to {output_path}")  # Removed emoji

def main():
    """Test the label map parser"""
    parser = LabelMapParser()
    words = parser.get_all_available_words()
    
    print(f"\nAvailable Words ({len(words)}):")  # Removed emoji
    print("=" * 40)
    
    for i, word in enumerate(words, 1):
        print(f"{i:3d}. {word}")
        
        # Show first 20, then summarize
        if i == 20 and len(words) > 20:
            print(f"    ... and {len(words) - 20} more words")
            break
    
    # Save word list
    parser.save_word_list()

if __name__ == "__main__":
    main()