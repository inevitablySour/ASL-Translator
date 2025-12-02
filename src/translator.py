"""
Translation service for ASL gestures to English/Dutch text
"""
from typing import Dict, List
from .config import settings


class Translator:
    """Translates ASL gestures to English or Dutch text"""
    
    def __init__(self):
        """Initialize translation dictionaries"""
        # ASL alphabet to English mapping (direct)
        self.asl_to_english = {
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
            'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
            'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
            'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
            'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y',
            'Z': 'Z', 'SPACE': ' ', 'UNKNOWN': '?'
        }
        
        # ASL alphabet to Dutch mapping (direct)
        self.asl_to_dutch = {
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
            'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
            'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
            'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
            'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y',
            'Z': 'Z', 'SPACE': ' ', 'UNKNOWN': '?'
        }
        
        # Common ASL words/phrases to English
        self.phrase_to_english = {
            'HELLO': 'Hello',
            'GOODBYE': 'Goodbye',
            'THANKS': 'Thank you',
            'PLEASE': 'Please',
            'YES': 'Yes',
            'NO': 'No',
            'SORRY': 'Sorry',
            'HELP': 'Help'
        }
        
        # Common ASL words/phrases to Dutch
        self.phrase_to_dutch = {
            'HELLO': 'Hallo',
            'GOODBYE': 'Tot ziens',
            'THANKS': 'Dank je',
            'PLEASE': 'Alsjeblieft',
            'YES': 'Ja',
            'NO': 'Nee',
            'SORRY': 'Sorry',
            'HELP': 'Help'
        }
    
    def translate(self, gesture: str, language: str = None) -> str:
        """
        Translate ASL gesture to target language
        
        Args:
            gesture: ASL gesture identifier (letter or phrase)
            language: Target language ('english' or 'dutch')
        
        Returns:
            Translated text
        """
        if language is None:
            language = settings.default_language
        
        language = language.lower()
        
        if language == 'dutch':
            return self._translate_to_dutch(gesture)
        else:
            return self._translate_to_english(gesture)
    
    def _translate_to_english(self, gesture: str) -> str:
        """
        Translate to English
        
        Args:
            gesture: ASL gesture identifier
        
        Returns:
            English translation
        """
        # Check if it's a phrase
        if gesture.upper() in self.phrase_to_english:
            return self.phrase_to_english[gesture.upper()]
        
        # Otherwise, treat as letter
        return self.asl_to_english.get(gesture.upper(), '?')
    
    def _translate_to_dutch(self, gesture: str) -> str:
        """
        Translate to Dutch
        
        Args:
            gesture: ASL gesture identifier
        
        Returns:
            Dutch translation
        """
        # Check if it's a phrase
        if gesture.upper() in self.phrase_to_dutch:
            return self.phrase_to_dutch[gesture.upper()]
        
        # Otherwise, treat as letter
        return self.asl_to_dutch.get(gesture.upper(), '?')
    
    def translate_sequence(self, gestures: List[str], language: str = None) -> str:
        """
        Translate a sequence of gestures to text
        
        Args:
            gestures: List of ASL gesture identifiers
            language: Target language ('english' or 'dutch')
        
        Returns:
            Translated text string
        """
        translations = [self.translate(g, language) for g in gestures]
        return ''.join(translations)
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported target languages
        
        Returns:
            List of language codes
        """
        return ['english', 'dutch']
    
    def get_gesture_info(self, gesture: str, language: str = None) -> Dict[str, str]:
        """
        Get detailed information about a gesture translation
        
        Args:
            gesture: ASL gesture identifier
            language: Target language
        
        Returns:
            Dictionary with gesture information
        """
        if language is None:
            language = settings.default_language
        
        translation = self.translate(gesture, language)
        
        return {
            'gesture': gesture,
            'language': language,
            'translation': translation,
            'type': 'phrase' if gesture.upper() in self.phrase_to_english else 'letter'
        }
