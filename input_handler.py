import re
import streamlit as st
from typing import Tuple, Optional

class InputValidator:
    """Validates and clarifies vague user input."""
    
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 500
    VAGUE_PHRASES = {
        'good': ['interesting', 'engaging', 'fun', 'entertaining', 'well-made'],
        'fun': ['enjoyable', 'entertaining', 'engaging', 'thrilling'],
        'bad': ['difficult', 'frustrating', 'poorly made', 'unfair'],
        'game': ['game to play', 'video game', 'something to play'],
    }
    
    @staticmethod
    def validate_query(query: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validates query and returns (is_valid, cleaned_query, suggestion).
        
        Returns:
            (bool, str, str|None): (is_valid, cleaned_query, suggestion_message)
        """
        # Strip whitespace
        query = query.strip()
        
        # Check empty
        if not query:
            return False, "", "Please describe what kind of game you're looking for."
        
        # Check length
        if len(query) < InputValidator.MIN_QUERY_LENGTH:
            return False, query, f"Query too short (min {InputValidator.MIN_QUERY_LENGTH} characters). Add more details!"
        
        if len(query) > InputValidator.MAX_QUERY_LENGTH:
            return False, query, f"Query too long (max {InputValidator.MAX_QUERY_LENGTH} characters)."
        
        # Check for only numbers/special chars
        if not re.search(r'[a-zA-Z]', query):
            return False, query, "Query must contain actual words, not just numbers or symbols."
        
        # Detect vague language and suggest expansion
        vague_score = InputValidator._detect_vagueness(query)
        cleaned = InputValidator._clean_query(query)
        
        if vague_score > 0.3:
            suggestion = InputValidator._generate_suggestion(query, vague_score)
            return True, cleaned, suggestion
        
        return True, cleaned, None
    
    @staticmethod
    def _detect_vagueness(query: str) -> float:
        """Returns vagueness score 0.0-1.0."""
        query_lower = query.lower()
        vague_count = 0
        
        for vague_word in ['good', 'fun', 'bad', 'cool', 'nice', 'game']:
            if vague_word in query_lower:
                vague_count += 1
        
        return min(vague_count / 5, 1.0)
    
    @staticmethod
    def _clean_query(query: str) -> str:
        """Normalizes query: remove extra spaces, fix casing."""
        # Remove multiple spaces
        query = re.sub(r'\s+', ' ', query)
        # Capitalize first letter
        query = query[0].upper() + query[1:] if query else query
        return query
    
    @staticmethod
    def _generate_suggestion(query: str, vagueness: float) -> str:
        """Generates a helpful suggestion for vague queries."""
        suggestions = []
        
        if vagueness > 0.5:
            suggestions.append("💡 Tip: Be more specific! Try mentioning:")
            suggestions.append("  • **Genre**: Action, RPG, Puzzle, Strategy, etc.")
            suggestions.append("  • **Tone**: Relaxing, intense, competitive, story-driven")
            suggestions.append("  • **Setting**: Fantasy, Sci-Fi, Modern, Horror")
            suggestions.append("  • **Features**: Multiplayer, Single-player, Co-op")
            return "\n".join(suggestions)
        
        return "💡 Tip: Adding more specific details helps find better matches!"
    
    @staticmethod
    def suggest_query_examples() -> list:
        """Returns example queries to guide users."""
        return [
            "A relaxing farming simulator with cozy vibes",
            "Fast-paced competitive shooter with good community",
            "Story-driven RPG with deep character development",
            "Puzzle game that's not too difficult for casual play",
            "Multiplayer strategy game like Civilization or Total War",
        ]