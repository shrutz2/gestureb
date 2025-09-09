from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.auth_models import User

search_bp = Blueprint('search', __name__, url_prefix='/api')

# Common sign language words database
SIGN_LANGUAGE_WORDS = {
    'hello': {'difficulty': 'easy', 'category': 'greetings'},
    'hi': {'difficulty': 'easy', 'category': 'greetings'},
    'goodbye': {'difficulty': 'easy', 'category': 'greetings'},
    'bye': {'difficulty': 'easy', 'category': 'greetings'},
    'please': {'difficulty': 'easy', 'category': 'politeness'},
    'thank you': {'difficulty': 'easy', 'category': 'politeness'},
    'thanks': {'difficulty': 'easy', 'category': 'politeness'},
    'sorry': {'difficulty': 'easy', 'category': 'politeness'},
    'excuse me': {'difficulty': 'medium', 'category': 'politeness'},
    'yes': {'difficulty': 'easy', 'category': 'responses'},
    'no': {'difficulty': 'easy', 'category': 'responses'},
    'maybe': {'difficulty': 'medium', 'category': 'responses'},
    'love': {'difficulty': 'medium', 'category': 'emotions'},
    'happy': {'difficulty': 'medium', 'category': 'emotions'},
    'sad': {'difficulty': 'medium', 'category': 'emotions'},
    'angry': {'difficulty': 'medium', 'category': 'emotions'},
    'help': {'difficulty': 'easy', 'category': 'assistance'},
    'good': {'difficulty': 'easy', 'category': 'descriptors'},
    'bad': {'difficulty': 'easy', 'category': 'descriptors'},
    'beautiful': {'difficulty': 'hard', 'category': 'descriptors'},
    'ugly': {'difficulty': 'medium', 'category': 'descriptors'},
    'big': {'difficulty': 'easy', 'category': 'descriptors'},
    'small': {'difficulty': 'easy', 'category': 'descriptors'},
    'hot': {'difficulty': 'easy', 'category': 'descriptors'},
    'cold': {'difficulty': 'easy', 'category': 'descriptors'},
    'water': {'difficulty': 'easy', 'category': 'necessities'},
    'food': {'difficulty': 'easy', 'category': 'necessities'},
    'eat': {'difficulty': 'easy', 'category': 'actions'},
    'drink': {'difficulty': 'easy', 'category': 'actions'},
    'sleep': {'difficulty': 'easy', 'category': 'actions'},
    'work': {'difficulty': 'medium', 'category': 'actions'},
    'play': {'difficulty': 'easy', 'category': 'actions'},
    'study': {'difficulty': 'medium', 'category': 'actions'},
    'read': {'difficulty': 'medium', 'category': 'actions'},
    'write': {'difficulty': 'medium', 'category': 'actions'},
    'listen': {'difficulty': 'medium', 'category': 'actions'},
    'see': {'difficulty': 'easy', 'category': 'actions'},
    'look': {'difficulty': 'easy', 'category': 'actions'},
    'understand': {'difficulty': 'hard', 'category': 'communication'},
    'know': {'difficulty': 'medium', 'category': 'communication'},
    'learn': {'difficulty': 'medium', 'category': 'communication'},
    'teach': {'difficulty': 'hard', 'category': 'communication'},
    'family': {'difficulty': 'medium', 'category': 'relationships'},
    'mother': {'difficulty': 'easy', 'category': 'relationships'},
    'father': {'difficulty': 'easy', 'category': 'relationships'},
    'sister': {'difficulty': 'medium', 'category': 'relationships'},
    'brother': {'difficulty': 'medium', 'category': 'relationships'},
    'friend': {'difficulty': 'medium', 'category': 'relationships'},
    'name': {'difficulty': 'easy', 'category': 'personal'},
    'age': {'difficulty': 'medium', 'category': 'personal'},
    'home': {'difficulty': 'easy', 'category': 'places'},
    'school': {'difficulty': 'medium', 'category': 'places'},
    'hospital': {'difficulty': 'hard', 'category': 'places'},
    'store': {'difficulty': 'medium', 'category': 'places'},
    'money': {'difficulty': 'medium', 'category': 'necessities'},
    'time': {'difficulty': 'medium', 'category': 'concepts'},
    'day': {'difficulty': 'easy', 'category': 'time'},
    'night': {'difficulty': 'easy', 'category': 'time'},
    'morning': {'difficulty': 'medium', 'category': 'time'},
    'afternoon': {'difficulty': 'hard', 'category': 'time'},
    'evening': {'difficulty': 'medium', 'category': 'time'},
    'today': {'difficulty': 'medium', 'category': 'time'},
    'tomorrow': {'difficulty': 'hard', 'category': 'time'},
    'yesterday': {'difficulty': 'hard', 'category': 'time'},
    'week': {'difficulty': 'medium', 'category': 'time'},
    'month': {'difficulty': 'hard', 'category': 'time'},
    'year': {'difficulty': 'hard', 'category': 'time'},
    'red': {'difficulty': 'easy', 'category': 'colors'},
    'blue': {'difficulty': 'easy', 'category': 'colors'},
    'green': {'difficulty': 'easy', 'category': 'colors'},
    'yellow': {'difficulty': 'easy', 'category': 'colors'},
    'black': {'difficulty': 'easy', 'category': 'colors'},
    'white': {'difficulty': 'easy', 'category': 'colors'},
    'orange': {'difficulty': 'medium', 'category': 'colors'},
    'purple': {'difficulty': 'medium', 'category': 'colors'},
    'pink': {'difficulty': 'medium', 'category': 'colors'},
    'brown': {'difficulty': 'medium', 'category': 'colors'},
    'one': {'difficulty': 'easy', 'category': 'numbers'},
    'two': {'difficulty': 'easy', 'category': 'numbers'},
    'three': {'difficulty': 'easy', 'category': 'numbers'},
    'four': {'difficulty': 'easy', 'category': 'numbers'},
    'five': {'difficulty': 'easy', 'category': 'numbers'},
    'six': {'difficulty': 'medium', 'category': 'numbers'},
    'seven': {'difficulty': 'medium', 'category': 'numbers'},
    'eight': {'difficulty': 'medium', 'category': 'numbers'},
    'nine': {'difficulty': 'medium', 'category': 'numbers'},
    'ten': {'difficulty': 'medium', 'category': 'numbers'},
}

def get_word_suggestions(query, limit=5):
    """Get word suggestions based on query"""
    query_lower = query.lower()
    suggestions = []
    
    # Exact matches first
    if query_lower in SIGN_LANGUAGE_WORDS:
        return [query_lower]
    
    # Partial matches
    for word in SIGN_LANGUAGE_WORDS.keys():
        if query_lower in word or word.startswith(query_lower):
            suggestions.append(word)
    
    # If no partial matches, suggest popular words
    if not suggestions:
        popular_words = ['hello', 'please', 'thank you', 'yes', 'no', 'help', 'good', 'bad']
        suggestions = popular_words
    
    return suggestions[:limit]

@search_bp.route('/search', methods=['POST'])
@jwt_required()
def search_word():
    """Search for sign language words"""
    try:
        current_user_id = get_jwt_identity()
        user = User.find_by_id(current_user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        query = data['query'].strip().lower()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Check if word exists in our database
        if query in SIGN_LANGUAGE_WORDS:
            word_info = SIGN_LANGUAGE_WORDS[query]
            
            return jsonify({
                'found': True,
                'word': query,
                'difficulty': word_info['difficulty'],
                'category': word_info['category'],
                'message': f'Found "{query}"! Get ready to practice.',
                'suggestions': []
            })
        else:
            # Word not found, provide suggestions
            suggestions = get_word_suggestions(query)
            
            return jsonify({
                'found': False,
                'word': query,
                'message': f'"{query}" not found in our database.',
                'suggestions': suggestions,
                'available_categories': list(set(info['category'] for info in SIGN_LANGUAGE_WORDS.values()))
            })
            
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': 'Search failed. Please try again.'}), 500

@search_bp.route('/words', methods=['GET'])
@jwt_required()
def get_available_words():
    """Get all available words"""
    try:
        # Group words by category
        words_by_category = {}
        for word, info in SIGN_LANGUAGE_WORDS.items():
            category = info['category']
            if category not in words_by_category:
                words_by_category[category] = []
            words_by_category[category].append({
                'word': word,
                'difficulty': info['difficulty']
            })
        
        return jsonify({
            'total_words': len(SIGN_LANGUAGE_WORDS),
            'categories': words_by_category,
            'popular_words': ['hello', 'please', 'thank you', 'yes', 'no', 'help', 'good', 'bad'],
            'easy_words': [word for word, info in SIGN_LANGUAGE_WORDS.items() if info['difficulty'] == 'easy'][:10]
        })
        
    except Exception as e:
        print(f"Get words error: {e}")
        return jsonify({'error': 'Failed to retrieve words'}), 500

@search_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get all word categories (public endpoint)"""
    try:
        categories = {}
        for word, info in SIGN_LANGUAGE_WORDS.items():
            category = info['category']
            if category not in categories:
                categories[category] = {'count': 0, 'difficulties': set()}
            categories[category]['count'] += 1
            categories[category]['difficulties'].add(info['difficulty'])
        
        # Convert sets to lists for JSON serialization
        for category in categories:
            categories[category]['difficulties'] = list(categories[category]['difficulties'])
        
        return jsonify({
            'categories': categories,
            'total_categories': len(categories)
        })
        
    except Exception as e:
        print(f"Get categories error: {e}")
        return jsonify({'error': 'Failed to retrieve categories'}), 500