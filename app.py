# primary import
from flask import Flask, render_template, request, jsonify
import redis
import json
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np
import hashlib
# transformer model initialiezer
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence transformer loaded successfully!")
except ImportError:
    print("sentence-transformers not found. Install with 'pip install sentence-transformers'")
    SENTENCE_TRANSFORMER_AVAILABLE = False
    embedder = None
except Exception as e:
    print(f" Error loading sentence transformer: {e}")
    SENTENCE_TRANSFORMER_AVAILABLE = False
    embedder = None

try:
    # Import classes needed for defining the RediSearch index
    from redis.commands.search.field import TextField, TagField, NumericField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    REDISEARCH_IMPORTS_AVAILABLE = True
    print(" RediSearch modules imported successfully.")
except ImportError as e:
    print(f"⚠️ RediSearch modules could not be imported: {e}")
    REDISEARCH_IMPORTS_AVAILABLE = False
    # Define dummy classes to prevent NameError if the function is called
    class DummyField:
        def __init__(self, *args, **kwargs): pass
    class DummyDefinition:
        def __init__(self, *args, **kwargs): pass
    TextField, TagField, NumericField = DummyField, DummyField, DummyField
    IndexDefinition, IndexType = DummyDefinition, None

# Load environment variables
load_dotenv()

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'redis-16546.c212.ap-south-1-1.ec2.redns.redis-cloud.com')
REDIS_PORT = int(os.getenv('REDIS_PORT', 16546))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
REDIS_DB = int(os.getenv('REDIS_DB', 0))
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

app = Flask(__name__)

# index create
def create_emergency_search_index():
    if not r:
        print("⚠️ Cannot create search index: No Redis connection.")
        return
    if not REDISEARCH_IMPORTS_AVAILABLE:
        print("⚠️ Cannot create search index: RediSearch Python modules not available.")
        return
    index_name = "idx:emergencies"
    try:
        print(f"[DEBUG] Attempting to create RediSearch index '{index_name}'...")
        schema = (
            TextField("$.message", as_name="message"),          
            TextField("$.location", as_name="location"),        
            TagField("$.ai_analysis.type_of_help", as_name="type"), 
            TagField("$.ai_analysis.urgency", as_name="urgency"),   
            NumericField("$.timestamp_numeric", as_name="timestamp_numeric", sortable=True)
        )
        definition = IndexDefinition(prefix=['emergency:'], index_type=IndexType.JSON)
        result = r.ft(index_name).create_index(fields=schema, definition=definition)
    except redis.exceptions.ResponseError as e:
        error_message = str(e)
        if "Index already exists" in error_message:
            print(f"ℹ️ RediSearch index '{index_name}' already exists.")
        else:
            print(f"⚠️ Error creating RediSearch index '{index_name}': {error_message}")
    except Exception as e:
        print(f"❌ Unexpected error in create_emergency_search_index: {e}")
        import traceback
        traceback.print_exc() 
# Redis connection
try:
    if REDIS_PASSWORD and REDIS_PASSWORD.strip():
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
        )
    else:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
        )
    r.ping()
    print("✅ Connected to Redis successfully!")
    
    # --- ADD THIS LINE TO CALL THE FUNCTION ---
    create_emergency_search_index()
    # --- END ADDITION ---
    
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    r = None


# Initialize Groq client
try:
    from groq import Groq

    if GROQ_API_KEY and GROQ_API_KEY.strip():
        groq_client = Groq(api_key=GROQ_API_KEY)

        # Light test to verify connection
        _ = groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=1
        )
        GROQ_AVAILABLE = True
        print("✅ Groq API connected successfully.")
    else:
        print("⚠️ GROQ_API_KEY not found or empty")
except Exception as e:
    print(f"❌ Failed to initialize Groq client: {e}")

# === Classifier Function ===
def classify_emergency(message):
    if not GROQ_AVAILABLE:
        return {
            "type_of_help": "unknown",
            "urgency": "medium",
            "people_affected": 1,
            "keywords": ["general"],
            "summary": "AI analysis disabled - using default values",
            "confidence": 0.5
        }

    prompt = f"""
You are an AI classifier for emergency relief.

Classify the message below into:
- type_of_help: ["medical", "food", "shelter", "rescue", "information", "transport", "donation", "security", "missing_person", "evacuation"]
- urgency: ["low", "medium", "high", "critical"]
- people_affected: estimate a number or calculte the total from the message
- keywords: extract key nouns or actions
- summary: one-sentence summary of the situation
- confidence: 0.0–1.0 confidence score

Message:
\"{message}\"

Respond with valid JSON only.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        response_text = response.choices[0].message.content.strip()
        print("[DEBUG] Raw Groq response:", response_text)

        # Clean markdown fences if present
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].rstrip("```").strip()
        elif response_text.startswith("```"):
            response_text = response_text[len("```"):].rstrip("```").strip()

        print("[DEBUG] Cleaned JSON text:", response_text)

        result = json.loads(response_text)
        return result

    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return {
            "type_of_help": "unknown",
            "urgency": "medium",
            "people_affected": 1,
            "keywords": ["general"],
            "summary": "AI response error - fallback used",
            "confidence": 0.5
        }
def cosine_similarity(list1, list2):
    """Calculate cosine similarity between two lists of floats."""
    if not list1 or not list2 or len(list1) != len(list2):
        return 0.0
    vec1 = np.array(list1)
    vec2 = np.array(list2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))
@app.route('/submit-emergency', methods=['POST'])
def submit_emergency():
    if not r:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        data = request.get_json()
        if not data or 'message' not in data or 'location' not in data:
             return jsonify({"error": "Missing required fields: message, location"}), 400
        request_id = str(uuid.uuid4())
        data['timestamp'] = datetime.now().isoformat()
        data['request_id'] = request_id
        data['timestamp_numeric'] = datetime.now().timestamp() 
        data['status'] = 'pending'
                # Semantic Caching for AI Classification ---
        ai_analysis = None
        cache_hit = False
        if r: 
            try:
                # 1. Generate a cache key based on the message content
                message_hash = hashlib.md5(data['message'].encode('utf-8')).hexdigest()
                cache_key = f"ai_cache:{message_hash}"

                # 2. Try to get the cached result
                cached_result_str = r.get(cache_key)
                if cached_result_str:
                    # 3. Cache Hit: Parse and use the cached result
                    ai_analysis = json.loads(cached_result_str)
                    cache_hit = True
                    print(f"[DEBUG] ✅ AI Classification Cache HIT for key: {cache_key}")
                    # Optional: Track cache hits
                    r.incr("ai_cache_hits")
                else:
                    # 4. Cache Miss: Perform the actual AI classification
                    print(f"[DEBUG] ⚠️ AI Classification Cache MISS for key: {cache_key}")
                    ai_analysis = classify_emergency(data['message']) # Your existing function

                    # 5. Store the result in cache with a TTL (e.g., 1 hour = 3600 seconds)
                    #    json.dumps is used to convert the dictionary to a string for Redis
                    r.setex(cache_key, 3600, json.dumps(ai_analysis))
                    print(f"[DEBUG] 📦 Stored AI result in cache for key: {cache_key}")
                    # Optional: Track cache misses
                    r.incr("ai_cache_misses")

            except Exception as e:
                print(f"[DEBUG] ⚠️ Error during AI caching logic: {e}. Proceeding without cache.")
                # Fallback to calling AI directly if caching fails
                ai_analysis = classify_emergency(data['message'])
        else:
            # If Redis is not available, just call AI directly
            ai_analysis = classify_emergency(data['message'])

        # Ensure ai_analysis is always defined
        if ai_analysis is None:
            # Final fallback, although classify_emergency should always return something
            ai_analysis = {
                "type_of_help": "unknown",
                "urgency": "medium",
                "people_affected": 1,
                "keywords": ["fallback"],
                "summary": "Fallback AI analysis.",
                "confidence": 0.0
            }

        # Attach the result (cached or fresh) to the data
        data['ai_analysis'] = ai_analysis
        # Add cache info for potential debugging/display
        data['cache_info'] = {"hit": cache_hit}
        # --- End Semantic Caching ---

        # Convert list fields to strings for Redis Stream (your existing logic)
        type_of_help = ai_analysis.get('type_of_help', 'unknown')
        if isinstance(type_of_help, list):
            type_of_help = ",".join(type_of_help)

        # --- Vector Search Integration ---
        vector_key = f"emergency:{request_id}:vector"
        vector_stored = False
        if SENTENCE_TRANSFORMER_AVAILABLE and r and embedder:
            try:
                # 1. Create embedding text (combine relevant fields)
                embedding_text = f"{data['message']} {data.get('location', '')} {ai_analysis.get('summary', '')}"

                # 2. Generate embedding
                embedding = embedder.encode(embedding_text)
                # 3. Convert to JSON-serializable list of floats
                embedding_list = embedding.tolist()
                print(f"[DEBUG] Created embedding for {request_id}, dim: {len(embedding_list)}")

                # 4. Store embedding and relevant metadata in a Redis Hash
                # Storing metadata here avoids extra lookups later in similarity search# Get type_of_help and ensure it's a string for Redis Hash storage
                raw_type_of_help = ai_analysis.get('type_of_help', 'unknown')
                if isinstance(raw_type_of_help, list):
                    type_of_help_for_hash = ",".join(raw_type_of_help) # Convert list to comma-separated string
                else:
                    type_of_help_for_hash = str(raw_type_of_help) # Ensure it's a string

                # Get keywords and ensure it's a string for Redis Hash storage
                raw_keywords = ai_analysis.get('keywords', [])
                if isinstance(raw_keywords, list):
                    keywords_for_hash = ",".join(str(k) for k in raw_keywords) # Convert list to comma-separated string
                else:
                    keywords_for_hash = str(raw_keywords) # Ensure it's a string

                vector_data_to_store = {
                    'vector': json.dumps(embedding_list),
                    'emergency_id': request_id,
                    'type_of_help': type_of_help_for_hash, # Use the string version
                    'urgency': ai_analysis.get('urgency', 'medium'),
                    'summary': ai_analysis.get('summary', 'No summary available'),
                    'location': data.get('location', ''),
                    'timestamp': data['timestamp'],
                    'people_affected': str(ai_analysis.get('people_affected', 1)),
                    'keywords': keywords_for_hash # Use the string version
                    # Add other fields, ensuring they are simple types (str, int, float, bytes)
                }

                               # Add other fields from ai_analysis or data if needed for quick lookup
                
                r.hset(vector_key, mapping=vector_data_to_store)
                vector_stored = True
                print(f"✅ Vector stored for emergency: {request_id}")
            except Exception as e:
                print(f"⚠️ Error storing vector for {request_id}: {e}")
                # Don't fail the whole request if vector storage fails
        else:
            reason = "Dependencies unavailable" if not SENTENCE_TRANSFORMER_AVAILABLE else "Redis unavailable" if not r else "Embedder not loaded"
            print(f"ℹ️ Vector storage skipped for {request_id} ({reason})")
        stream_data = {
            'request_id': request_id,
            'message': data['message'],
            'location': data['location'],
            'contact': data.get('contact', ''),
            'timestamp': data['timestamp'],
            'status': 'pending',
            'type_of_help': type_of_help, 
            'urgency': ai_analysis.get('urgency', 'medium'),
            'people_affected': str(ai_analysis.get('people_affected', 1))
        }
        stream_result = r.xadd('emergencies', stream_data)
        print(f"[DEBUG] Added to Redis Stream, result ID: {stream_result}")
        json_set_result = r.json().set(f"emergency:{request_id}", "$", data)
        print(f"[DEBUG] Stored full JSON, result: {json_set_result}")

        print(f"✅ Emergency stored with AI analysis {'and vector' if vector_stored else '(vector skipped)'}: {request_id}")
        return jsonify({
            "request_id": request_id,
            "status": "success",
            "ai_analysis": ai_analysis,
            "vector_stored": vector_stored 
        })

    except Exception as e:
        print(f"❌ Error in /submit-emergency route: {e}")
        import traceback
        traceback.print_exc() 
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
#  ROUTES 


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    try:
        r.ping()
        return jsonify({"status": "healthy", "redis": "connected"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "redis": "disconnected", "error": str(e)}), 500

@app.route('/search-emergencies', methods=['GET']) #re in the URL
def search_emergencies():
    """
    Search emergencies using RediSearch.
    Query parameters:
    - q: The search query string (e.g., 'earthquake Kathmandu')
    - offset: For pagination (default 0)
    - num: Number of results (default 10, max 100)
    """
    if not r:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # Get query parameters from the URL
        print(f"[DEBUG] /search-emergencies called")
        print(f"[DEBUG] request.args: {request.args}")
        query_string = request.args.get('q', '').strip()
        print(f"[DEBUG] Raw query_string: '{request.args.get('q', '')}'")
        print(f"[DEBUG] Stripped query_string: '{query_string}'")
        print(f"[DEBUG] Type of query_string: {type(query_string)}")

        # Get offset and num, providing defaults and converting to int
        try:
            offset = int(request.args.get('offset', 0))
            # Ensure offset is not negative
            offset = max(offset, 0)
        except ValueError:
            offset = 0

        try:
            num = int(request.args.get('num', 10))
            # Ensure num is within reasonable limits (e.g., 1 to 100)
            num = max(1, min(num, 100))
        except ValueError:
            num = 10

        # Validate the main search query
        if not query_string:
            return jsonify({"error": "Missing search query parameter 'q'"}), 400
        index_name = "idx:emergencies"
        search_query = query_string if query_string else '*'
        from redis.commands.search.query import Query
        query_obj = Query(search_query).paging(offset, num).return_fields('$') # 
        results = r.ft(index_name).search(query_obj)
        formatted_results = []
        for doc in results.docs:
             doc_id = getattr(doc, 'id', 'unknown_id')
             emergency_json_str = None
             # Try getting it as an attribute named 'json'
             if hasattr(doc, 'json'):
                 emergency_json_str = getattr(doc, 'json', None)
             if emergency_json_str:
                 # Parse the JSON string back into a Python dictionary
                 try:
                     emergency_data = json.loads(emergency_json_str)
                     formatted_results.append({
                         'id': doc_id, # The Redis key
                         'data': emergency_data 
                     })
                 except json.JSONDecodeError as je:
                     print(f"⚠️ Could not parse JSON for document {doc_id}: {je}")
                     continue 
             else:
                 print(f"⚠️ No JSON data found in document {doc_id}")
                 continue 
        return jsonify({
            "total": getattr(results, 'total', len(formatted_results)),
            "count": len(formatted_results),
            "offset": offset,
            "requested_num": num,
            "query": query_string,
            "results": formatted_results
        })
    except redis.exceptions.ResponseError as e:
        error_msg = str(e)
        print(f"❌ RediSearch ResponseError: {error_msg}")
        if "unknown index name" in error_msg.lower():
            return jsonify({"error": f"Search index '{index_name}' not found. Please ensure the index is created."}), 404 # Not Found
        else:
            return jsonify({"error": f"Search query error: {error_msg}"}), 400 
    except ValueError as e:
        return jsonify({"error": f"Invalid parameter value: {str(e)}"}), 400 
    except Exception as e:
        print(f"❌ Unexpected error in /search-emergencies: {e}")
        import traceback
        traceback.print_exc() # Print the full stack trace for debugging
        return jsonify({"error": f"Internal server error during search: {str(e)}"}), 500 # Internal Server Error


@app.route('/view-emergencies')
def view_emergencies():
    if not r:
        return "Redis connection failed"
    
    try:
        emergencies = r.xrevrange('emergencies', count=10)
        formatted_emergencies = []
        for emergency in emergencies:
            emergency_id = emergency[0].decode('utf-8') if isinstance(emergency[0], bytes) else emergency[0]
            data = emergency[1]
            
            # Decode bytes to strings if needed
            formatted_data = {}
            for key, value in data.items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                formatted_data[key] = value
            
            formatted_emergencies.append({
                'id': emergency_id,
                'data': formatted_data
            })
        
        return jsonify(formatted_emergencies)
        
    except Exception as e:
        return f"Error retrieving emergencies: {str(e)}"

@app.route('/find-similar-emergencies', methods=['POST'])
def find_similar_emergencies_api():
    """
    Find similar emergencies based on a text query using vector search.
    Expects JSON: {"text": "description of the emergency"}
    Returns JSON: List of similar emergencies with similarity scores.
    """
    if not r or not SENTENCE_TRANSFORMER_AVAILABLE or not embedder:
         return jsonify({"error": "Vector search not available"}), 500

    try:
        data = request.get_json()
        query_text = data.get('text', '')
        limit = int(data.get('limit', 5)) # Allow client to specify number of results

        if not query_text:
            return jsonify({"error": "No text provided for similarity search"}), 400

        # 1. Create embedding for the query text
        query_embedding = embedder.encode(query_text).tolist()
        print(f"[DEBUG] Created query embedding, dim: {len(query_embedding)}")

        # 2. Find all emergency vector keys
        vector_keys = r.keys("emergency:*:vector")
        print(f"[DEBUG] Found {len(vector_keys)} vector keys for similarity search")
        if not vector_keys:
            return jsonify([]) # No emergencies stored yet

        similarities = []
        # 3. Compare query vector with each stored vector
        for key in vector_keys:
            try:
                # 4. Retrieve stored vector data (Hash)
                vector_data = r.hgetall(key)
                if not vector_data:
                    print(f"[DEBUG] Empty hash data for key {key}")
                    continue

                # 5. Parse the stored embedding
                stored_embedding_str = vector_data.get('vector')
                if not stored_embedding_str:
                    print(f"[DEBUG] No 'vector' field in hash {key}")
                    continue
                stored_embedding = json.loads(stored_embedding_str)
                print(f"[DEBUG] Loaded stored embedding for {key}, dim: {len(stored_embedding)}")

                # 6. Calculate similarity
                similarity_score = cosine_similarity(query_embedding, stored_embedding)
                print(f"[DEBUG] Similarity between query and {key}: {similarity_score:.4f}")

                # 7. Collect relevant data for the result
                similarities.append({
                    'emergency_id': vector_data.get('emergency_id'),
                    'similarity': similarity_score,
                    'type_of_help': vector_data.get('type_of_help'),
                    'urgency': vector_data.get('urgency'),
                    'summary': vector_data.get('summary'),
                    'location': vector_data.get('location'),
                    'timestamp': vector_data.get('timestamp'),
                    'people_affected': vector_data.get('people_affected')
                    # Add other fields from the vector hash if needed
                })
            except (json.JSONDecodeError, ValueError) as parse_error:
                # Log error but continue with other vectors
                print(f"⚠️ Error parsing vector data from {key}: {parse_error}")
                continue
            except Exception as e:
                # Log error but continue with other vectors
                print(f"⚠️ Error processing vector {key}: {e}")
                continue

        # 8. Sort by similarity (descending) and limit results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"[DEBUG] Top {min(limit, len(similarities))} similarities calculated")
        return jsonify(similarities[:limit])

    except Exception as e:
        print(f"❌ Error in /find-similar-emergencies: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error during similarity search"}), 500


@app.route('/cache-stats/hits')
def get_cache_hits():
    if not r:
        return "0"
    try:
        hits = r.get("ai_cache_hits") or "0"
        return hits
    except:
        return "0"

@app.route('/cache-stats/misses')
def get_cache_misses():
    if not r:
        return "0"
    try:
        misses = r.get("ai_cache_misses") or "0"
        return misses
    except:
        return "0"


@app.route('/dashboard')
def dashboard():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emergency Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; }
            .emergency {
                border: 1px solid #ddd;
                margin: 15px 0;
                padding: 20px;
                border-radius: 8px;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .critical { border-left: 5px solid #d32f2f; background: #ffebee; }
            .high { border-left: 5px solid #f57c00; background: #fff3e0; }
            .medium { border-left: 5px solid #fbc02d; background: #fffde7; }
            .low { border-left: 5px solid #388e3c; background: #e8f5e8; }
            button {
                background: #1976d2;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
            }
            button:hover { background: #1565c0; }
            .urgency-badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }
            .critical-badge { background: #d32f2f; color: white; }
            .high-badge { background: #f57c00; color: white; }
            .medium-badge { background: #fbc02d; color: black; }
            .low-badge { background: #388e3c; color: white; }
            /* --- Styles for Pattern Analysis --- */
            .pattern-analysis {
                margin-top: 15px;
                padding: 10px;
                background-color: #e3f2fd; /* Light blue background */
                border-radius: 4px;
                border-left: 4px solid #2196F3; /* Blue accent */
            }
            .match-result {
                padding: 8px;
                margin: 8px 0;
                border-radius: 4px;
            }
            .high-similarity {
                background-color: #c8e6c9; /* Light green */
                border-left: 4px solid #4caf50; /* Green accent */
            }
            .medium-similarity {
                background-color: #fff9c4; /* Light yellow */
                border-left: 4px solid #fbc02d; /* Yellow accent */
            }
            .low-similarity {
                background-color: #ffecb3; /* Light orange */
                border-left: 4px solid #ff9800; /* Orange accent */
            }
            .loading { color: #666; font-style: italic; }
            .error { color: #d32f2f; }
            hr { margin: 15px 0; border: 0; border-top: 1px solid #eee; }
            .cache-analytics {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
            .cache-stats {
                display: flex;
                justify-content: space-around;
                text-align: center;
                margin-top: 15px;
            }
            .cache-stat-item {
                padding: 10px;
            }
            .cache-stat-number {
                font-size: 2em;
                font-weight: bold;
                display: block;
            }
            .cache-stat-label {
                font-size: 0.9em;
                opacity: 0.9;
            }
            .progress-bar-container {
                background: rgba(255,255,255,0.2);
                border-radius: 10px;
                height: 20px;
                margin: 10px 0;
                overflow: hidden;
            }
            .progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                border-radius: 10px;
                transition: width 0.3s ease;
}
        </style>
    </head>
    <body>
        <h1>🚑 AidBridge - Emergency Response Dashboard</h1>
        <!-- --- ADD THIS NEW CACHE PANEL HTML HERE --- -->
<div class="cache-analytics">
    <h2>⚡ Cache Performance Dashboard</h2>
    <div class="progress-bar-container">
        <div class="progress-bar" id="cacheHitRateBar" style="width: 0%"></div>
    </div>
    <div class="cache-stats">
        <div class="cache-stat-item">
            <span class="cache-stat-number" id="hitRate">0%</span>
            <span class="cache-stat-label">Hit Rate</span>
        </div>
        <div class="cache-stat-item">
            <span class="cache-stat-number" id="hits">0</span>
            <span class="cache-stat-label">Cache Hits</span>
        </div>
        <div class="cache-stat-item">
            <span class="cache-stat-number" id="misses">0</span>
            <span class="cache-stat-label">Cache Misses</span>
        </div>
        <div class="cache-stat-item">
            <span class="cache-stat-number" id="total">0</span>
            <span class="cache-stat-label">Total Requests</span>
        </div>
    </div>
</div>
        <button onclick="loadEmergencies()">🔄 Refresh Emergencies</button>
        <div style="margin: 20px 0; display: flex;">
    <input type="text" id="searchInput" placeholder="Search emergencies (e.g., 'earthquake', 'Kathmandu')..." style="flex-grow: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px 0 0 4px;">
    <button onclick="performSearch()" style="background: #4CAF50; color: white; border: none; padding: 10px 15px; border-radius: 0 4px 4px 0; cursor: pointer;">🔍 Search</button>
    <button onclick="clearSearch()" style="background: #f44336; color: white; border: none; padding: 10px 15px; margin-left: 5px; border-radius: 4px; cursor: pointer;">❌ Clear</button>
</div>
        <div id="emergencies"></div>

                <script>
            // --- Add state variables for search ---
            let currentSearchQuery = '';
            let isSearching = false;
            // --- End Add state variables for search ---

            // --- Keep existing loadCacheAnalytics function ---
            async function loadCacheAnalytics() {
                try {
                    // Fetch cache statistics from Redis
                    const [hitsResponse, missesResponse] = await Promise.all([
                        fetch('/cache-stats/hits'),
                        fetch('/cache-stats/misses')
                    ]);

                    const hits = parseInt(await hitsResponse.text()) || 0;
                    const misses = parseInt(await missesResponse.text()) || 0;
                    const total = hits + misses;
                    const hitRate = total > 0 ? Math.round((hits / total) * 100) : 0;

                    // Update the UI
                    document.getElementById('hits').textContent = hits;
                    document.getElementById('misses').textContent = misses;
                    document.getElementById('total').textContent = total;
                    document.getElementById('hitRate').textContent = hitRate + '%';
                    document.getElementById('cacheHitRateBar').style.width = hitRate + '%';

                } catch (error) {
                    console.error('Error loading cache analytics:', error);
                }
            }
            // --- End Keep existing loadCacheAnalytics function ---

            // --- Modify the existing loadEmergencies function ---
            // This function now checks the search state and delegates accordingly
            function loadEmergencies() {
                // If currently searching, load search results instead
                if (isSearching && currentSearchQuery) {
                    loadSearchResults(currentSearchQuery);
                    return;
                }

                // Otherwise, load the standard view (last N emergencies)
                fetch('/view-emergencies')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        displayEmergencies(data); // Use the new common display function
                    })
                    .catch(error => {
                        console.error('Error loading emergencies:', error);
                        document.getElementById('emergencies').innerHTML =
                            '<p style="color: red;">Error loading emergencies: ' + error.message + '</p>';
                    });
            }
            // --- End Modify loadEmergencies ---

            // --- Add new function to display emergencies ---
            // This centralizes the logic for rendering emergency cards
            function displayEmergencies(data) {
                const container = document.getElementById('emergencies');
                container.innerHTML = '';
                if (data.length === 0) {
                    container.innerHTML = '<p style="text-align: center; color: #666;">No emergencies found.</p>';
                    return;
                }
                data.forEach(emergency => {
                    // --- Use the existing card creation logic ---
                    const urgency = emergency.data.urgency || 'medium';
                    const typeOfHelp = emergency.data.type_of_help || 'unknown';
                    const emergencyId = emergency.data.request_id || 'N/A';

                    const div = document.createElement('div');
                    div.className = `emergency ${urgency}`;
                    div.id = `emergency-${emergencyId}`;

                    const cacheHit = emergency.data.cache_info?.hit || false;
                    const cacheBadge = cacheHit ?
                    '<span style="background: linear-gradient(45deg, #4CAF50, #8BC34A); color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.7em; margin-left: 8px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">⚡ CACHED</span>' :
                    '<span style="background: linear-gradient(45deg, #FF9800, #FFC107); color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.7em; margin-left: 8px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🔄 PROCESSED</span>';

                    // --- Basic Emergency Info ---
                    div.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <h3>🚨 ${typeOfHelp.toUpperCase()} EMERGENCY</h3>
                                <p><strong>Message:</strong> ${emergency.data.message || 'N/A'}</p>
                                <p><strong>Location:</strong> ${emergency.data.location || 'N/A'}</p>
                                <p><strong>People Affected:</strong> ${emergency.data.people_affected || '1'}</p>
                            </div>
                            <div>
                                <span class="urgency-badge ${urgency}-badge">${urgency.toUpperCase()}</span>
                                ${cacheBadge}
                            </div>
                        </div>
                        <p><strong>ID:</strong> ${emergencyId}</p>
                        <p><strong>Time:</strong> ${new Date(emergency.data.timestamp).toLocaleString() || 'N/A'}</p>
                        <p><strong>Contact:</strong> ${emergency.data.contact || 'Not provided'}</p>

                        <!-- ========== ACTION BUTTONS ========== -->
                        <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee;">
                            <strong>Actions:</strong>
                            ${
                                emergency.data.contact ?
                                `<a href="tel:${emergency.data.contact}" style="display: inline-block; background-color: #4CAF50; color: white; padding: 8px 12px; text-decoration: none; border-radius: 4px; margin-right: 10px; font-size: 0.9em;" target="_blank">📞 Call</a>` :
                                ''
                            }
                            ${
                                emergency.data.location ?
                                `<a href="https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(emergency.data.location)}" style="display: inline-block; background-color: #2196F3; color: white; padding: 8px 12px; text-decoration: none; border-radius: 4px; margin-right: 10px; font-size: 0.9em;" target="_blank">📍 Navigate</a>` :
                                ''
                            }
                            <button onclick="alert('Dispatch alert sent for ID: ${emergencyId}')" style="background-color: #f44336; color: white; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; font-size: 0.9em;">🟥 Dispatch</button>
                        </div>
                        <!-- ========== END ACTION BUTTONS ========== -->

                        <!-- Placeholder for Pattern Analysis -->
                        <div class="pattern-analysis" id="pattern-${emergencyId}">
                            <h4>📊 Pattern Analysis</h4>
                            <p class="loading">🔍 Analyzing similar cases...</p>
                        </div>
                        <hr>
                    `;
                    container.appendChild(div);

                    // --- Fetch and Display Similar Emergencies ---
                    fetch('/find-similar-emergencies', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            text: `${emergency.data.message || ''} ${emergency.data.location || ''}`,
                            limit: 3 // Get top 3 matches
                        })
                    })
                    .then(res => {
                        if (!res.ok) throw new Error(`API error: ${res.status}`);
                        return res.json();
                    })
                    .then(similarData => {
                        // Filter out the emergency itself from the results
                        const filteredSimilar = similarData.filter(item => item.emergency_id !== emergencyId);
                        const patternDiv = document.getElementById(`pattern-${emergencyId}`);
                        if (!patternDiv) return; // Safety check

                        if (filteredSimilar.length === 0) {
                            patternDiv.innerHTML = `
                                <h4>📊 Pattern Analysis</h4>
                                <p><em>No sufficiently similar past emergencies found.</em></p>
                                <p><strong>Suggested Response:</strong> Standard protocol for ${typeOfHelp}.</p>
                            `;
                            return;
                        }

                        const topMatch = filteredSimilar[0];
                        const similarityPercent = Math.round(topMatch.similarity * 100);

                        // Simple styling based on similarity
                        let matchClass = '';
                        if (similarityPercent > 70) matchClass = 'high-similarity';
                        else if (similarityPercent > 40) matchClass = 'medium-similarity';
                        else matchClass = 'low-similarity';

                        patternDiv.innerHTML = `
                            <h4>📊 Pattern Analysis</h4>
                            <div class="match-result ${matchClass}">
                                <strong>Most Similar Past Case:</strong> ${similarityPercent}% match<br>
                                <strong>Type:</strong> ${topMatch.type_of_help || 'N/A'}<br>
                                <strong>Summary:</strong> ${topMatch.summary || 'N/A'}<br>
                                <small><strong>ID:</strong> ${topMatch.emergency_id.substring(0, 8)}...</small>
                            </div>
                            <p><strong>Suggested Response:</strong> Based on ${similarityPercent}% similar case (${topMatch.type_of_help || 'unknown'}).</p>
                        `;
                    })
                    .catch(error => {
                         console.error('Error fetching similar emergencies for ID ' + emergencyId + ':', error);
                         const patternDiv = document.getElementById(`pattern-${emergencyId}`);
                         if (patternDiv) {
                             patternDiv.innerHTML = `
                                 <h4>📊 Pattern Analysis</h4>
                                 <p class="error">⚠️ Could not load pattern analysis: ${error.message}</p>
                                 <p><strong>Suggested Response:</strong> Standard protocol for ${typeOfHelp}.</p>
                             `;
                         }
                    });
                    // --- End Fetch and Display Similar Emergencies ---
                }); // End of forEach
            }
            // --- End Add new function to display emergencies ---

            // --- Add new function to perform search ---
            function performSearch() {
                const queryInput = document.getElementById('searchInput');
                if (!queryInput) {
                    console.error("Search input element not found.");
                    return;
                }
                const query = queryInput.value.trim();
                if (!query) {
                    alert('Please enter a search query.');
                    return;
                }
                currentSearchQuery = query;
                isSearching = true;
                loadSearchResults(query); // Trigger the search
            }
            // --- End Add new function to perform search ---

            // --- Add new function to load search results ---
            function loadSearchResults(query) {
                 // Show loading state
                      console.log("Attempting search for query:", `"${query}"`); // Add this line

                const container = document.getElementById('emergencies');
                if (container) {
                    container.innerHTML = '<p style="text-align: center; color: #666;">🔍 Searching for "' + query + '"...</p>';
                }

                // Encode query for URL
                const encodedQuery = encodeURIComponent(query);
                console.log("Fetching URL:", `/search-emergencies?q=${encodedQuery}`); // Add this line

                fetch(`/search-emergencies?q=${encodedQuery}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log("Search results:", data); // For debugging
                        // The data.results array contains objects with {id: "...", data: {...}}
                        // This format matches what displayEmergencies expects.
                        displayEmergencies(data.results); // Pass data.results to displayEmergencies
                    })
                    .catch(error => {
                        console.error('Error performing search:', error);
                        if (container) {
                            container.innerHTML =
                                '<p style="color: red;">Error performing search: ' + error.message + '</p>';
                        }
                    });
            }
            // --- End Add new function to load search results ---

            // --- Add new function to clear search ---
            function clearSearch() {
                const queryInput = document.getElementById('searchInput');
                if (queryInput) {
                    queryInput.value = '';
                }
                currentSearchQuery = '';
                isSearching = false;
                // Reload the standard view
                loadEmergencies();
            }
            // --- End Add new function to clear search ---

            // --- Update the initialization and setInterval logic ---
            // Load on page load
            loadEmergencies();
            loadCacheAnalytics(); // Load cache stats on initial load too

            // Auto-refresh every 30 seconds - now calls the modified loadEmergencies which handles search state
            setInterval(() => {
                loadEmergencies(); // This will now check isSearching and call the appropriate load function
                loadCacheAnalytics(); // Ensures cache stats are also refreshed
            }, 30000); // Refresh every 30 seconds
            // --- End Update initialization and setInterval logic ---
        </script>
    </body>
    </html>
    '''
            

@app.route('/emergency/<emergency_id>')
def get_emergency(emergency_id):
    if not r:
        return jsonify({"error": "Redis connection failed"}), 500
    
    try:
        # Try to get from JSON storage
        emergency_data = r.json().get(f"emergency:{emergency_id}")
        if emergency_data:
            return jsonify(emergency_data)
        
        # If not found, return 404
        return jsonify({"error": "Emergency not found"}), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)