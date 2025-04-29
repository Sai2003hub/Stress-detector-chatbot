import os
import json
import re
import sqlite3
import hashlib
import torch
import asyncio
import streamlit as st
import google.generativeai as genai
import pandas as pd
import altair as alt
from datetime import datetime
from textblob import TextBlob 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification
import streamlit.components.v1 as components
import time
import random
import numpy as np
from PIL import Image
import io

# Global stress level mappings
STRESS_LEVEL_MAP = {
    "No stress": 0,
    "Low stress": 1,
    "Average stress": 2,
    "High stress": 3,
    "Very high stress": 4
}

REVERSE_STRESS_LEVEL_MAP = {v: k for k, v in STRESS_LEVEL_MAP.items()}
# Helper functions to convert between stress levels and numerical scores

STRESS_LEVEL_MAP = {
    "No stress": 0,
    "Low stress": 1,
    "Average stress": 2,
    "High stress": 3,
    "Very high stress": 4
}

def stress_level_to_score(stress_level):
    return STRESS_LEVEL_MAP.get(stress_level, 2)  

def score_to_stress_level(score):
    if score < 0.8:
        return "No stress"
    elif score < 1.6:
        return "Low stress"
    elif score < 2.4:
        return "Average stress"
    elif score < 3.2:
        return "High stress"
    else:
        return "Very high stress"

def calculate_overall_stress(user_id):
    """
    Calculate the overall stress level for a user based on all games they've played.
    
    Parameters:
    - user_id: The ID of the user
    
    Returns:
    - Overall stress level string (e.g., "Average stress") or None if no games played
    """
    # Load game results for the user
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    c = conn.cursor()
    c.execute('SELECT predicted_stress FROM game_results WHERE user_id = ?', (user_id,))
    results = c.fetchall()
    conn.close()
    
    if not results:
        return None  # No games played
    
    # Convert stress levels to scores
    stress_scores = [stress_level_to_score(result['predicted_stress']) for result in results]
    
    # Calculate average stress score
    avg_stress_score = sum(stress_scores) / len(stress_scores)
    
    # Map average score back to stress level
    overall_stress = score_to_stress_level(avg_stress_score)
    return overall_stress

def calculate_overall_chat_stress(user_id):
    """
    Calculate the overall chat stress level for a user based on all their chat messages.
    
    Parameters:
    - user_id: The ID of the user
    
    Returns:
    - Overall chat stress level string (e.g., "Average stress") or None if no messages
    """
    sessions = load_all_sessions(user_id)
    if not sessions:
        return None  # No chat sessions
    
    all_stress_scores = []
    for session in sessions:
        chat_history = session['chat_history']
        for role, message in chat_history:
            if role == "user" and isinstance(message, dict) and 'predicted_stress' in message:
                stress_score = stress_level_to_score(message['predicted_stress'])
                all_stress_scores.append(stress_score)
    
    if not all_stress_scores:
        return None  # No messages with predicted stress
    
    # Calculate average stress score
    avg_stress_score = sum(all_stress_scores) / len(all_stress_scores)
    
    # Map average score back to stress level
    overall_stress = score_to_stress_level(avg_stress_score)
    return overall_stress

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         username TEXT UNIQUE NOT NULL,
         password TEXT NOT NULL,
         is_admin INTEGER DEFAULT 0)
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS game_results
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_id INTEGER,
         game_type TEXT,
         timestamp TEXT,
         reaction_time REAL,
         errors INTEGER,
         false_alarms INTEGER,
         pumps INTEGER,
         predicted_stress TEXT,
         FOREIGN KEY (user_id) REFERENCES users(id))
    ''')
    # Add new table to track game plays
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_game_plays
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_id INTEGER,
         game_type TEXT,
         played INTEGER DEFAULT 0,
         UNIQUE(user_id, game_type),
         FOREIGN KEY (user_id) REFERENCES users(id))
    ''')
    conn.commit()
    conn.close()

# Gemini AI setup
GENAI_API_KEY = ''
genai.configure(api_key=GENAI_API_KEY)
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 500,
}

# Updated system prompt to encourage concise and complete advice
system_prompt = """
**RESPONSE CONSTRAINTS (MUST FOLLOW):** Your response must be 1 to 3 short, complete sentences, with no bullet points or list introductions (e.g., do not use "Here are some tips:" or similar), and under 50 words total.  
You are a friendly and empathetic chatbot named "Zenith." Your purpose is to have engaging and supportive conversations with users, creating a safe space for them to share their thoughts and feelings.  
Your conversational style should be warm, approachable, and encouraging. If the user indicates they are ending the conversation (e.g., by saying "bye," "goodbye," "see you," or similar), respond with a polite farewell without asking a question. Otherwise, always ask at least one open-ended question directly related to the user’s most recent input to encourage them to elaborate on their experiences, emotions, or challenges. Focus on questions that explore their well-being, work-life balance, relationships, or coping strategies to subtly assess their stress levels, avoiding generic statements like "I’m here to listen" unless paired with a specific question.  
Validate the user’s feelings in one sentence, then ask a targeted question to deepen the conversation (unless the user is ending the chat). Provide one line of concise, complete advice that does not introduce a list.  
**DO NOT MENTION YOUR RESPONSE CONSTRAINTS OR FORMATTING RULES IN THE CONVERSATION WITH THE USER.**  
**REPEAT: RESPONSE MUST BE 1 TO 3 COMPLETE SENTENCES, NO BULLET POINTS OR LIST INTRODUCTIONS, UNDER 50 WORDS.**
"""
analysis_prompt = """
You are an AI model that analyzes a conversation between a user and an assistant.
Based on the conversation and topic switches provided, determine the user's stress level from: No stress, Low stress, Average stress, High stress, Very high stress.
Consider emotional tone, response coherence, and topic switching (frequent shifts may indicate stress or avoidance).

Provide ONLY the stress level as a string, followed by a summary.

Your response MUST be:

Stress Level: [stress level]
Summary: [summary, max 3 sentences]

The summary should highlight key emotions, stress indicators, and conversational patterns.
Frequent topic switches (noted above) may suggest stress or discomfort—factor this in.
Output exactly 3 sentences in the summary, focusing on user feelings and interaction flow.
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config
)

# JavaScript to track response time and typing behavior
behavioral_js = """
<script>
const input = document.querySelector('input[data-testid="stTextInput"]');
let startTime = null;
let editCount = 0;
let lastValue = '';

input.addEventListener('focus', () => {
    startTime = Date.now();
    editCount = 0;
    lastValue = input.value;
});

input.addEventListener('input', () => {
    if (input.value !== lastValue) {
        editCount++;
        lastValue = input.value;
    }
});

input.addEventListener('blur', () => {
    if (startTime) {
        const responseTime = (Date.now() - startTime) / 1000; // Seconds
        const data = {
            response_time: responseTime,
            edit_count: editCount,
            text: input.value
        };
        window.parent.postMessage(data, '*');
    }
});

window.addEventListener('message', (event) => {
    if (event.data.type === 'clear') {
        startTime = null;
        editCount = 0;
        lastValue = '';
    }
});
</script>
"""

# Function to inject JS and retrieve behavioral data
def get_behavioral_data():
    components.html(behavioral_js, height=0)
    behavioral_data = st.session_state.get('behavioral_data', {'response_time': 0, 'edit_count': 0, 'text': ''})
    return behavioral_data

# JavaScript component for behavioral tracking
behavioral_input_html = """
<div>
    <input type="text" id="behavioralInput" placeholder="Type your message..." style="width: 100%; padding: 8px;">
    <button id="submitBtn" style="margin-top: 5px; padding: 5px 10px;">Send</button>
</div>
<script>
const input = document.getElementById('behavioralInput');
const submitBtn = document.getElementById('submitBtn');
let startTime = null;
let editCount = 0;
let lastValue = '';

input.addEventListener('focus', () => {
    startTime = Date.now();
    editCount = 0;
    lastValue = input.value;
});

input.addEventListener('input', () => {
    if (input.value !== lastValue) {
        editCount++;
        lastValue = input.value;
    }
});

submitBtn.addEventListener('click', () => {
    if (startTime && input.value) {
        const responseTime = (Date.now() - startTime) / 1000; // Seconds
        const data = {
            type: 'behavioral_data',
            text: input.value,
            response_time: responseTime,
            edit_count: editCount
        };
        window.parent.Streamlit.setComponentValue(data);
        input.value = '';  // Clear input
        startTime = null;  // Reset
        editCount = 0;
        lastValue = '';
    }
});
</script>
"""

# Custom component to get behavioral data
def behavioral_chat_input(placeholder="Type your message..."):
    component_value = components.html(behavioral_input_html, height=100, scrolling=False)
    if component_value and isinstance(component_value, dict) and 'text' in component_value:
        return component_value
    return None

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, is_admin=False):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                 (username, hashed_password, int(is_admin)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, password, is_admin FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result and result[1] == hash_password(password):
        return result[0], result[2]
    return None, False

def get_username_by_id(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT username FROM users WHERE id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else "Unknown"

def get_all_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT username FROM users WHERE is_admin = 0')
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users

def get_user_id_by_username(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# Chat session management
def start_new_chat_session():
    st.session_state.viewing_past_session = False
    try:
        chat = model.start_chat(history=[])
        chat.send_message(system_prompt)
        st.session_state.current_chat_session = chat
        st.session_state.chat_history = []
        return "Hello! How's your day going so far? Let's chat."
    except Exception as e:
        return f"Error starting chat session: {e}"

def save_chat_session(user_id):
    user_folder = os.path.join('chat_history', str(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    chat_history_text = [(role, msg['text'] if isinstance(msg, dict) else msg) for role, msg in st.session_state.chat_history]
    stress_level, summary, topic_switches = analyze_conversation(chat_history_text, model, analysis_prompt)

    session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_file_path = os.path.join(user_folder, f'{session_name}.json')

    session_data = {
        "chat_history": json.dumps(st.session_state.chat_history),  # Now includes response_time, edit_count, predicted_stress
        "stress_level": stress_level,
        "summary": summary,
        "topic_switches": topic_switches
    }

    with open(session_file_path, 'w') as f:
        json.dump(session_data, f)
    st.success("Session saved!")

def load_chat_sessions(user_id=None):
    if user_id is None:
        user_folder = 'chat_history'
        session_files = []
        for user_dir in os.listdir(user_folder):
            user_path = os.path.join(user_folder, user_dir)
            if os.path.isdir(user_path):
                for filename in os.listdir(user_path):
                    if filename.endswith(".json"):
                        session_files.append(os.path.join(user_path, filename))
    else:
        user_folder = os.path.join('chat_history', str(user_id))
        session_files = []
        if os.path.exists(user_folder):
            for filename in os.listdir(user_folder):
                if filename.endswith(".json"):
                    session_files.append(os.path.join(user_folder, filename))
    
    return session_files

def load_session_chat_history(file_path):
    with open(file_path, 'r') as f:
        session_data = json.load(f)
        try:
            chat_history = json.loads(session_data["chat_history"])
            return [(role, message) for role, message in chat_history]
        except Exception as e:
            return []

def analyze_conversation(chat_history, chat_model, analysis_prompt):
    conversation_lines = []
    user_msgs = [msg for role, msg in chat_history if role == 'user']
    topic_switches = 0
    
    for i, (role, msg) in enumerate(chat_history):
        if isinstance(msg, dict):
            text = msg['text']
        else:
            text = msg
        conversation_lines.append(f"{role}: {text}")
        
        if role == 'user' and i > 0 and chat_history[i-1][0] == 'assistant':
            prev_user_msg_idx = next((j for j in range(i-1, -1, -1) if chat_history[j][0] == 'user'), None)
            if prev_user_msg_idx is not None:
                prev_msg = chat_history[prev_user_msg_idx][1]['text'] if isinstance(chat_history[prev_user_msg_idx][1], dict) else chat_history[prev_user_msg_idx][1]
                current_words = set(text.lower().split())
                prev_words = set(prev_msg.lower().split())
                overlap = len(current_words & prev_words) / min(len(current_words), len(prev_words)) if min(len(current_words), len(prev_words)) > 0 else 0
                if overlap < 0.3:
                    topic_switches += 1
    
    conversation = "\n".join(conversation_lines)
    full_prompt = f"{conversation}\nTopic Switches Detected: {topic_switches}\n{analysis_prompt}"

    try:
        response = chat_model.generate_content(full_prompt)
        analysis_text = response.text

        stress_match = re.search(r"Stress Level:\s*(No stress|Low stress|Average stress|High stress|Very high stress)", analysis_text)
        summary_match = re.search(r"Summary:\s*(.*)", analysis_text)

        stress_level = stress_match.group(1).strip() if stress_match else "Average stress"
        summary = summary_match.group(1).strip() if summary_match else "Conversation analysis failed."
        
        return stress_level, summary, topic_switches
    except Exception as e:
        return "Average stress", f"Error analyzing conversation: {e}", 0

# Updated enforce_response_constraints to avoid "Let's talk more" and prevent incomplete sentences
def enforce_response_constraints(response, user_input=""):
    # Check if the user is signaling the end of the conversation
    goodbye_phrases = ["bye", "goodbye", "see you", "later", "i'm done", "i have to go"]
    is_goodbye = any(phrase in user_input.lower() for phrase in goodbye_phrases)

    # Check if the response is a standalone statement that doesn't require a follow-up question
    # Examples: Welcoming statements, acknowledgments, or simple affirmations
    standalone_phrases = [
        "welcome home", "i hope you", "that sounds", "i understand", "glad you", 
        "nice to hear", "sounds like", "i see", "good to know", "thanks for sharing"
    ]
    is_standalone = any(phrase in response.lower() for phrase in standalone_phrases) and len(response.split()) <= 30

    # Remove bullet points, list introductions, and extra formatting
    response = re.sub(r'[-*•]\s+', '', response)  # Remove bullet points
    response = re.sub(r'(\*1\.|\d+\.|Let’s try to break this down|Here are [a-zA-Z\s]+:|Tips:|Solutions:|Try these:|Quick Fixes:|Short-Term Solutions \(for|How long have you been)\s*', '', response, flags=re.IGNORECASE)  # Remove list introductions and numbered items
    response = re.sub(r'\n+', ' ', response)  # Replace newlines with spaces
    response = re.sub(r'\s+', ' ', response).strip()  # Normalize spaces

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', response)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Remove sentences that are list introductions or incomplete
    sentences = [s for s in sentences if not s.endswith(':') and s.endswith(('.', '!', '?'))]

    # If no valid sentences remain, return a fallback
    if not sentences:
        if is_goodbye:
            return "Take care! Goodbye."
        return "I understand how you feel. What’s been on your mind lately?"

    # Truncate to 3 sentences if necessary
    if len(sentences) > 3:
        sentences = sentences[:3]

    # Ensure the response is under 50 words, prioritizing complete sentences
    current_response = ' '.join(sentences)
    words = current_response.split()
    if len(words) > 50:
        word_count = 0
        truncated_sentences = []
        for sentence in sentences:
            sentence_words = sentence.split()
            if word_count + len(sentence_words) <= 50:
                truncated_sentences.append(sentence)
                word_count += len(sentence_words)
            else:
                break
        sentences = truncated_sentences

    # If still no sentences fit, use a fallback
    if not sentences:
        if is_goodbye:
            return "Take care! Goodbye."
        return "I understand how you feel. What’s been on your mind lately?"

    # Join the sentences back together
    response = ' '.join(sentences)

    # Final check to ensure the response is under 50 words
    words = response.split()
    if len(words) > 50:
        response = ' '.join(words[:50]) + '.'

    # Ensure the response doesn’t promise a list or question without delivering
    if "things you can try" in response.lower() and not any(keyword in response.lower() for keyword in ["breathe", "splash", "drink", "relax", "bath"]):
        suggestion = "Try a warm bath to relax."
        current_words = response.split()
        if len(current_words) + len(suggestion.split()) <= 50 and len(sentences) < 3:
            response = response + ' ' + suggestion
    elif "consider a few things" in response.lower() and not response.endswith('?'):
        question = "What’s been most overwhelming for you?"
        current_words = response.split()
        if len(current_words) + len(question.split()) <= 50 and len(sentences) < 3:
            response = response + ' ' + question

    # Ensure the response contains at least one question, unless the user is saying goodbye or the response is a standalone statement
    if not is_goodbye and not is_standalone and not response.endswith('?'):
        default_question = "What’s been most challenging for you?"
        current_words = response.split()
        if len(current_words) + len(default_question.split()) <= 50 and len(sentences) < 3:
            response = response + ' ' + default_question

    # If the user is saying goodbye, ensure the response is a farewell without a question
    if is_goodbye:
        # Remove any question if present
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if not s.endswith('?')]
        response = ' '.join(sentences) if sentences else "Take care! Goodbye."
        # Ensure the response ends with a farewell
        if not any(phrase in response.lower() for phrase in goodbye_phrases):
            response = response + " Goodbye."

    # Final constraint check with logging
    final_response = response
    final_word_count = len(final_response.split())
    final_sentences = len(re.split(r'(?<=[.!?])\s+', final_response))
    if final_word_count > 50 or final_sentences > 3 or final_sentences < 1:
        print(f"Constraint violation: {final_response} | Words: {final_word_count} | Sentences: {final_sentences}")
        if is_goodbye:
            return "Take care! Goodbye."
        return "I understand how you feel. What’s been on your mind lately?"

    return final_response

# Main UI components
def show_auth_page():
    st.title("Zenith Chat")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user_id, is_admin = verify_user(username, password)
                if user_id:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_id = user_id
                    st.session_state.is_admin = is_admin
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if register_user(new_username, new_password):  # is_admin defaults to False
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

# Updated chat interface


def show_chat_interface():
    st.title(f"Chat with Zenith - {st.session_state.username}", help="Welcome to your personal stress-relief chat!")
    
    # Sidebar for session controls
    with st.sidebar:
        st.markdown("### Session Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Session", key="new_session", help="Start a fresh chat session"):
                initial_msg = start_new_chat_session()
                st.session_state.chat_history.append(("assistant", initial_msg))
                st.rerun()
        with col2:
            if st.button("Save Session", key="save_session", help="Save the current chat session"):
                save_chat_session(st.session_state.user_id)

        st.markdown("---")
        st.subheader("Past Sessions")
        session_files = load_chat_sessions(st.session_state.user_id)
        if not session_files:
            st.info("No past sessions available.")
        else:
            for session_file in session_files:
                session_name = os.path.basename(session_file)
                if st.button(f"{session_name}", key=f"load_{session_name}", help=f"Load session from {session_name}"):
                    st.session_state.chat_history = load_session_chat_history(session_file)
                    st.session_state.viewing_past_session = True
                    st.rerun()

        st.markdown("---")
        if st.button("Logout", key="logout", help="End your session"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.session_state.chat_history = []
            st.session_state.current_chat_session = None
            st.rerun()

    # Main chat container
    st.markdown("### Conversation")
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Start a new session to begin chatting!")
        else:
            for role, message in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(message if isinstance(message, str) else message['text'], unsafe_allow_html=True)

    # Inject JavaScript to track behavioral data
    components.html(behavioral_js, height=0)

    # Chat input using st.chat_input (original UI)
    if not st.session_state.viewing_past_session:
        # Get behavioral data from session state (populated by JavaScript)
        behavioral_data = st.session_state.get('behavioral_data', {'response_time': 0, 'edit_count': 0, 'text': ''})
        
        # Use st.chat_input for the original UI
        prompt = st.chat_input("Type your message...", key="chat_input")
        if prompt:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt, unsafe_allow_html=True)
                
                timestamp = datetime.now().timestamp()
                # Use behavioral data captured by JavaScript
                response_time = behavioral_data.get('response_time', 0)
                edit_count = behavioral_data.get('edit_count', 0)
                
                # Predict stress based on behavioral data
                predicted_stress = predict_stress_from_chat(response_time, edit_count)
                
                # Store message with behavioral data and stress prediction
                user_message = {
                    'text': prompt,
                    'timestamp': timestamp,
                    'response_time': response_time,
                    'edit_count': edit_count,
                    'predicted_stress': predicted_stress
                }
                st.session_state.chat_history.append(("user", user_message))
                
                try:
                    response = model.generate_content(prompt).text
                    response = enforce_response_constraints(response, user_input=prompt)
                    with st.chat_message("assistant"):
                        st.markdown(response, unsafe_allow_html=True)
                    st.session_state.chat_history.append(("assistant", response))
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                
                # Clear behavioral data after processing
                st.session_state.behavioral_data = {'response_time': 0, 'edit_count': 0, 'text': ''}
                st.rerun()
    else:
        st.warning("You’re viewing a past session. Start a new session to chat again.")
# Stress dashboard

def show_stress_dashboard():
    st.title("Stress Level Analysis Dashboard")
    
    if st.session_state.is_admin:
        users = get_all_users()
        selected_user = st.selectbox(
            "Select User:",
            ["All Users"] + users,
            key="dashboard_user_selector"
        )
        user_id = None if selected_user == "All Users" else get_user_id_by_username(selected_user)
    else:
        user_id = st.session_state.user_id
    
    sessions = load_all_sessions(user_id)
    
    if not sessions:
        st.info("No session data available for analysis yet.")
        return
    
    # Convert sessions to DataFrame, excluding chat_history
    df = pd.DataFrame(sessions)
    df = df.drop(columns=['chat_history'])
    df = df.sort_values('timestamp')
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Calculate summary metrics
    total_sessions = len(df)
    avg_stress_score = df['stress_score'].mean()
    rounded_avg = round(avg_stress_score)
    avg_label = REVERSE_STRESS_LEVEL_MAP.get(rounded_avg, "Unknown")
    avg_response_time = df['avg_response_time'].mean()
    avg_topic_switches = df['topic_switches'].mean()
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Average Stress Level", f"{rounded_avg} ({avg_label})")
    with col2:
        st.metric("Mean Stress Score", f"{avg_stress_score:.2f}/4")
    with col3:
        st.metric("Total Sessions", total_sessions)
    with col4:
        st.metric("Avg Response Time (s)", f"{avg_response_time:.2f}", help="Average time between messages")
    with col5:
        st.metric("Avg Topic Switches", f"{avg_topic_switches:.1f}", help="Average number of topic changes per session")
    
    # Combined Trend Chart
    st.subheader("Session Trends Over Time")
    df_melted = df.melt(
        id_vars=['timestamp', 'session_name', 'username', 'summary'],
        value_vars=['stress_score', 'avg_response_time', 'topic_switches'],
        var_name='Metric',
        value_name='Value'
    )
    combined_chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title="Date", axis=alt.Axis(format="%Y-%m-%d")),
        y=alt.Y('Value:Q', title="Value", scale=alt.Scale(domain=[0, max(4, df['avg_response_time'].max(), df['topic_switches'].max() + 1)])),
        color=alt.Color('Metric:N', legend=alt.Legend(title="Metrics"), scale=alt.Scale(
            domain=['stress_score', 'avg_response_time', 'topic_switches'],
            range=['#1f77b4', '#2ca02c', '#ff7f0e']
        )),
        tooltip=['session_name', 'timestamp:T', 'Metric', 'Value', 'username', 'summary']
    ).properties(
        height=400,
        width=700
    ).interactive()
    st.altair_chart(combined_chart, use_container_width=True)
    
    # Stress Level Distribution
    st.subheader("Stress Level Distribution")
    distribution = alt.Chart(df).mark_bar().encode(
        x=alt.X('stress_level:N', axis=alt.Axis(title='Stress Level')),
        y=alt.Y('count():Q', axis=alt.Axis(title='Number of Sessions')),
        color=alt.Color('stress_level:N', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title="Stress Level")),
        tooltip=['stress_level', 'count()']
    ).properties(
        height=400
    ).interactive()
    st.altair_chart(distribution, use_container_width=True)
    
    # Session Details
    st.subheader("Session Details")
    for index, row in df.iterrows():
        with st.expander(f"Session {row['session_name']}"):
            st.markdown(f"**Date & Time:** {row['date']}")
            st.markdown(f"**Stress Level:** <span style='color:#1f77b4'>{row['stress_level']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Avg Response Time (s):** <span style='color:#2ca02c'>{row['avg_response_time']:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Topic Switches:** <span style='color:#ff7f0e'>{row['topic_switches']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Analysis Summary:** {row['summary']}")
    
    # Session Overview Table
    st.subheader("Session Overview")
    session_overview = df[['session_name', 'date', 'stress_level', 'avg_response_time', 'topic_switches', 'summary']]
    st.dataframe(
        session_overview.style.format({
            'avg_response_time': '{:.2f}',
            'topic_switches': '{:.0f}'
        }).set_properties(**{
            'border-color': '#ddd',
            'text-align': 'left'
        }),
        column_config={
            "session_name": "Session ID",
            "date": "Date & Time",
            "stress_level": "Stress Level",
            "avg_response_time": st.column_config.NumberColumn("Avg Response Time (s)", format="%.2f"),
            "topic_switches": st.column_config.NumberColumn("Topic Switches", format="%.0f"),
            "summary": st.column_config.TextColumn("Analysis Summary", width="large")
        },
        hide_index=True,
        use_container_width=True
    )

# Updated calculate_lie_and_forced_scores with lie detection model
def calculate_lie_and_forced_scores(message, bert_outputs, attention_weights, tokenizer, lie_model, lie_tokenizer):
    """Calculate lie/forced scores and return key phrases from attention, using the fine-tuned lie model."""
    # Sentiment analysis
    blob = TextBlob(message)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Linguistic markers
    word_count = len(message.split())
    # Expanded list
    amplifiers = len(re.findall(r'\b(very|extremely|really|totally|so|super|absolutely|incredibly)\b', message.lower()))
    hedges = len(re.findall(r'\b(maybe|perhaps|might|sort of|guess|uh|well|possibly|probably)\b', message.lower()))  
    neg_words = len(re.findall(r'\b(stress|anxious|worried|sad|angry|bad)\b', message.lower()))
    pos_words = len(re.findall(r'\b(happy|great|awesome|good|fine|excellent)\b', message.lower()))
    amp_ratio = amplifiers / word_count if word_count > 0 else 0
    hedge_ratio = hedges / word_count if word_count > 0 else 0

    # Use the fine-tuned lie detection model
    encoding = lie_tokenizer(
        message,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = lie_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    lie_label = "Truthful" if pred == 1 else "Deceptive"
    lie_score = 1.0 if pred == 0 else 0.0  # Deceptive = 1, Truthful = 0

    # Adjust lie score with linguistic markers
    if amp_ratio > 0.15:  
        lie_score += 0.2
    if hedge_ratio > 0.08:  
        lie_score += 0.2

    # More sensitive forced emotion detection thresholds
    FORCED_EMOTION_THRESHOLDS = {
        'strong': 0.2,  
        'moderate': 0.1,  
        'weak': 0.05  
    }

    # Initialize forced emotion scores
    forced_score = 0.0
    forced_type = "None"
    
    # Positive sentiment cases - with cumulative scoring and more sensitive thresholds
    if sentiment > 0.4:  # Lowered threshold from 0.5
        if amp_ratio > 0.15 or amplifiers > 1:  # More sensitive
            forced_score += 0.3  # Changed from max() to cumulative
            forced_type = "Forced Happiness"
        if neg_words > 0:
            forced_score += 0.2
            forced_type = "Forced Happiness (Masking)"
        if subjectivity < 0.4:  # More lenient
            forced_score += 0.2
            forced_type = "Forced Happiness (Vague)"
    
    # Negative sentiment cases - with cumulative scoring and more sensitive thresholds
    elif sentiment < -0.4:  # Lowered threshold from -0.5
        if amp_ratio > 0.15 or amplifiers > 1:
            forced_score += 0.4
            forced_type = "Forced Stress"
        if pos_words > 0:
            forced_score += 0.3
            forced_type = "Forced Stress (Inconsistent)"
        if word_count < 8 and neg_words > 0:  # More lenient
            forced_score += 0.3
            forced_type = "Forced Stress (Overstated)"
    
    # Neutral sentiment cases - with cumulative scoring and wider range
    elif -0.3 <= sentiment <= 0.3:  # Wider range than -0.2 to 0.2
        if hedge_ratio > 0.08 or hedges > 0:  # More sensitive
            forced_score += 0.3
            forced_type = "Neutral Masking"
        if subjectivity < 0.3 and (neg_words > 0 or pos_words > 0):
            forced_score += 0.4
            forced_type = "Neutral Masking (Suppressed)"
        if re.search(r'\b(just|only|fine|okay)\b', message.lower()):
            forced_score += 0.3
            forced_type = "Neutral Masking (Dismissive)"
    
    # Apply threshold - only reset if score is below minimum
    if forced_score < FORCED_EMOTION_THRESHOLDS['weak']:
        forced_type = "None"
    else:
        # Cap the forced_score at 1.0
        forced_score = min(1.0, forced_score)
    
    # Adjust lie score only for significant forced emotions
    if forced_score >= FORCED_EMOTION_THRESHOLDS['moderate']:
        lie_score += forced_score * 0.5

    # Attention analysis: Get top words by attention weight
    tokens = tokenizer.tokenize(message)
    if len(tokens) > 0 and attention_weights is not None:
        # Get attention weights from BERT outputs
        # attention_weights shape: [num_layers, batch_size, num_heads, seq_len, seq_len]
        # We'll use the last layer's attention
        last_layer_attention = attention_weights[-1]  # shape: [batch_size, num_heads, seq_len, seq_len]
        
        # Average across all attention heads
        avg_attention = last_layer_attention.mean(dim=1)  # shape: [batch_size, seq_len, seq_len]
        avg_attention = avg_attention.squeeze(0)  # Remove batch dimension
        
        # Get self-attention scores (diagonal elements)
        token_attention = avg_attention.diagonal()[:len(tokens)]  # shape: [seq_len]
        
        # Get top 3 tokens with highest attention
        top_indices = token_attention.argsort(descending=True)[:3]
        key_phrases = [tokens[i] for i in top_indices if i < len(tokens)]
    else:
        key_phrases = []

    return min(1.0, lie_score), min(1.0, forced_score), forced_type, key_phrases, lie_label

# Updated process_live_text
async def process_live_text(text, bert_model, bert_tokenizer, lie_model, lie_tokenizer):
    """Process live text input for real-time analysis."""
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    attention_weights = outputs.attentions
    lie_score, forced_score, forced_type, key_phrases, lie_label = calculate_lie_and_forced_scores(
        text, outputs, attention_weights, bert_tokenizer, lie_model, lie_tokenizer
    )
    stress_label = outputs.logits.argmax().item()
    return stress_label, lie_score, forced_score, forced_type, key_phrases, lie_label

# Updated show_bert_analysis
def show_bert_analysis():
    st.title("BERT Stress, Lie & Forced Emotion Dashboard")
    
    mode = st.selectbox("Analysis Mode", ["Batch", "Real-Time"], key="mode_selector")
    
    with st.spinner("Loading models..."):
        bert_model = AutoModelForSequenceClassification.from_pretrained("./saved_bert_model", output_attentions=True)
        bert_tokenizer = AutoTokenizer.from_pretrained("./saved_bert_model")
        lie_model = DistilBertForSequenceClassification.from_pretrained("C:/Users/WinX/Downloads/Stress/saved_lie_model")
        lie_tokenizer = DistilBertTokenizer.from_pretrained("C:/Users/WinX/Downloads/Stress/saved_lie_model")
        lie_model.eval()
    
    if mode == "Real-Time":
        live_text = st.text_input("Enter text to analyze live:", "")
        if live_text:
            with st.spinner("Analyzing..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                stress_label, lie_score, forced_score, forced_type, key_phrases, lie_label = loop.run_until_complete(
                    process_live_text(live_text, bert_model, bert_tokenizer, lie_model, lie_tokenizer)
                )
                behavioral_data = st.session_state.get('behavioral_data', {'response_time': 0, 'edit_count': len(live_text.split()) // 5, 'text': live_text})
                adjusted_stress = adjust_stress_with_behavior(stress_label, lie_score, behavioral_data)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Adjusted Stress", f"{adjusted_stress:.2f}", help="Stress adjusted with lie and behavior (0-4)")
                with col2:
                    st.metric("Lie Score", f"{lie_score:.2f}", help="0 (truthful) to 1 (deceptive)")
                with col3:
                    st.metric("Lie Prediction", lie_label, help="Truthful or Deceptive")
                with col4:
                    st.metric("Forced Score", f"{forced_score:.2f}", help=f"Type: {forced_type}")
                with col5:
                    st.metric("Response Time (s)", f"{behavioral_data['response_time']:.2f}")
                st.markdown(f"**Key Phrases (Attention):** {', '.join(key_phrases) if key_phrases else 'None'}", unsafe_allow_html=True)
                
                if st.button("Save Real-Time Session"):
                    save_real_time_session(st.session_state.user_id, live_text, adjusted_stress, lie_score, forced_score, forced_type, key_phrases, lie_label)
        return
    
    # Batch processing
    if st.session_state.is_admin:
        users = get_all_users()
        selected_user = st.selectbox(
            "Select User:",
            ["All Users"] + users,
            key="bert_user_selector"
        )
        user_id = None if selected_user == "All Users" else get_user_id_by_username(selected_user)
    else:
        user_id = st.session_state.user_id
    
    sessions = load_all_sessions(user_id)
    
    if not sessions:
        st.info("No sessions available")
        return
    
    with st.spinner("Processing sessions..."):
        bert_sessions = []
        for session in sessions:
            user_msgs = [msg if isinstance(msg, str) else msg['text'] for role, msg in session['chat_history'] if role == 'user']
            session_scores = []
            session_lie_scores = []
            session_lie_labels = []
            session_forced_scores = []
            session_forced_types = []
            session_key_phrases = []
            session_response_times = []
            session_edit_counts = []
            
            for i, msg in enumerate(user_msgs):
                inputs = bert_tokenizer(msg, return_tensors="pt", truncation=True, padding=True)
                outputs = bert_model(**inputs)
                stress_label = outputs.logits.argmax().item()
                attention_weights = outputs.attentions
                lie_score, forced_score, forced_type, key_phrases, lie_label = calculate_lie_and_forced_scores(
                    msg, outputs, attention_weights, bert_tokenizer, lie_model, lie_tokenizer
                )
                
                behavioral_data = {
                    'response_time': session.get('avg_response_time', 0),
                    'edit_count': session.get('avg_edit_count', len(msg.split()) // 5),
                    'text': msg
                }
                
                adjusted_stress = adjust_stress_with_behavior(stress_label, lie_score, behavioral_data)
                
                session_scores.append(adjusted_stress)
                session_lie_scores.append(lie_score)
                session_lie_labels.append(lie_label)
                session_forced_scores.append(forced_score)
                session_forced_types.append(forced_type)
                session_key_phrases.append(key_phrases)
                session_response_times.append(behavioral_data['response_time'])
                session_edit_counts.append(behavioral_data['edit_count'])
            
            session_avg = sum(session_scores) / len(session_scores) if session_scores else 0
            session_max = max(session_scores) if session_scores else 0
            session_min = min(session_scores) if session_scores else 0
            session_avg_lie = sum(session_lie_scores) / len(session_lie_scores) if session_lie_scores else 0
            session_lie_label = max(set(session_lie_labels), key=session_lie_labels.count) if session_lie_labels else "Unknown"
            session_avg_forced = sum(session_forced_scores) / len(session_forced_scores) if session_forced_scores else 0
            session_forced_type = max(set(session_forced_types), key=session_forced_types.count) if session_forced_types else "None"
            all_phrases = [phrase for sublist in session_key_phrases for phrase in sublist]
            session_key_phrases_agg = list(set(all_phrases))[:3] if all_phrases else []
            session_avg_response_time = sum(session_response_times) / len(session_response_times) if session_response_times else 0
            session_avg_edit_count = sum(session_edit_counts) / len(session_edit_counts) if session_edit_counts else 0
            
            bert_sessions.append({
                'session_name': session['session_name'],
                'timestamp': session['timestamp'],
                'username': session['username'],
                'avg_stress': session_avg,
                'max_stress': session_max,
                'min_stress': session_min,
                'avg_lie': session_avg_lie,
                'lie_label': session_lie_label,
                'avg_forced': session_avg_forced,
                'forced_type': session_forced_type,
                'key_phrases': session_key_phrases_agg,
                'avg_response_time': session_avg_response_time,
                'avg_edit_count': session_avg_edit_count,
                'summary': session['summary']
            })
    
    # Create DataFrame
    df_sessions = pd.DataFrame(bert_sessions).sort_values('timestamp')
    
    # Calculate overall metrics
    overall_avg = df_sessions['avg_stress'].mean()
    overall_max = df_sessions['max_stress'].max()
    overall_min = df_sessions['min_stress'].min()
    overall_avg_lie = df_sessions['avg_lie'].mean()
    overall_avg_forced = df_sessions['avg_forced'].mean()
    overall_avg_response_time = df_sessions['avg_response_time'].mean()
    overall_avg_edit_count = df_sessions['avg_edit_count'].mean()
    
    # Display metrics
    st.subheader("Overall Metrics")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        st.metric("Avg Stress", f"{overall_avg:.2f}", help="Adjusted stress (0-4)")
    with col2:
        st.metric("Max Stress", f"{overall_max:.2f}")
    with col3:
        st.metric("Min Stress", f"{overall_min:.2f}")
    with col4:
        st.metric("Avg Lie Score", f"{overall_avg_lie:.2f}", help="0-1")
    with col5:
        st.metric("Avg Forced Score", f"{overall_avg_forced:.2f}", help="0-1")
    with col6:
        st.metric("Avg Response Time (s)", f"{overall_avg_response_time:.2f}")
    with col7:
        st.metric("Avg Edit Count", f"{overall_avg_edit_count:.0f}")
    with col8:
        st.metric("Total Sessions", len(df_sessions))
    
    # Combined Trends
    st.subheader("Session Trends Over Time")
    df_melted = df_sessions.melt(
        id_vars=['timestamp', 'session_name', 'username', 'forced_type'],
        value_vars=['avg_stress', 'avg_lie', 'avg_forced', 'avg_response_time'],
        var_name='Metric',
        value_name='Value'
    )
    combined_chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title="Date", axis=alt.Axis(format="%Y-%m-%d")),
        y=alt.Y('Value:Q', title="Value", scale=alt.Scale(domain=[0, max(4, overall_avg_response_time + 1)])),
        color=alt.Color('Metric:N', legend=alt.Legend(title="Metrics"), scale=alt.Scale(
            domain=['avg_stress', 'avg_lie', 'avg_forced', 'avg_response_time'],
            range=['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']
        )),
        tooltip=['session_name', 'timestamp:T', 'Metric', 'Value', 'username', 'forced_type']
    ).properties(
        height=400,
        width=700
    ).interactive()
    st.altair_chart(combined_chart, use_container_width=True)
    
    # Stress Distribution
    st.subheader("Stress Level Distribution")
    distribution = alt.Chart(df_sessions).mark_bar().encode(
        x=alt.X('avg_stress:Q', bin=alt.Bin(step=0.5), title='Adjusted Stress Level (0-4)'),
        y=alt.Y('count()', title='Number of Sessions'),
        color=alt.Color('avg_stress:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title="Stress Level")),
        tooltip=['avg_stress:Q', 'count()']
    ).properties(
        height=350,
        width=600
    ).interactive()
    st.altair_chart(distribution, use_container_width=True)
    
    # Session Details
    st.subheader("Session Details")
    for index, row in df_sessions.iterrows():
        with st.expander(f"Session {row['session_name']}"):
            st.markdown(f"**Date & Time:** {row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Username:** {row['username']}")
            st.markdown(f"**Adjusted Stress:** <span style='color:#1f77b4'>{row['avg_stress']:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Max Stress:** {row['max_stress']:.2f}")
            st.markdown(f"**Min Stress:** {row['min_stress']:.2f}")
            st.markdown(f"**Average Lie Score:** <span style='color:#ff7f0e'>{row['avg_lie']:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Lie Prediction:** {row['lie_label']}", unsafe_allow_html=True)
            st.markdown(f"**Average Forced Score:** <span style='color:#9467bd'>{row['avg_forced']:.2f}</span> ({row['forced_type']})", unsafe_allow_html=True)
            st.markdown(f"**Avg Response Time (s):** <span style='color:#2ca02c'>{row['avg_response_time']:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Avg Edit Count:** {row['avg_edit_count']:.0f}")
            st.markdown(f"**Key Phrases (Attention):** {', '.join(row['key_phrases']) if row['key_phrases'] else 'None'}")
            st.markdown(f"**Analysis Summary:** {row['summary']}")
    
    # Session Overview Table
    st.subheader("Session Overview")
    session_overview = df_sessions[['session_name', 'timestamp', 'username', 'avg_stress', 'avg_lie', 'lie_label', 'avg_forced', 'forced_type', 'avg_response_time', 'avg_edit_count', 'key_phrases', 'summary']]
    session_overview['timestamp'] = session_overview['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(
        session_overview.style.format({
            'avg_stress': '{:.2f}',
            'avg_lie': '{:.2f}',
            'avg_forced': '{:.2f}',
            'avg_response_time': '{:.2f}',
            'avg_edit_count': '{:.0f}'
        }).set_properties(**{
            'border-color': '#ddd',
            'text-align': 'left'
        }),
        column_config={
            'session_name': "Session ID",
            'timestamp': "Date & Time",
            'username': "Username",
            'avg_stress': st.column_config.NumberColumn("Adjusted Stress", help="0-4 scale", format="%.2f"),
            'avg_lie': st.column_config.NumberColumn("Average Lie Score", help="0-1 scale", format="%.2f"),
            'lie_label': "Lie Prediction",
            'avg_forced': st.column_config.NumberColumn("Average Forced Score", help="0-1 scale", format="%.2f"),
            'forced_type': "Forced Type",
            'avg_response_time': st.column_config.NumberColumn("Avg Response Time (s)", format="%.2f"),
            'avg_edit_count': st.column_config.NumberColumn("Avg Edit Count", format="%.0f"),
            'key_phrases': st.column_config.ListColumn("Key Phrases"),
            'summary': st.column_config.TextColumn("Analysis Summary", width="large")
        },
        hide_index=True,
        use_container_width=True
    )

# Updated save_real_time_session
def save_real_time_session(user_id, text, stress_label, lie_score, forced_score, forced_type, key_phrases, lie_label):
    user_folder = os.path.join('chat_history', str(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_file_path = os.path.join(user_folder, f'{session_name}.json')
    
    chat_history = [["user", text], ["assistant", "Analyzed in real-time"]]
    stress_map = {0: "Very Low Stress", 1: "Low Stress", 2: "Moderate Stress", 3: "High Stress", 4: "Very High Stress"}
    stress_level = stress_map[int(stress_label)]
    
    behavioral_data = st.session_state.get('behavioral_data', {'response_time': 0, 'edit_count': len(text.split()) // 5, 'text': text})
    
    session_data = {
        "chat_history": json.dumps(chat_history),
        "stress_level": stress_level,
        "summary": f"Real-time analysis: Stress={stress_label:.2f}, Lie={lie_score:.2f} ({lie_label}), Forced={forced_score:.2f} ({forced_type})",
        "lie_score": lie_score,
        "lie_label": lie_label,
        "forced_score": forced_score,
        "forced_type": forced_type,
        "key_phrases": key_phrases,
        "response_time": behavioral_data['response_time'],
        "edit_count": behavioral_data['edit_count']
    }
    
    with open(session_file_path, 'w') as f:
        json.dump(session_data, f)
    st.success(f"Real-time session saved as {session_name}.json!")

def adjust_stress_with_behavior(stress_label, lie_score, behavioral_data):
    adjusted_stress = stress_label
    
    if lie_score > 0.7:
        adjusted_stress += 1 * (lie_score - 0.7)
    elif lie_score < 0.3 and stress_label > 2:
        adjusted_stress -= 0.5 * (0.3 - lie_score)
    
    response_time = behavioral_data['response_time']
    edit_count = behavioral_data['edit_count']
    
    if response_time > 5:
        adjusted_stress += 0.5
    elif response_time < 1 and stress_label > 2:
        adjusted_stress -= 0.3
    
    if edit_count > 3:
        adjusted_stress += 0.5
    
    return min(4.0, max(0.0, adjusted_stress))

# Load sessions
def load_all_sessions(user_id=None):
    session_files = load_chat_sessions(user_id)
    sessions = []
    
    for file_path in session_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                session_name = os.path.basename(file_path)
                timestamp_part = session_name.split('.')[0]
                timestamp = datetime.strptime(timestamp_part, "%Y-%m-%d_%H-%M-%S")
                
                stress_level = data.get('stress_level')
                stress_score = STRESS_LEVEL_MAP.get(stress_level, 0)
                summary = data.get('summary')
                chat_history = json.loads(data.get('chat_history', '[]'))
                topic_switches = data.get('topic_switches', 0)
                
                user_dir = os.path.dirname(file_path)
                user_id_str = os.path.split(user_dir)[1]
                user_id = int(user_id_str)
                username = get_username_by_id(user_id)
                
                total_response_time = 0
                total_edit_count = 0
                msg_count = 0
                last_timestamp = None
                
                for i, (role, msg) in enumerate(chat_history):
                    if role == "user":
                        msg_data = msg if isinstance(msg, dict) else {'text': msg, 'timestamp': None}
                        current_time = msg_data.get('timestamp', timestamp.timestamp() + i)
                        if last_timestamp and current_time:
                            response_time = current_time - last_timestamp
                            total_response_time += response_time
                        last_timestamp = current_time
                        msg_count += 1
                        edit_count = len(msg_data['text'].split()) // 5
                        total_edit_count += edit_count
                
                avg_response_time = total_response_time / msg_count if msg_count > 1 else 0
                avg_edit_count = total_edit_count / msg_count if msg_count > 0 else 0
                
                sessions.append({
                    'session_name': session_name,
                    'timestamp': timestamp,
                    'stress_level': stress_level,
                    'stress_score': stress_score,
                    'summary': summary,
                    'chat_history': chat_history,
                    'username': username,
                    'avg_response_time': avg_response_time,
                    'avg_edit_count': avg_edit_count,
                    'topic_switches': topic_switches
                })
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    
    return sessions

def save_game_result(user_id, game_type, reaction_time, errors, false_alarms, pumps, predicted_stress):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute('''
        INSERT INTO game_results (user_id, game_type, timestamp, reaction_time, errors, false_alarms, pumps, predicted_stress)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, game_type, timestamp, reaction_time, errors, false_alarms, pumps, predicted_stress))
    conn.commit()
    conn.close()

def load_game_results(user_id=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    if user_id:
        c.execute('SELECT * FROM game_results WHERE user_id = ?', (user_id,))
    else:
        c.execute('SELECT * FROM game_results')
    rows = c.fetchall()
    conn.close()
    return [
        {
            'id': row[0],
            'user_id': row[1],
            'game_type': row[2],
            'timestamp': datetime.fromisoformat(row[3]),
            'reaction_time': row[4],
            'errors': row[5],
            'false_alarms': row[6],
            'pumps': row[7],
            'predicted_stress': row[8],
            'username': get_username_by_id(row[1])
        }
        for row in rows
    ]
def has_user_played_game(user_id, game_type):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Check if the user has an entry for this game
    c.execute('SELECT played FROM user_game_plays WHERE user_id = ? AND game_type = ?', (user_id, game_type))
    result = c.fetchone()
    if result is None:
        # If no entry exists, create one with played = 0
        c.execute('INSERT INTO user_game_plays (user_id, game_type, played) VALUES (?, ?, 0)', (user_id, game_type))
        conn.commit()
        conn.close()
        return False
    conn.close()
    return result[0] == 1

def mark_game_as_played(user_id, game_type):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Update the played status to 1
    c.execute('UPDATE user_game_plays SET played = 1 WHERE user_id = ? AND game_type = ?', (user_id, game_type))
    conn.commit()
    conn.close()



def predict_stress_from_game(game_type, reaction_time, errors, risky_choices, pumps, final_resources=None, burst=None):
    """
    Predict stress level based on game performance using a weighted scoring system aligned with psychological research.
    
    Parameters:
    - game_type: "Stroop Test", "Crisis Management", or "BART"
    - reaction_time: Average reaction time in seconds (for Stroop and Crisis Management)
    - errors: Number of errors made (for Stroop and Crisis Management)
    - risky_choices: Number of risky decisions (for Crisis Management)
    - pumps: Number of pumps (for BART)
    - final_resources: Dict with final resource values (for Crisis Management)
    - burst: Boolean indicating if the balloon burst (for BART)
    
    Returns:
    - Stress level string: "No stress", "Low stress", "Average stress", "High stress", "Very high stress"
    """
    stress_score = 0.0  # Score ranges from 0 to 4

    if game_type == "Stroop Test":
        # Normalize reaction time (baseline: 1 second is average, 1.5 seconds is high stress)
        # Research: MacLeod (1991) - typical RT 600-1000 ms; Renaud & Blondin (1997) - stressed RT 1000-1500 ms
        reaction_time_score = min(reaction_time / 1.5, 1.0)  # 0 to 1 (0.8s → 0.53, 1.5s → 1.0)
        # Normalize errors (5 trials total, so 5 errors max)
        # Research: Booth & Sharma (2022) - error rate increases 20-30% under stress
        error_score = errors / 5.0  # 0 to 1 (1 error → 0.2, 5 errors → 1.0)
        # Weighted stress score (reaction time: 40%, errors: 60%)
        # Research: Renaud & Blondin (1997) - errors more indicative of stress due to cognitive interference
        stress_score = (0.4 * reaction_time_score + 0.6 * error_score) * 4  # Scale to 0-4

    elif game_type == "Crisis Management":
        # Normalize reaction time (average time limit per stage is ~14.4 seconds)
        # Research: van der Vijgh et al. (2018) - low stress: 5-7s, high stress: 10-15s
        reaction_time_score = min(reaction_time / 14.4, 1.0)  # 0 to 1 (7.2s → 0.5, 14.4s → 1.0)
        # Normalize errors (5 stages, assume max 5 errors)
        # Research: van der Vijgh et al. (2018) - errors double under stress (2-3 in 5 decisions)
        error_score = errors / 5.0  # 0 to 1
        # Normalize risky choices (5 stages, max 5 risky choices)
        # Research: van der Vijgh et al. (2018) - risky choices increase 50-60% under stress (3-4 out of 5)
        risky_choice_score = risky_choices / 5.0  # 0 to 1
        # Normalize resource depletion (sum of resources: max 225, min 0)
        # Research: van der Vijgh et al. (2018) - high stress when resources drop below 20-30%
        if final_resources:
            total_resources = final_resources['budget'] + final_resources['personnel'] + final_resources['trust']
            resource_score = 1 - (total_resources / 225.0)  # 0 to 1 (lower resources → higher stress)
        else:
            resource_score = 0.5  # Default if not provided
        # Weighted stress score (reaction time: 20%, errors: 30%, risky choices: 20%, resources: 30%)
        # Research: van der Vijgh et al. (2018) - errors and resource depletion are strong stress indicators
        stress_score = (0.2 * reaction_time_score + 0.3 * error_score + 0.2 * risky_choice_score + 0.3 * resource_score) * 4

    elif game_type == "BART":
        # Normalize pumps relative to average burst point (20 pumps)
        # Research: Lighthall et al. (2009) - low stress: 10-15 pumps, high stress: <8 or >20 pumps
        if pumps <= 20:
            # Low pumps indicate caution (stress from anxiety)
            pump_score = (20 - pumps) / 20.0  # 0 to 1 (20 pumps → 0, 0 pumps → 1)
        else:
            # High pumps indicate risk-taking (stress from impulsivity)
            pump_score = (pumps - 20) / 10.0  # 0 to 1 (20 pumps → 0, 30 pumps → 1)
        # Burst penalty: Add stress if the balloon burst
        # Research: Lighthall et al. (2009) - bursts increase stress by 15-20%
        burst_score = 0.5 if burst else 0.0  # Add 0.5 to stress score if burst
        # Weighted stress score (pumps: 70%, burst: 30%)
        # Research: Lighthall et al. (2009) - pumps are primary stress indicator
        stress_score = (0.7 * pump_score + 0.3 * burst_score) * 4

    # Map stress score to stress level (0 to 4)
    # Research: Cohen et al. (1983) - aligns with PSS quintile-based categorization
    if stress_score < 0.8:
        return "No stress"
    elif stress_score < 1.6:
        return "Low stress"
    elif stress_score < 2.4:
        return "Average stress"
    elif stress_score < 3.2:
        return "High stress"
    else:
        return "Very high stress"
    
def predict_stress_from_chat(response_time, edit_count):
    """
    Predict stress level based on chat behavioral data using a weighted scoring system.
    
    Parameters:
    - response_time: Time taken to respond in seconds
    - edit_count: Number of edits made to the message before sending
    
    Returns:
    - Stress level string: "No stress", "Low stress", "Average stress", "High stress", "Very high stress"
    """
    stress_score = 0.0  # Score ranges from 0 to 4

    # Normalize response time (baseline: 5s is average, 15s is high stress)
    # Research: Vizer et al. (2009) - low stress: 2-5s, high stress: 7-15s
    response_time_score = min(response_time / 15.0, 1.0)  # 0 to 1 (5s → 0.33, 15s → 1.0)

    # Normalize edit count (baseline: 0-2 edits is low stress, 5+ is high stress)
    # Research: Gao et al. (2014) - stressed users make 3-5 edits
    edit_count_score = min(edit_count / 5.0, 1.0)  # 0 to 1 (2 edits → 0.4, 5 edits → 1.0)

    # Weighted stress score (response time: 60%, edit count: 40%)
    # Rationale: Response time is a stronger stress indicator (Vizer et al., 2009)
    stress_score = (0.6 * response_time_score + 0.4 * edit_count_score) * 4  # Scale to 0-4

    # Map stress score to stress level (0 to 4)
    if stress_score < 0.8:
        return "No stress"
    elif stress_score < 1.6:
        return "Low stress"
    elif stress_score < 2.4:
        return "Average stress"
    elif stress_score < 3.2:
        return "High stress"
    else:
        return "Very high stress"
    
def stroop_test(user_id):
    st.subheader("Stroop Test")
    st.write("Name the color of the text, not the word itself. Click the button matching the color.")
    
    # Check if the user has already played this game
    if has_user_played_game(user_id, "Stroop Test"):
        st.warning("You have already played the Stroop Test. You can only play each game once.")
        return
    
    colors = ["red", "blue", "green", "purple"]
    words = ["RED", "BLUE", "GREEN", "PURPLE"]
    
    # Initialize stroop_state if it doesn't exist
    if 'stroop_state' not in st.session_state:
        st.session_state.stroop_state = {
            'trial': 0,
            'correct_color': None,
            'display_word': None,
            'start_time': None,
            'game_start': None,
            'errors': 0,
            'reaction_times': [],
            'started': False,
            'result_saved': False
        }
    
    state = st.session_state.stroop_state
    
    if not state['started']:
        if st.button("Start Game", key="stroop_start"):
            state['started'] = True
            state['game_start'] = time.time()
            state['correct_color'] = random.choice(colors)
            state['display_word'] = random.choice(words)
            state['start_time'] = time.time()
            state['result_saved'] = False
            st.rerun()
    else:
        elapsed_time = time.time() - state['game_start']
        st.markdown(f"<div class='timer'>{int(elapsed_time)}s</div>", unsafe_allow_html=True)
        
        if state['trial'] < 5:
            st.markdown(
                f"<h3 style='color:{state['correct_color']};text-align:center;animation:fadeIn 0.5s'>{state['display_word']}</h3>",
                unsafe_allow_html=True
            )
            
            cols = st.columns(4)
            for i, color in enumerate(colors):
                if cols[i].button(color.capitalize(), key=f"stroop_{color}_{state['trial']}", help=f"Click if the text is {color}"):
                    reaction_time = time.time() - state['start_time']
                    state['reaction_times'].append(reaction_time)
                    if color != state['correct_color']:
                        state['errors'] += 1
                        st.error("Wrong color!")
                    else:
                        st.success("Correct!")
                    state['trial'] += 1
                    if state['trial'] < 5:
                        state['correct_color'] = random.choice(colors)
                        state['display_word'] = random.choice(words)
                        state['start_time'] = time.time()
                    st.rerun()
        else:
            avg_reaction_time = sum(state['reaction_times']) / len(state['reaction_times']) if state['reaction_times'] else 0
            predicted_stress = predict_stress_from_game("Stroop Test", avg_reaction_time, state['errors'], 0, 0)
            if not state['result_saved']:
                save_game_result(user_id, "Stroop Test", avg_reaction_time, state['errors'], 0, 0, predicted_stress)
                mark_game_as_played(user_id, "Stroop Test")
                state['result_saved'] = True
            if st.session_state.is_admin:
                st.markdown(f"<h4 style='color:#4CAF50'>Results: Avg Reaction Time: {avg_reaction_time:.2f}s, Errors: {state['errors']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color:#1f77b4'>Predicted Stress Level: {predicted_stress}</h4>", unsafe_allow_html=True)
            else:
                st.success("Stroop Test completed! Your results have been saved.")
            if st.button("Play Again", key="stroop_reset"):
                del st.session_state.stroop_state
                st.rerun()

def crisis_management_sim(user_id):
    st.subheader("Crisis Management Simulation")
    st.write("You are the crisis manager for a city facing a natural disaster. Make decisions to save lives and manage resources under time pressure!")
    
    # Check if the user has already played this game
    if has_user_played_game(user_id, "Crisis Management"):
        st.warning("You have already played the Crisis Management Simulation. You can only play each game once.")
        return
    
    # Initialize game state
    if 'crisis_state' not in st.session_state:
        st.session_state.crisis_state = {
            'stage': 0,
            'game_start': None,
            'started': False,
            'reaction_times': [],
            'risky_choices': 0,
            'errors': 0,
            'resources': {'budget': 100, 'personnel': 50, 'trust': 75},
            'scenarios': [
                {
                    'desc': "A massive flood hits the city. Water levels are rising fast. What do you do?",
                    'options': [
                        ("Send all personnel to evacuate low-lying areas (High Risk)", {'budget': -20, 'personnel': -30, 'trust': 10}, True),
                        ("Deploy sandbags and pumps to critical zones", {'budget': -30, 'personnel': -10, 'trust': 5}, False),
                        ("Wait for more data before acting", {'budget': 0, 'personnel': 0, 'trust': -10}, False)
                    ],
                    'time_limit': 15
                },
                {
                    'desc': "Power outages spread due to flooded substations. Citizens are panicking. Your move?",
                    'options': [
                        ("Rush emergency generators to hospitals (High Risk)", {'budget': -40, 'personnel': -15, 'trust': 15}, True),
                        ("Prioritize repairs to main grid", {'budget': -25, 'personnel': -20, 'trust': 5}, False),
                        ("Tell citizens to conserve power", {'budget': -5, 'personnel': 0, 'trust': -15}, False)
                    ],
                    'time_limit': 12
                },
                {
                    'desc': "Looters exploit the chaos. Police are stretched thin. How do you respond?",
                    'options': [
                        ("Declare martial law (High Risk)", {'budget': -30, 'personnel': -25, 'trust': -20}, True),
                        ("Redirect personnel to patrol key areas", {'budget': -10, 'personnel': -15, 'trust': 10}, False),
                        ("Ask for volunteer patrols", {'budget': 0, 'personnel': -5, 'trust': -5}, False)
                    ],
                    'time_limit': 10
                },
                {
                    'desc': "A hospital floods, needing urgent evacuation. Resources are low. What’s your call?",
                    'options': [
                        ("Use all remaining budget for airlifts (High Risk)", {'budget': -50, 'personnel': -10, 'trust': 20}, True),
                        ("Coordinate with nearby cities for help", {'budget': -15, 'personnel': -5, 'trust': 10}, False),
                        ("Focus on other priorities", {'budget': 0, 'personnel': 0, 'trust': -25}, False)
                    ],
                    'time_limit': 15
                },
                {
                    'desc': "The crisis peaks: a dam might burst. Time’s running out. Final decision?",
                    'options': [
                        ("Evacuate downstream now (High Risk)", {'budget': -40, 'personnel': -30, 'trust': 15}, True),
                        ("Reinforce the dam with all resources", {'budget': -50, 'personnel': -20, 'trust': 5}, False),
                        ("Monitor and pray", {'budget': 0, 'personnel': 0, 'trust': -30}, False)
                    ],
                    'time_limit': 20
                }
            ],
            'result_saved': False
        }
    
    state = st.session_state.crisis_state
    
    if not state['started']:
        if st.button("Start Simulation", key="crisis_start"):
            state['started'] = True
            state['game_start'] = time.time()
            state['last_decision_time'] = time.time()
            state['result_saved'] = False
            st.rerun()
    else:
        elapsed_time = time.time() - state['game_start']
        st.markdown(f"<div class='timer'>{int(elapsed_time)}s</div>", unsafe_allow_html=True)
        
        if state['stage'] < len(state['scenarios']):
            scenario = state['scenarios'][state['stage']]
            time_left = scenario['time_limit'] - (time.time() - state['last_decision_time'])
            
            # Display scenario and resources
            st.markdown(f"### Stage {state['stage'] + 1}: {scenario['desc']}")
            st.markdown(
                f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'>"
                f"Budget: {state['resources']['budget']} | Personnel: {state['resources']['personnel']} | Public Trust: {state['resources']['trust']}"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # JavaScript timer for this stage
            start_time_ms = int(state['last_decision_time'] * 1000)
            js_code = f"""
            <div id="timer" style="text-align:center;font-size:20px;color:#ff7f0e;">Time Left: {time_left:.1f}s</div>
            <script>
            const startTime = {start_time_ms};
            const duration = {scenario['time_limit'] * 1000};
            function updateTimer() {{
                const now = Date.now();
                const elapsed = now - startTime;
                const timeLeft = Math.max(0, duration - elapsed) / 1000;
                document.getElementById("timer").innerText = `Time Left: ${{timeLeft.toFixed(1)}}s`;
                if (timeLeft > 0) {{
                    requestAnimationFrame(updateTimer);
                }}
            }}
            updateTimer();
            </script>
            """
            components.html(js_code, height=50)
            
            # Decision options
            cols = st.columns(3)
            for i, (option, impact, is_risky) in enumerate(scenario['options']):
                if cols[i].button(option, key=f"decision_{state['stage']}_{i}"):
                    reaction_time = time.time() - state['last_decision_time']
                    state['reaction_times'].append(reaction_time)
                    if is_risky:
                        state['risky_choices'] += 1
                    for key, value in impact.items():
                        state['resources'][key] += value
                        if state['resources'][key] < 0:
                            state['errors'] += 1
                            state['resources'][key] = 0
                    state['stage'] += 1
                    state['last_decision_time'] = time.time()
                    st.rerun()
            
            # Auto-advance if time runs out
            if time_left <= 0:
                state['reaction_times'].append(scenario['time_limit'])
                state['resources']['trust'] -= 10
                if state['resources']['trust'] < 0:
                    state['errors'] += 1
                    state['resources']['trust'] = 0
                state['stage'] += 1
                state['last_decision_time'] = time.time()
                st.warning("Time’s up! No decision made.")
                st.rerun()
        
        else:
            total_time = sum(state['reaction_times'])
            avg_reaction_time = total_time / len(state['reaction_times']) if state['reaction_times'] else 0
            final_resources = state['resources']
            predicted_stress = predict_stress_from_game(
                "Crisis Management", avg_reaction_time, state['errors'], state['risky_choices'], 0, final_resources=final_resources
            )
            if not state['result_saved']:
                save_game_result(user_id, "Crisis Management", avg_reaction_time, state['errors'], state['risky_choices'], 0, predicted_stress)
                mark_game_as_played(user_id, "Crisis Management")
                state['result_saved'] = True
            st.markdown(f"### Simulation Complete!")
            if st.session_state.is_admin:
                st.markdown(
                    f"<h4 style='color:#4CAF50'>Results: Total Time: {total_time:.2f}s, Avg Reaction Time: {avg_reaction_time:.2f}s, "
                    f"Risky Choices: {state['risky_choices']}, Errors: {state['errors']}</h4>",
                    unsafe_allow_html=True
                )
                st.markdown(f"<h4 style='color:#1f77b4'>Predicted Stress Level: {predicted_stress}</h4>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'>"
                    f"Final Resources - Budget: {state['resources']['budget']} | Personnel: {state['resources']['personnel']} | Trust: {state['resources']['trust']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.success("Crisis Management Simulation completed! Your results have been saved.")
            if st.button("Play Again", key="crisis_reset"):
                del st.session_state.crisis_state
                st.rerun()

def bart(user_id):
    st.subheader("Balloon Analogue Risk Task (BART)")
    st.write("Pump the balloon to earn points, but it might burst! Cash out to keep points.")
    
    # Check if the user has already played this game
    if has_user_played_game(user_id, "BART"):
        st.warning("You have already played the BART game. You can only play each game once.")
        return
    
    # Initialize bart_state if it doesn't exist
    if 'bart_state' not in st.session_state:
        st.session_state.bart_state = {
            'pumps': 0,
            'burst_point': random.randint(10, 30),
            'points': 0,
            'burst': False,
            'started': False,
            'game_start': None,
            'result_saved': False
        }
    
    state = st.session_state.bart_state
    
    if not state['started']:
        if st.button("Start Game", key="bart_start"):
            state['started'] = True
            state['game_start'] = time.time()
            state['result_saved'] = False
            st.rerun()
    else:
        elapsed_time = time.time() - state['game_start']
        st.markdown(f"<div class='timer'>{int(elapsed_time)}s</div>", unsafe_allow_html=True)
        
        if not state['burst']:
            balloon_size = min(300, 100 + state['pumps'] * 10)
            st.markdown(
                f"<div style='width:{balloon_size}px;height:{balloon_size}px;border-radius:50%;background-color:#ff7f0e;margin:20px auto;animation:bounce 0.5s infinite'></div>",
                unsafe_allow_html=True
            )
            st.markdown(f"<h4 style='text-align:center'>Points: {state['points']}</h4>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            if col1.button("Pump", key=f"pump_{state['pumps']}", help="Increase points, risk bursting"):
                state['pumps'] += 1
                state['points'] += 1
                if state['pumps'] >= state['burst_point']:
                    state['burst'] = True
                    state['points'] = 0
                    st.error("Balloon Burst!")
                else:
                    st.success(f"Pumped! Points: {state['points']}")
                st.rerun()
            if col2.button("Cash Out", key="cash_out", help="Keep your points"):
                predicted_stress = predict_stress_from_game("BART", 0, 0, 0, state['pumps'], burst=False)
                if not state['result_saved']:
                    save_game_result(user_id, "BART", 0, 0, 0, state['pumps'], predicted_stress)
                    mark_game_as_played(user_id, "BART")
                    state['result_saved'] = True
                if st.session_state.is_admin:
                    st.markdown(f"<h4 style='color:#4CAF50'>Cashed Out! Points: {state['points']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color:#1f77b4'>Predicted Stress Level: {predicted_stress}</h4>", unsafe_allow_html=True)
                else:
                    st.success("BART completed! Your points have been saved.")
                del st.session_state.bart_state
                st.rerun()
        else:
            st.markdown(
                f"<div style='width:150px;height:150px;border-radius:50%;background-color:grey;margin:20px auto;'></div>",
                unsafe_allow_html=True
            )
            predicted_stress = predict_stress_from_game("BART", 0, 0, 0, state['pumps'], burst=True)
            if not state['result_saved']:
                save_game_result(user_id, "BART", 0, 0, 0, state['pumps'], predicted_stress)
                mark_game_as_played(user_id, "BART")
                state['result_saved'] = True
            if st.session_state.is_admin:
                st.markdown(f"<h4 style='color:#ff7f0e'>Balloon Burst! Points: 0</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color:#1f77b4'>Predicted Stress Level: {predicted_stress}</h4>", unsafe_allow_html=True)
            else:
                st.warning("Balloon burst! BART completed.")
            if st.button("Play Again", key="bart_reset"):
                del st.session_state.bart_state
                st.rerun()


def show_games_tab():
    st.markdown("""
        <style>
        .game-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .timer {
            font-size: 48px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin: 20px 0;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #45a049;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Fun Challenges")
    
    if st.session_state.is_admin:
        st.subheader("Admin View: Game Results")
        users = get_all_users()
        selected_user = st.selectbox("Select User:", ["All Users"] + users, key="games_user_selector")
        user_id = None if selected_user == "All Users" else get_user_id_by_username(selected_user)
        
        results = load_game_results(user_id)
        if not results:
            st.info("No game results available yet.")
        else:
            df = pd.DataFrame(results)
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            if selected_user == "All Users":
                # Calculate overall stress for each user
                overall_stress_dict = {}
                for user in users:
                    user_id_temp = get_user_id_by_username(user)
                    overall_stress = calculate_overall_stress(user_id_temp)
                    overall_stress_dict[user] = overall_stress if overall_stress else "Not enough data"
                
                # Add overall stress column to the DataFrame
                df['overall_stress'] = df['username'].map(overall_stress_dict)
                
                st.dataframe(
                    df[['username', 'game_type', 'timestamp', 'reaction_time', 'errors', 'false_alarms', 'pumps', 'predicted_stress', 'overall_stress']],
                    column_config={
                        "username": "Username",
                        "game_type": "Game",
                        "timestamp": "Date & Time",
                        "reaction_time": "Reaction Time (s)",
                        "errors": "Errors",
                        "false_alarms": "False Alarms",
                        "pumps": "Pumps",
                        "predicted_stress": "Predicted Stress (Game)",
                        "overall_stress": "Overall Stress"
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                # Display overall stress for the selected user above the table
                overall_stress = calculate_overall_stress(user_id)
                if overall_stress:
                    st.markdown(f"### Overall Stress Level for {selected_user}: {overall_stress}")
                else:
                    st.markdown(f"### Overall Stress Level for {selected_user}: Not enough data")
                
                st.dataframe(
                    df[['username', 'game_type', 'timestamp', 'reaction_time', 'errors', 'false_alarms', 'pumps', 'predicted_stress']],
                    column_config={
                        "username": "Username",
                        "game_type": "Game",
                        "timestamp": "Date & Time",
                        "reaction_time": "Reaction Time (s)",
                        "errors": "Errors",
                        "false_alarms": "False Alarms",
                        "pumps": "Pumps",
                        "predicted_stress": "Predicted Stress"
                    },
                    hide_index=True,
                    use_container_width=True
                )
    else:
        st.markdown("<div class='game-container'>", unsafe_allow_html=True)
        st.subheader("Play a Game")
        
        # Check which games the user has played
        user_id = st.session_state.user_id
        games = ["Stroop Test", "Trail Making Test", "BART"]
        game_status = {game: has_user_played_game(user_id, game if game != "Trail Making Test" else "Crisis Management") for game in games}
        
        # Display game status
        st.markdown("### Game Status")
        for game in games:
            status = "✅ Played" if game_status[game] else "⬜ Not Played"
            st.markdown(f"- {game}: {status}")
        
        # Filter available games (not played yet)
        available_games = [game for game in games if not game_status[game]]
        if not available_games:
            st.info("You have played all available games! Check back later for more.")
        else:
            game_choice = st.selectbox("Choose a Game", available_games)
            if game_choice == "Stroop Test":
                stroop_test(st.session_state.user_id)
            elif game_choice == "Trail Making Test":
                crisis_management_sim(st.session_state.user_id)
            elif game_choice == "BART":
                bart(st.session_state.user_id)
        st.markdown("</div>", unsafe_allow_html=True)

# Define STRESS_LEVELS at the top of the script (or before show_results_tab)
STRESS_LEVELS = ["No stress", "Low stress", "Average stress", "High stress", "Very high stress"]
def show_results_tab():
    st.title("User Stress Results")
    
    if not st.session_state.is_admin:
        st.warning("This section is for admins only.")
        return
    
    st.subheader("Overall Stress Results")
    users = get_all_users()
    selected_user = st.selectbox("Select User:", ["All Users"] + users, key="results_user_selector")
    user_id = None if selected_user == "All Users" else get_user_id_by_username(selected_user)
    
    # Load BERT models for lie detection and stress detection
    bert_model = AutoModelForSequenceClassification.from_pretrained("./saved_bert_model")
    bert_tokenizer = AutoTokenizer.from_pretrained("./saved_bert_model")
    lie_model = DistilBertForSequenceClassification.from_pretrained("C:/Users/WinX/Downloads/Stress/saved_lie_model")
    lie_tokenizer = DistilBertTokenizer.from_pretrained("C:/Users/WinX/Downloads/Stress/saved_lie_model")
    
    # Calculate overall stress, lie scores, and BERT-based stress for all users
    overall_stress_dict = {}
    for user in users:
        user_id_temp = get_user_id_by_username(user)
        
        # Game stress
        game_stress = calculate_overall_stress(user_id_temp)
        game_score = STRESS_LEVEL_MAP.get(game_stress, 2) if game_stress else None
        
        # Gemini-based stress (from session data)
        sessions = load_all_sessions(user_id_temp)
        gemini_score = None
        if sessions:
            gemini_scores = [STRESS_LEVEL_MAP.get(session['stress_level'], 2) for session in sessions]
            gemini_score = sum(gemini_scores) / len(gemini_scores) if gemini_scores else None
        
        # BERT-based lie detection and stress detection
        lie_scores = []
        bert_stress_scores = []
        for session in sessions:
            chat_history = session['chat_history']
            for role, message in chat_history:
                if role == "user":
                    msg_text = message['text'] if isinstance(message, dict) else message
                    inputs = bert_tokenizer(msg_text, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                    stress_label = outputs.logits.argmax().item()
                    attention_weights = outputs.attentions
                    lie_score, forced_score, forced_type, key_phrases, lie_label = calculate_lie_and_forced_scores(
                        msg_text, outputs, attention_weights, bert_tokenizer, lie_model, lie_tokenizer
                    )
                    # Behavioral data for adjustment
                    behavioral_data = {
                        'response_time': message.get('response_time', session.get('avg_response_time', 0)) if isinstance(message, dict) else 0,
                        'edit_count': message.get('edit_count', len(msg_text.split()) // 5) if isinstance(message, dict) else len(msg_text.split()) // 5,
                        'text': msg_text
                    }
                    adjusted_stress = adjust_stress_with_behavior(stress_label, lie_score, behavioral_data)
                    lie_scores.append(lie_score)
                    bert_stress_scores.append(adjusted_stress)
        
        # Average lie score and map to lie level
        avg_lie_score = sum(lie_scores) / len(lie_scores) if lie_scores else None
        lie_level = "High" if avg_lie_score and avg_lie_score > 0.7 else "Low" if avg_lie_score and avg_lie_score < 0.3 else "Moderate" if avg_lie_score else "Not enough data"
        
        # Average BERT-based stress score and map to stress level
        bert_stress_score = sum(bert_stress_scores) / len(bert_stress_scores) if bert_stress_scores else None
        bert_stress_level = score_to_stress_level(bert_stress_score) if bert_stress_score is not None else "Not enough data"
        
        # Combine available scores for overall stress
        available_scores = [score for score in [game_score, gemini_score, bert_stress_score] if score is not None]
        if available_scores:
            avg_score = sum(available_scores) / len(available_scores)
            overall_stress = score_to_stress_level(avg_score)
        else:
            overall_stress = "Not enough data"
        overall_stress_dict[user] = {
            'stress_level': overall_stress,
            'stress_score': avg_score if available_scores else -1,  # For sorting
            'game_stress': game_stress or "Not enough data",
            'gemini_stress': score_to_stress_level(gemini_score) if gemini_score else "Not enough data",
            'lie_score': avg_lie_score,
            'lie_level': lie_level,
            'bert_stress': bert_stress_level,
            'bert_stress_score': bert_stress_score
        }
    
    if selected_user == "All Users":
        # Prepare DataFrame for all users
        df_overall = pd.DataFrame([
            {
                'username': user,
                'overall_stress': data['stress_level'],
                'stress_score': data['stress_score'],
                'game_stress': data['game_stress'],
                'gemini_stress': data['gemini_stress'],
                'lie_level': data['lie_level'],
                'bert_stress': data['bert_stress']
            }
            for user, data in overall_stress_dict.items()
        ])
        
        # Sort by stress_score in descending order
        df_overall = df_overall.sort_values(by='stress_score', ascending=False)
        
        st.markdown("### Combined Stress Levels (All Users)")
        st.dataframe(
            df_overall[['username', 'overall_stress', 'gemini_stress', 'lie_level', 'bert_stress', 'game_stress']],
            column_config={
                "username": "Username",
                "overall_stress": "Overall Stress Level",
                "gemini_stress": "Gemini-Based Stress Detection",
                "lie_level": "BERT-Based Lie Detection (Lie Level)",
                "bert_stress": "BERT-Based Stress Detection",
                "game_stress": "Game Results"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        # Display selected user's overall stress
        user_data = overall_stress_dict[selected_user]
        st.markdown(f"### Overall Stress Level for {selected_user}: {user_data['stress_level']}")
        
        # Display detailed breakdown
        st.markdown(f"**Gemini-Based Stress Detection:** {user_data['gemini_stress']}")
        st.markdown(f"**BERT-Based Lie Detection (Lie Level):** {user_data['lie_level']}")
        st.markdown(f"**BERT-Based Stress Detection:** {user_data['bert_stress']}")
        st.markdown(f"**Game Results:** {user_data['game_stress']}")
        
        # Detailed Gemini-Based Stress Detection
        st.markdown("#### Gemini-Based Stress Detection Results")
        sessions = load_all_sessions(user_id)
        if not sessions:
            st.info("No chat sessions available.")
        else:
            gemini_results = []
            for session in sessions:
                gemini_results.append({
                    'session_name': session['session_name'],
                    'timestamp': session['timestamp'],
                    'username': session['username'],
                    'stress_level': session['stress_level'],
                    'summary': session['summary']
                })
            df_gemini = pd.DataFrame(gemini_results)
            df_gemini['timestamp'] = df_gemini['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(
                df_gemini[['username', 'session_name', 'timestamp', 'stress_level', 'summary']],
                column_config={
                    "username": "Username",
                    "session_name": "Session ID",
                    "timestamp": "Date & Time",
                    "stress_level": "Stress Level (Gemini)",
                    "summary": "Analysis Summary"
                },
                hide_index=True,
                use_container_width=True
            )
        
        # Detailed BERT-Based Lie Detection
        st.markdown("#### BERT-Based Lie Detection Results")
        if not sessions:
            st.info("No chat sessions available for lie detection.")
        else:
            lie_results = []
            for session in sessions:
                chat_history = session['chat_history']
                for role, message in chat_history:
                    if role == "user":
                        msg_text = message['text'] if isinstance(message, dict) else message
                        inputs = bert_tokenizer(msg_text, return_tensors="pt", truncation=True, padding=True)
                        with torch.no_grad():
                            outputs = bert_model(**inputs)
                        attention_weights = outputs.attentions
                        lie_score, forced_score, forced_type, key_phrases, lie_label = calculate_lie_and_forced_scores(
                            msg_text, outputs, attention_weights, bert_tokenizer, lie_model, lie_tokenizer
                        )
                        lie_results.append({
                            'username': session['username'],
                            'message': msg_text,
                            'timestamp': datetime.fromtimestamp(message['timestamp']) if isinstance(message, dict) and 'timestamp' in message else session['timestamp'],
                            'lie_score': lie_score,
                            'lie_label': lie_label,
                            'forced_type': forced_type
                        })
            if not lie_results:
                st.info("No messages available for lie detection.")
            else:
                df_lie = pd.DataFrame(lie_results)
                df_lie['timestamp'] = df_lie['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(
                    df_lie[['username', 'message', 'timestamp', 'lie_score', 'lie_label', 'forced_type']],
                    column_config={
                        "username": "Username",
                        "message": "Message",
                        "timestamp": "Date & Time",
                        "lie_score": "Lie Score (0-1)",
                        "lie_label": "Lie Label",
                        "forced_type": "Forced Emotion Type"
                    },
                    hide_index=True,
                    use_container_width=True
                )
        
        # Detailed BERT-Based Stress Detection
        st.markdown("#### BERT-Based Stress Detection Results")
        if not sessions:
            st.info("No chat sessions available for stress detection.")
        else:
            bert_stress_results = []
            for session in sessions:
                chat_history = session['chat_history']
                for role, message in chat_history:
                    if role == "user":
                        msg_text = message['text'] if isinstance(message, dict) else message
                        inputs = bert_tokenizer(msg_text, return_tensors="pt", truncation=True, padding=True)
                        with torch.no_grad():
                            outputs = bert_model(**inputs)
                        stress_label = outputs.logits.argmax().item()
                        attention_weights = outputs.attentions
                        lie_score, forced_score, forced_type, key_phrases, lie_label = calculate_lie_and_forced_scores(
                            msg_text, outputs, attention_weights, bert_tokenizer, lie_model, lie_tokenizer
                        )
                        behavioral_data = {
                            'response_time': message.get('response_time', session.get('avg_response_time', 0)) if isinstance(message, dict) else 0,
                            'edit_count': message.get('edit_count', len(msg_text.split()) // 5) if isinstance(message, dict) else len(msg_text.split()) // 5,
                            'text': msg_text
                        }
                        adjusted_stress = adjust_stress_with_behavior(stress_label, lie_score, behavioral_data)
                        adjusted_stress_level = score_to_stress_level(adjusted_stress)
                        bert_stress_results.append({
                            'username': session['username'],
                            'message': msg_text,
                            'timestamp': datetime.fromtimestamp(message['timestamp']) if isinstance(message, dict) and 'timestamp' in message else session['timestamp'],
                            'base_stress': STRESS_LEVELS[stress_label],
                            'lie_score': lie_score,
                            'adjusted_stress': adjusted_stress_level
                        })
            if not bert_stress_results:
                st.info("No messages available for stress detection.")
            else:
                df_bert_stress = pd.DataFrame(bert_stress_results)
                df_bert_stress['timestamp'] = df_bert_stress['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(
                    df_bert_stress[['username', 'message', 'timestamp', 'base_stress', 'lie_score', 'adjusted_stress']],
                    column_config={
                        "username": "Username",
                        "message": "Message",
                        "timestamp": "Date & Time",
                        "base_stress": "Base Stress (BERT)",
                        "lie_score": "Lie Score (0-1)",
                        "adjusted_stress": "Adjusted Stress (BERT)"
                    },
                    hide_index=True,
                    use_container_width=True
                )
        
        # Detailed Game Results
        st.markdown("#### Game Results")
        game_results = load_game_results(user_id)
        if not game_results:
            st.info("No game results available.")
        else:
            df_games = pd.DataFrame(game_results)
            df_games['timestamp'] = df_games['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(
                df_games[['username', 'game_type', 'timestamp', 'reaction_time', 'errors', 'false_alarms', 'pumps', 'predicted_stress']],
                column_config={
                    "username": "Username",
                    "game_type": "Game",
                    "timestamp": "Date & Time",
                    "reaction_time": "Reaction Time (s)",
                    "errors": "Errors",
                    "false_alarms": "False Alarms",
                    "pumps": "Pumps",
                    "predicted_stress": "Predicted Stress"
                },
                hide_index=True,
                use_container_width=True
            )

# Main app
def main():
    init_db()
    
    if not st.session_state.get('authenticated', False):
        show_auth_page()
    else:
        if 'current_chat_session' not in st.session_state:
            st.session_state.current_chat_session = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'viewing_past_session' not in st.session_state:
            st.session_state.viewing_past_session = False
        
        tabs = ["Chat", "Stress Analysis", "BERT Analysis", "Games", "Results"]
        if not st.session_state.is_admin:
            tabs = ["Chat", "Games"]
        
        selected_tab = st.sidebar.selectbox("Select Interface", tabs)
        
        if selected_tab == "Chat":
            show_chat_interface()
        elif selected_tab == "Stress Analysis":
            show_stress_dashboard()
        elif selected_tab == "BERT Analysis":
            show_bert_analysis()
        elif selected_tab == "Games":
            show_games_tab()
        elif selected_tab == "Results":
            show_results_tab()

if __name__ == "__main__":
    main()
