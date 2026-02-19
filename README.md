# AI Conversational Chatbot for Stress Detection 🧠

A comprehensive AI-powered stress monitoring platform built with Streamlit that combines conversational AI, cognitive assessment games, and advanced machine learning models to detect and track user stress levels with 85% accuracy in real-time.

## 🏆 Key Achievements

- ✅ **85% Accuracy**: Built AI chatbot using Gemini API with 85% accuracy in stress prediction through empathetic, context-aware conversations
- ✅ **Fine-tuned DistilBERT Models**: Achieved 61.8% accuracy in stress detection and 64.64% accuracy in lie detection
- ✅ **Cognitive Game Integration**: Integrated three scientifically-designed games (Stroop Test, BART, Crisis Simulation) for behavioral stress assessment
- ✅ **Real-Time Dashboard**: Developed comprehensive admin dashboard for ethical stress monitoring with interactive insight visualization
- ✅ **Multi-Modal Analysis**: Combined conversational AI, cognitive games, and BERT models for holistic stress assessment

## 🌟 Overview

This system provides a multi-modal approach to stress assessment through:
- **AI Chatbot Conversations** with Gemini API achieving **85% accuracy** in stress prediction
- **Cognitive Assessment Games** for behavioral stress analysis
- **Fine-tuned DistilBERT Models** for stress detection (61.8% accuracy) and lie detection (64.64% accuracy)
- **Real-time Admin Dashboard** for ethical stress monitoring and insight visualization
- **Context-Aware Analysis** through empathetic, intelligent conversations

## ✨ Key Features

### 1. Intelligent Chatbot (Zenith)
- **Gemini API Integration**: Leverages Google's Gemini API for natural language understanding
- **85% Stress Prediction Accuracy**: Highly accurate stress level classification through conversation analysis
- **Empathetic Conversations**: Context-aware, supportive dialogue that builds user trust
- **Background Stress Analysis**: Analyzes conversation patterns to detect stress levels (results visible to administrators only)
- **Session Management**: Users can save, resume, and review past chat sessions
- **Sentiment Analysis**: Tracks emotional states throughout conversations for admin monitoring

### 2. Cognitive Assessment Games
Three scientifically-designed cognitive games to assess stress through behavioral patterns. Users play the games, and performance metrics are analyzed by administrators:

**Stroop Test**
- Classic cognitive interference task
- Measures processing speed and cognitive flexibility
- Tests attention control under conflicting information
- Analyzes reaction time and error patterns for admin review
- Assesses mental processing efficiency and stress impact

**Balloon Analog Risk Task (BART)**
- Measures risk-taking behavior under uncertainty
- Tracks balloon pumps before explosion
- Assesses decision-making and impulse control
- Evaluates stress coping mechanisms through risk tolerance for admin analysis
- Analyzes strategic thinking under pressure

**Crisis Simulation**
- Simulates high-pressure decision-making scenarios
- Tests response to urgent, stressful situations
- Measures emotional regulation and problem-solving
- Analyzes reaction time and decision quality for admin monitoring
- Evaluates stress response in critical situations

### 3. Advanced AI Analysis

**BERT-Based Stress Detection (61.8% Accuracy)**
- Fine-tuned DistilBERT model for stress classification
- Achieves 61.8% accuracy in stress level prediction
- 5-level stress scale: No stress → Very high stress
- Analyzes linguistic patterns and emotional cues
- Behavioral adjustment based on response time and editing patterns

**Lie Detection System (64.64% Accuracy)**
- Fine-tuned DistilBERT for lie and deception detection
- Achieves 64.64% accuracy in identifying dishonest responses
- Detects forced or dishonest responses through linguistic analysis
- Attention-based analysis using BERT architecture
- Identifies key phrases indicating deception
- Classifies forced emotion types (overly positive/negative)

**Multi-Modal Stress Assessment (Admin-Only)**
- Combines chat analysis, game performance, and behavioral data
- Calculates overall stress scores across all interactions
- Provides comprehensive stress profile for each user
- All analytics and stress predictions visible only to administrators
- Ensures ethical monitoring while maintaining user privacy

### 4. Comprehensive Dashboard & Analytics

**User Interface**
- Access to chat conversations with Zenith
- View and resume previous chat sessions
- Play cognitive assessment games
- Session management (save and load conversations)
- Note: Stress analysis results and game performance metrics are only visible to administrators

**Real-Time Admin Dashboard for Ethical Monitoring**
- Real-time stress monitoring across all users
- Ethical oversight with privacy-preserving analytics
- Detailed BERT analysis results and visualizations
- Interactive charts for insight visualization
- Game performance statistics and trends
- Lie detection reports with context
- Export capabilities for research and reporting
- Comprehensive data views while respecting user privacy

### 5. Secure User Management & Access Control
- User authentication with hashed passwords (SHA-256)
- **Role-Based Access Control**:
  - **Regular Users**: Access to chat interface, game playing, and viewing their own chat history only
  - **Administrators**: Full access to all analytics, stress detection results, BERT analysis, game performance metrics, and user monitoring dashboards
- Isolated user data storage ensuring privacy
- SQLite database for data persistence
- **Privacy-First Design**: Users cannot see their own stress levels or analysis results—only administrators can view these for ethical monitoring purposes

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Streamlit** - Web application framework
- **SQLite3** - Database for user data and results

### AI & Machine Learning
- **Google Gemini AI** - Conversational AI and stress analysis
- **Transformers (Hugging Face)** - BERT models for NLP
- **PyTorch** - Deep learning framework
- **DistilBERT** - Lightweight BERT for stress detection
- **TextBlob** - Sentiment analysis and NLP

### Data Processing & Visualization
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Altair** - Interactive data visualizations
- **Pillow (PIL)** - Image processing

## 📋 Prerequisites

Before running this application, ensure you have:

- Python 3.8 or higher
- pip package manager
- Google Gemini API key
- Minimum 4GB RAM
- Internet connection for AI model downloads

## ⚙️ Installation & Setup

### 1. Clone or Download the Repository

```bash
# If using git
git clone https://github.com/yourusername/stress-detection-system.git
cd stress-detection-system

# Or download and extract the ZIP file
```

### 2. Install Required Dependencies

```bash
# Install all required packages
pip install streamlit
pip install google-generativeai
pip install transformers
pip install torch
pip install pandas
pip install altair
pip install textblob
pip install pillow
pip install numpy
```

Or create a `requirements.txt` file:

```txt
streamlit>=1.28.0
google-generativeai>=0.3.0
transformers>=4.35.0
torch>=2.0.0
pandas>=2.0.0
altair>=5.0.0
textblob>=0.17.0
pillow>=10.0.0
numpy>=1.24.0
```

Then install:
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Open the Python file and add your Google Gemini API key:

```python
# Line 158 in the code
GENAI_API_KEY = 'your-gemini-api-key-here'
```

To get a Gemini API key:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste it into the code

### 4. Run the Application

```bash
streamlit run d5.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🚀 Usage Guide

### For Regular Users

**1. Registration & Login**
- Create a new account with username and password
- Login to access the system
- First-time users start with a clean profile

**2. Chat with Zenith**
- Navigate to the "Chat" tab
- Start a new chat session or continue existing ones
- Have natural conversations about your feelings and experiences
- Zenith will provide empathetic responses and ask follow-up questions
- Your conversations are analyzed in the background (results visible only to administrators)

**3. Play Assessment Games**
- Go to the "Games" tab
- Each game can be played once per session
- Follow the on-screen instructions
- Results are automatically saved and sent for analysis
- Note: Game results and stress predictions are only visible to administrators

**4. Manage Your Sessions**
- View your previous chat sessions
- Resume conversations from where you left off
- Start new sessions anytime
- **Important**: Stress analysis results, game performance metrics, and stress levels are only visible to administrators for ethical monitoring purposes

### For Administrators

**1. Access Admin Features**
- Login with admin credentials
- Additional tabs appear: "Stress Analysis", "BERT Analysis", "Results"

**2. Stress Analysis Dashboard**
- View aggregate statistics across all users
- See stress distribution charts
- Analyze trends and patterns
- Filter by date ranges

**3. BERT Analysis**
- Review detailed NLP analysis results
- Check lie detection scores
- View forced emotion classifications
- Analyze attention weights and key phrases

**4. Results Panel**
- Access all user data in tabular format
- View individual chat sessions
- Review game performance metrics
- Export data for research purposes

## 🔬 How It Works

### Stress Detection Methodology

**1. Conversation Analysis (Gemini API - 85% Accuracy)**
- Leverages Google Gemini API for advanced natural language understanding
- Achieves 85% accuracy in stress prediction through conversation analysis
- Analyzes entire chat conversations for contextual understanding
- Detects topic switches and emotional patterns
- Employs empathetic, context-aware dialogue strategies
- Considers sentiment and response patterns
- Generates stress level classification (5 levels)
- Provides detailed summary of findings with actionable insights

**2. BERT-Based Text Analysis (61.8% Accuracy)**
- Fine-tuned DistilBERT model specifically for stress detection
- Achieves 61.8% accuracy in linguistic stress classification
- Tokenizes user messages for deep semantic processing
- Runs through fine-tuned DistilBERT architecture
- Extracts attention weights for interpretable analysis
- Classifies stress on 5-level scale (No stress → Very high stress)
- Adjusts predictions based on behavioral indicators
- Combines with lie detection for comprehensive assessment

**3. Behavioral Analysis**
- **Response Time**: Longer response times may indicate stress
- **Edit Count**: Frequent edits suggest uncertainty or anxiety
- **Message Length**: Very short or very long messages analyzed
- **Typing Patterns**: Speed and consistency tracked

**4. Game Performance Analysis**
- **Stroop Test**: Cognitive interference and processing speed analyzed
- **BART (Balloon Analog Risk Task)**: Risk-taking behavior indicates stress coping style
- **Crisis Simulation**: Decision-making under pressure and emotional regulation assessed
- Machine learning model predicts stress from game metrics

**5. Lie Detection Process (64.64% Accuracy)**
- Fine-tuned DistilBERT model for deception detection
- Achieves 64.64% accuracy in identifying dishonest responses
- Analyzes semantic coherence and linguistic patterns
- Detects overly positive or negative language (forced emotions)
- Identifies inconsistencies in responses
- Uses attention mechanisms to find suspicious phrases
- Scores on 0-1 scale (0 = truthful, 1 = likely deceptive)
- Enhances overall stress analysis accuracy and reliability

### Stress Level Classification

| Level | Score Range | Indicators |
|-------|-------------|------------|
| No stress | 0.0 - 0.8 | Calm, positive, coherent responses |
| Low stress | 0.8 - 1.6 | Minor concerns, generally stable |
| Average stress | 1.6 - 2.4 | Moderate stressors, coping well |
| High stress | 2.4 - 3.2 | Significant stressors, struggling |
| Very high stress | 3.2 - 4.0 | Severe stress, immediate support needed |

## 🎮 Cognitive Game Mechanics

### Stroop Test
- **Duration**: 30-60 seconds
- **Objective**: Identify the color of the text, not the word itself
- **Challenge**: Words spell color names that may differ from their display color
- **Metrics**: Reaction time, accuracy, error rate, cognitive interference score
- **Scoring**: Faster reactions + higher accuracy = better cognitive control and lower stress

### Balloon Analog Risk Task (BART)
- **Duration**: Until all balloons processed
- **Objective**: Pump balloons to maximize points without popping
- **Metrics**: Average pumps per balloon, explosions, cash-outs, risk tolerance
- **Scoring**: Balanced risk-taking = lower stress and better decision-making

### Crisis Simulation
- **Duration**: Variable based on scenario complexity
- **Objective**: Make optimal decisions under high-pressure, time-sensitive scenarios
- **Challenge**: Handle urgent situations requiring quick thinking and emotional control
- **Metrics**: Decision quality, reaction time, emotional regulation, problem-solving effectiveness
- **Scoring**: Calm, rational decisions under pressure = lower stress and strong coping skills

## 🔐 Security & Privacy

- **Password Hashing**: All passwords hashed using SHA-256
- **User Data Access**: Users can only view their own chat history and sessions
- **Admin-Only Analytics**: All stress predictions, game results, and analysis visible only to administrators
- **Ethical Monitoring**: Privacy-preserving analytics ensure user confidentiality
- **Session Management**: Secure session handling with Streamlit
- **Local Database**: Data stored locally in SQLite (users.db)
- **No External Sharing**: User data never shared without explicit consent
- **Role-Based Access**: Clear separation between user and admin privileges

## 📊 Database Schema

**users Table**
- `id`: Primary key (auto-increment)
- `username`: Unique username
- `password`: SHA-256 hashed password
- `is_admin`: Admin flag (0 or 1)

**game_results Table**
- `id`: Primary key
- `user_id`: Foreign key to users
- `game_type`: Name of the game
- `timestamp`: When game was played
- `reaction_time`: Average reaction time
- `errors`: Number of errors
- `false_alarms`: Number of false positives
- `pumps`: Average balloon pumps (BART)
- `predicted_stress`: AI-predicted stress level

**user_game_plays Table**
- `id`: Primary key
- `user_id`: Foreign key to users
- `game_type`: Name of the game
- `played`: Play count
- Ensures each game played once per session

## 🔧 Configuration Options

### Gemini AI Settings (Lines 160-165)
```python
generation_config = {
    "temperature": 0.5,      # Creativity (0.0-1.0)
    "top_p": 0.95,           # Nucleus sampling
    "top_k": 40,             # Top-k sampling
    "max_output_tokens": 500 # Response length
}
```

### Stress Level Thresholds (Lines 46-55)
Adjust score ranges to change sensitivity:
```python
def score_to_stress_level(score):
    if score < 0.8:
        return "No stress"
    elif score < 1.6:
        return "Low stress"
    # ... etc
```

## 🎯 Use Cases

### Personal Mental Health Tracking
- Monitor your stress levels over time
- Identify patterns and triggers
- Track improvement with interventions
- Maintain a mental health journal

### Clinical Research
- Collect stress data from participants
- Analyze behavioral and linguistic patterns
- Validate new assessment methods
- Study stress in different populations

### Corporate Wellness
- Monitor employee stress levels
- Identify high-risk individuals
- Measure intervention effectiveness
- Create data-driven wellness programs

### Educational Settings
- Track student stress during exams
- Identify struggling students early
- Provide timely support
- Research academic stress factors

## 🌟 Future Enhancements

- [ ] Multi-language support for global accessibility
- [ ] Mobile app version for on-the-go monitoring
- [ ] Integration with wearable devices (heart rate, sleep)
- [ ] Voice-based stress analysis
- [ ] Personalized coping strategy recommendations
- [ ] Professional therapist dashboard integration
- [ ] Advanced visualizations (heatmaps, network graphs)
- [ ] Automated alerts for critical stress levels
- [ ] Export to PDF/CSV for reports
- [ ] Calendar integration for stress patterns
- [ ] Group therapy session support
- [ ] Gamification with achievements and rewards

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Model Enhancement**: Improve BERT fine-tuning for better accuracy
2. **New Games**: Design additional assessment games
3. **UI/UX**: Enhance user interface and experience
4. **Documentation**: Expand technical documentation
5. **Testing**: Add unit tests and integration tests
6. **Accessibility**: Improve support for users with disabilities

## ⚠️ Disclaimer

**Important Medical Disclaimer:**

This system is designed for research and informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

- **Not a Diagnostic Tool**: Results should not be used for clinical diagnosis
- **Seek Professional Help**: If experiencing severe stress or mental health concerns, consult a licensed mental health professional
- **Emergency Situations**: For crisis situations, contact emergency services or crisis hotlines immediately
- **Data Accuracy**: AI predictions may not be 100% accurate
- **Privacy**: While data is stored locally, users should be cautious about sensitive information

## 📄 License

All rights reserved. This project is proprietary software developed for research and educational purposes.

## 👨‍💻 Author

**Sai Akshaya R**
- GitHub: [@Sai2003hub](https://github.com/Sai2003hub)
- LinkedIn: [Sai Akshaya R](https://linkedin.com/in/saiakshaya-r/)
- Email: saiakshaya2003@gmail.com

## 🙏 Acknowledgments

- Google Gemini AI for conversational capabilities
- Hugging Face for transformer models
- Streamlit team for the amazing framework
- Open-source community for various libraries

## 📧 Support & Contact

For questions, issues, or suggestions:
- **Email**: saiakshaya2003@gmail.com
- **GitHub Issues**: Open an issue in the repository
- **LinkedIn**: Connect for professional inquiries

---

**Built with ❤️ for better mental health awareness and support**
