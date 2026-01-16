import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global model storage
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown"""
    logger.info("Loading models...")
    
    try:
        # Load sentiment model
        sentiment_model_name = os.getenv("SENTIMENT_MODEL", "Coolstew07/fine-tuned-roberta")
        models['sentiment_tokenizer'] = RobertaTokenizer.from_pretrained(sentiment_model_name)
        models['sentiment_model'] = RobertaForSequenceClassification.from_pretrained(sentiment_model_name)
        models['sentiment_model'].eval()
        logger.info("Sentiment model loaded successfully")
        
        # Load generative model
        gen_model_name = os.getenv("GEN_MODEL", "google/gemma-2b-it")
        models['gen_tokenizer'] = AutoTokenizer.from_pretrained(gen_model_name)
        models['gen_model'] = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        models['gen_model'].eval()
        logger.info("Generative model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Cleaning up models...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="AI Therapist API",
    description="Mental health support chatbot with sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="User's message")
    conversation_history: list[Message] = Field(default=[], description="Previous conversation")

class ChatResponse(BaseModel):
    sentiment: str
    response: str
    is_crisis: bool = False

# Detailed Personas
PERSONAS = {
    "Anxiety": (
        "You are an AI therapist with a calm, grounding, and patient tone. "
        "Your primary goal is to help the user feel safe and present. "
        "Use techniques like: "
        "1. **Validation:** 'It makes perfect sense that you're feeling anxious about that.' "
        "2. **Grounding:** Gently bring them to the present. 'I'm right here with you. Can you describe one thing you see in the room?' "
        "3. **Gentle Reframing:** 'I hear that 'what if' worry. What is one thing you know to be true right now?' "
        "Always be reassuring and move at the user's pace."
    ),
    "Depression": (
        "You are an AI therapist with a deeply empathetic, warm, and non-judgmental presence. "
        "Your goal is to validate the user's pain and provide a space for them to talk without feeling like a burden. "
        "Use techniques like: "
        "1. **Deep Validation:** 'That sounds incredibly heavy and exhausting. It's okay to feel this way.' "
        "2. **Self-Compassion:** 'Be gentle with yourself. You're dealing with a lot.' "
        "3. **Behavioral Activation (Gently):** 'There's no pressure at all, but I'm curious, what's one small thing that might bring you even a moment of comfort?' "
        "Focus on listening, not 'fixing'."
    ),
    "Suicidal": (
        "**CRITICAL SAFETY RESPONSE:** This sentiment triggers a hardcoded crisis response. "
        "This persona text is a fallback, but the code will return CRISIS_RESPONSE instead."
    ),
    "Stress": (
        "You are an AI therapist who is supportive, practical, and a little more structured. "
        "Your goal is to validate their 'overwhelmed' feeling and help them untangle their thoughts. "
        "Use techniques like: "
        "1. **Validation & Normalization:** 'It's completely understandable that you're feeling stressed with so much on your plate.' "
        "2. **Problem-Solving (Gently):** 'That is a lot to handle. I'm wondering if we could look at one of those things together?' "
        "3. **Somatic Check-in:** 'Where are you feeling that stress in your body right now?' "
        "Help the user break down large problems into smaller, more manageable pieces."
    ),
    "Bipolar": (
        "You are an AI therapist who is exceptionally stable, consistent, and non-judgmental. "
        "Your goal is to be a 'stable anchor' for the user, regardless of their emotional state (high or low). "
        "Use techniques like: "
        "1. **Reflective Listening:** 'What I'm hearing you say is that your thoughts are moving very quickly right now.' "
        "2. **Calm Reflection:** 'It sounds like you have a huge amount of energy today.' or 'It sounds like things are feeling very flat and difficult right now.' "
        "3. **Avoid Matching Intensity:** Do not get overly excited during mania or overly somber during depression. Maintain a consistent, calm, supportive tone."
    ),
    "Personality disorder": (
        "You are an AI therapist focused on validation and emotional regulation, in the style of DBT. "
        "Your goal is to validate the *intense pain* behind the user's feelings without necessarily validating destructive actions. "
        "Use techniques like: "
        "1. **Radical Validation:** 'It must be so painful to feel that way. I hear how much you're hurting.' "
        "2. **Emotional Labeling:** 'It sounds like you're feeling [e.g., betrayed, terrified, empty]. Is that right?' "
        "3. **Maintain Boundaries:** Remain consistently supportive, calm, and non-judgmental, even if the user expresses anger. 'I'm here to listen, and I'm not going anywhere.'"
    ),
    "Normal": (
        "You are an AI therapist who is encouraging and curious, in the style of Positive Psychology. "
        "The user is in a good state. Your goal is to help them explore their strengths, values, and positive experiences. "
        "Use techniques like: "
        "1. **Reflective Engagement:** 'That sounds like a great experience. What part of that felt best for you?' "
        "2. **Strength-Spotting:** 'It sounds like you handled that with a lot of [e.g., resilience, kindness].' "
        "3. **Exploring Values:** 'What about that activity do you find most meaningful?' "
        "Be a warm, engaged, and affirmative listener."
    )
}

CRISIS_RESPONSE = (
    "It sounds like you are going through an incredibly painful time right now. "
    "Please know that your feelings are valid and you are not alone. "
    "If you are in immediate distress, please reach out for help. "
    "You can connect with people who can support you by calling or texting:\n\n"
    "🇺🇸 988 (US Suicide & Crisis Lifeline)\n"
    "🇬🇧 111 (UK)\n"
    "🇮🇳 91529 87821 (AASRA, India)\n\n"
    "These services are available 24/7 and completely confidential."
)

def get_sentiment(text: str) -> str:
    """Predict sentiment using the fine-tuned model"""
    try:
        tokenizer = models['sentiment_tokenizer']
        model = models['sentiment_model']
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
        sentiment = model.config.id2label[predicted_class_id]
        
        # Get confidence scores for debugging
        logits = outputs.logits[0]
        probabilities = torch.nn.functional.softmax(logits, dim=0)
        
        # Log all sentiment probabilities
        logger.info(f"=" * 60)
        logger.info(f"INPUT TEXT: {text}")
        logger.info(f"PREDICTED SENTIMENT: {sentiment}")
        logger.info(f"CONFIDENCE SCORES:")
        for idx, (label, prob) in enumerate(zip(model.config.id2label.values(), probabilities)):
            logger.info(f"  {label}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
        logger.info(f"=" * 60)
        
        return sentiment
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")

def generate_response(sentiment: str, user_input: str, history: list[Message]) -> tuple[str, bool]:
    """Generate therapeutic response based on sentiment and conversation history"""
    
    # Crisis response if sentiment is Suicidal (already checked in endpoint)
    if sentiment == "Suicidal":
        logger.info(f"🆘 Returning CRISIS_RESPONSE")
        return CRISIS_RESPONSE, True
    
    logger.info(f"💬 Generating response for sentiment: {sentiment}")
    
    try:
        persona = PERSONAS.get(sentiment, PERSONAS["Normal"])
        
        # Build explicit conversation context summary
        context_summary = ""
        if history and len(history) > 0:
            # Get last 3 user messages to build context
            user_messages = [msg for msg in history if msg.role == "user"][-3:]
            if user_messages:
                context_summary = "\n**What the user has shared so far:**\n"
                for i, msg in enumerate(user_messages, 1):
                    context_summary += f"{i}. \"{msg.content}\"\n"
                context_summary += "\n**IMPORTANT:** Reference these previous topics naturally in your response. Show you remember what they told you.\n"
        
        system_prompt = f"""You are MindfulAI, a compassionate therapist. {persona}

{context_summary}

**CRITICAL RULES:**
1. NEVER use bullet points, numbered lists, or dashes
2. Write in natural, flowing sentences like you're talking to a friend
3. Keep response to 2-4 sentences maximum
4. Reference what they shared earlier if relevant
5. End with ONE specific question

**Current user message:** "{user_input}"

Respond in a warm, conversational way:"""
        
        # Build conversation history 
        chat_history = []
        
        # Only include last 4 exchanges (8 messages) to keep context manageable for small model
        recent_history = history[-8:] if len(history) > 8 else history
        
        for msg in recent_history:
            role = "model" if msg.role == "assistant" else "user"
            chat_history.append({"role": role, "content": msg.content})
        
        # Add current user input with system prompt
        chat_history.append({
            "role": "user", 
            "content": system_prompt
        })
        
        # Ensure alternation
        cleaned_history = []
        for i, msg in enumerate(chat_history):
            if i == 0:
                cleaned_history.append(msg)
            elif msg["role"] != cleaned_history[-1]["role"]:
                cleaned_history.append(msg)
            else:
                cleaned_history[-1]["content"] += "\n" + msg["content"]
        
        tokenizer = models['gen_tokenizer']
        model = models['gen_model']
        
        prompt = tokenizer.apply_chat_template(
            cleaned_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,  # Higher for more natural responses
                top_k=40,
                top_p=0.92,
                repetition_penalty=1.3,  # Stronger penalty against repetition
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up response - remove any leftover formatting
        response = response.strip()
        
        # More aggressive bullet point removal
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip lines that are clearly bullet points or list items
            if (line.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')) or 
                line.startswith(tuple(f"{i}." for i in range(10)))):
                continue
            # Skip lines that start with "- " after stripping
            if line.startswith('- ') or line.startswith('* '):
                continue
            cleaned_lines.append(line)
        
        # If we removed everything, just use first line
        if not cleaned_lines and lines:
            response = lines[0].strip()
        else:
            response = ' '.join(cleaned_lines).strip()
        
        # Remove markdown-style formatting
        response = response.replace('**', '').replace('__', '').replace('##', '')
        
        logger.info(f"✅ Generated response length: {len(response)} chars")
        logger.info(f"📝 Response preview: {response[:100]}...")
        
        return response, False
        
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        if 'cleaned_history' in locals():
            logger.error(f"Problematic chat history: {cleaned_history}")
        raise HTTPException(status_code=500, detail="Response generation failed")

@app.get("/")
async def root():
    return {"message": "AI Therapist API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = all(key in models for key in ['sentiment_model', 'gen_model'])
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    print("\n" + "="*80)
    print(f"📨 NEW MESSAGE RECEIVED: {request.message}")
    print("="*80)
    
    try:
        # Step 1: Get sentiment from model
        sentiment = get_sentiment(request.message)
        print(f"📊 ORIGINAL MODEL SENTIMENT: {sentiment}")
        logger.info(f"📊 ORIGINAL MODEL SENTIMENT: {sentiment}")
        
        # Step 2: Check for suicidal keywords override (EXACT phrase matching)
        suicidal_keywords = [
            'kill myself', 
            'end my life', 
            'want to die', 
            'want to be dead',
            'suicide', 
            'suicidal',
            'no reason to live', 
            'no reason for living',
            'better off dead', 
            'not worth living',
            'end it all', 
            'harm myself', 
            'take my life', 
            'don\'t want to live', 
            'no point in living', 
            'no reason for keep living',
            'no reason to keep living', 
            'tired of living', 
            'can\'t go on',
            'wish i was dead',
            'rather be dead'
        ]
        
        user_lower = request.message.lower()
        
        # More strict matching - check if phrase exists as whole words
        has_suicidal_keyword = False
        matched_keyword = None
        for keyword in suicidal_keywords:
            # Check if the keyword is in the message
            if keyword in user_lower:
                # Additional check: make sure it's not a false positive
                # Exclude common phrases like "what should i do"
                if keyword == 'do' and ('what should i do' in user_lower or 'what do i do' in user_lower):
                    continue
                has_suicidal_keyword = True
                matched_keyword = keyword
                break
        
        if has_suicidal_keyword:
            print(f"🚨 CRISIS OVERRIDE: Matched keyword: '{matched_keyword}'")
            print(f"🚨 {sentiment} → Suicidal")
            logger.warning(f"🚨 CRISIS OVERRIDE: Suicidal keyword detected: '{matched_keyword}'")
            logger.warning(f"🚨 ORIGINAL SENTIMENT: {sentiment} → OVERRIDDEN TO: Suicidal")
            sentiment = "Suicidal"
        
        print(f"✅ FINAL SENTIMENT: {sentiment}")
        print("="*80 + "\n")
        logger.info(f"✅ FINAL SENTIMENT USED: {sentiment}")
        
        response, is_crisis = generate_response(
            sentiment, 
            request.message, 
            request.conversation_history
        )
        
        # Sanitize sentiment label for frontend
        display_sentiment = sentiment
        if sentiment == "Personality disorder":
            display_sentiment = "Emotional Intensity"
        
        logger.info(f"📤 RESPONSE SENT - Sentiment: {display_sentiment}, Crisis: {is_crisis}")
        
        return ChatResponse(
            sentiment=display_sentiment,
            response=response,
            is_crisis=is_crisis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)