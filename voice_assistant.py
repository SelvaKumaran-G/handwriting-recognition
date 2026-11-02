"""
DigitVision AI - Voice Assistant Module
Provides text-to-speech feedback using browser SpeechSynthesis API in a clean White & Blue style.
Announces prediction results, confidence scores, and explanations clearly.
"""

import streamlit as st
import streamlit.components.v1 as components
from config import DIGIT_WORDS


def get_prediction_speech_text(predicted_digit: int, confidence: float, is_uncertain: bool = False) -> str:
    """
    Constructs the natural speech announcement text for a given prediction.
    """
    if predicted_digit < 0 or predicted_digit > 9:
        return "Prediction could not be determined. Please provide a clearer handwritten image."

    digit_word = DIGIT_WORDS[predicted_digit]
    confidence_pct = round(confidence * 100, 1)

    if is_uncertain:
        return f"Warning. The detected digit appears to be {digit_word}, but confidence is low at {confidence_pct} percent. Please upload a clearer image."

    return f"The detected digit is {digit_word}. The model confidence is {confidence_pct} percent."


def render_voice_assistant(
    predicted_digit: int = 0,
    confidence: float = 0.98,
    is_uncertain: bool = False,
    auto_speak: bool = False,
    key_suffix: str = "main",
    spoken_message: str = None
):
    """
    Renders an interactive voice widget with audio controls, speech rate, and Web Speech API invocation.
    Styled with crisp white and electric-blue design.
    """
    if spoken_message:
        text = spoken_message
    else:
        text = get_prediction_speech_text(predicted_digit, confidence, is_uncertain)
    clean_text = text.replace("'", "\\'").replace('"', '\\"')

    # Embed HTML5 SpeechSynthesis component
    speech_html = f"""
    <div style="
        background: #f8fafc;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="display: flex; align-items: center; gap: 10px; flex: 1;">
            <div style="
                background: linear-gradient(135deg, #2563eb, #3b82f6);
                border-radius: 50%;
                width: 34px;
                height: 34px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                color: #ffffff;
                box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
            ">
                🔊
            </div>
            <div>
                <div style="font-size: 13px; font-weight: 700; color: #1e293b; letter-spacing: 0.2px;">
                    DigitVision Voice Assistant
                </div>
                <div style="font-size: 12px; color: #64748b; font-style: italic;">
                    "{clean_text}"
                </div>
            </div>
        </div>
        <button id="voice-speak-btn-{key_suffix}" onclick="speakResult()" style="
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #ffffff;
            border: none;
            padding: 7px 16px;
            font-size: 12.5px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
            transition: all 0.2s ease;
            white-space: nowrap;
        ">
            🗣️ Speak Result
        </button>
    </div>

    <script>
    function speakResult() {{
        if (!('speechSynthesis' in window)) {{
            alert('Speech synthesis is not supported in this browser.');
            return;
        }}
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance("{clean_text}");
        utterance.rate = 0.95;
        utterance.pitch = 1.05;
        utterance.volume = 1.0;

        const voices = window.speechSynthesis.getVoices();
        const preferred = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Neural')));
        if (preferred) {{
            utterance.voice = preferred;
        }}
        window.speechSynthesis.speak(utterance);
    }}

    {'speakResult();' if auto_speak else ''}
    </script>
    """
    components.html(speech_html, height=80)
