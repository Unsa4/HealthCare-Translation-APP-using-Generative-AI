import os
import whisper
from gtts import gTTS
import streamlit as st
from io import BytesIO
import openai

# Set page configuration
st.set_page_config(
    page_title='HealthCare Translation APP',
    page_icon='🧟🏽‍⚕️🌎',
    layout='centered',
    initial_sidebar_state='auto'
)

# Hide Streamlit footer
hide_streamlit_style = """
<style>
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

def main():
    st.header('HealthCare Translation APP')

    # Language selection
    supported_languages = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Hindi": "hi"
    }
    selected_language = st.selectbox("Select the target language", list(supported_languages.keys()))
    target_lang_code = supported_languages[selected_language]

    # Audio input
    try:
        audio_bytes = st.experimental_audio_input("Record a voice message")
    except Exception as e:
        st.error(f"Error during audio input: {e}")
        return

    if audio_bytes:
        st.audio(audio_bytes)
        st.session_state.audio_bytes = audio_bytes

    # Form for real-time translation
    with st.form('input_form'):
        submit_button = st.form_submit_button(label='Translate', type='primary')
        if submit_button and 'audio_bytes' in st.session_state and st.session_state.audio_bytes.size > 0:
            try:
                # Save audio input as a temporary file
                audio_file_path = "temp_audio.wav"
                with open(audio_file_path, "wb") as audio_file:
                    audio_file.write(st.session_state.audio_bytes.getvalue())

                # Use Whisper model for transcription
                model = whisper.load_model("base")
                result = model.transcribe(audio_file_path)

                # Get the transcribed text
                transcribed_text = result["text"]

                # Use GPT-3.5 Turbo for text translation
                try:
                    prompt = (
                        f"Translate the following text to {selected_language}:\n\n"
                        f"{transcribed_text}"
                    )
                    response = openai.Completion.create(
                        model="gpt-3.5-turbo",  # Using gpt-3.5-turbo for translation
                        prompt=prompt,
                        max_tokens=200,
                        temperature=0.7,
                    )
                    translated_text = response.choices[0].text.strip()

                    st.markdown("***Translation Transcript***")
                    st.text_area('Transcription', translated_text, label_visibility='collapsed')

                    # Convert translated text to speech using TTS model
                    sound_file = BytesIO()
                    tts = gTTS(translated_text, lang=target_lang_code)
                    tts.write_to_fp(sound_file)
                    st.markdown("***Synthesized Speech Translation***")
                    st.audio(sound_file)

                except Exception as e:
                    st.error(f"Error during text translation: {e}")

                # Cleanup temporary audio file
                os.remove(audio_file_path)

            except Exception as e:
                st.error(f"An error occurred while processing the audio: {e}")
        else:
            st.warning('No audio recorded, please ensure your audio was recorded correctly.')

    # Text to speech section
    with st.expander("Text to speech"):
        with st.form('text_to_speech'):
            text_to_speech = st.text_area('Enter text to convert to speech')
            selected_lang_for_tts = st.selectbox("Select the language for Text-to-Speech", list(supported_languages.keys()), key="tts")
            tts_lang_code = supported_languages[selected_lang_for_tts]
            submit_button = st.form_submit_button(label='Convert')
            if submit_button and text_to_speech:
                try:
                    # Convert text to speech using gTTS
                    sound_file = BytesIO()
                    tts = gTTS(text_to_speech, lang=tts_lang_code)
                    tts.write_to_fp(sound_file)
                    st.audio(sound_file)
                except Exception as e:
                    st.error(f"Error during text-to-speech conversion: {e}")
            elif submit_button:
                st.warning("Please enter text to convert to speech.")

if __name__ == '__main__':
    main()
