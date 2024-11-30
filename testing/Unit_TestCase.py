import pytest
from unittest.mock import patch, MagicMock
import whisper
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import HtmlTestRunner
import os
import streamlit as st
import whisper
import sys
import sys
sys.path.append(r'c:\Users\HP\Documents\PROGRAMMING\Healthcare Translation App using Generative AI')
from app import main

class TestHealthCareTranslationApp(unittest.TestCase):

    @patch('st.session_state', new_callable=dict)
    def test_audio_input_failure(self, mock_session_state):
        with patch('streamlit.experimental_audio_input', side_effect=Exception("Audio input error")):
            with patch('streamlit.error') as mock_error:
                from app import main
                main()
                mock_error.assert_called_with("Error during audio input: Audio input error")
    
    @patch('whisper.load_model')
    @patch('streamlit.session_state', new_callable=dict)
    def test_successful_audio_transcription(self, mock_session_state, mock_whisper_model):
        # Mock Whisper transcription
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        mock_model.transcribe.return_value = {"text": "Hello, this is a test transcription"}
        
        audio_data = BytesIO(b"fake audio data")
        mock_session_state['audio_bytes'] = audio_data
        
        with patch('streamlit.experimental_audio_input', return_value=audio_data):
            with patch('streamlit.audio') as mock_audio, \
                    patch('streamlit.text_area') as mock_text_area:
                from app import main
                main()
                mock_audio.assert_called_once()
                mock_text_area.assert_called_with('Transcription', 'Hello, this is a test transcription', label_visibility='collapsed')

    @patch('gtts.gTTS.write_to_fp')
    @patch('streamlit.text_area', return_value="Sample text")
    def test_text_to_speech_conversion(self, mock_text_area, mock_gtts):
        with patch('streamlit.audio') as mock_audio:
            with patch('streamlit.form_submit_button', return_value=True):
                from app import main
                main()
                mock_gtts.assert_called_once()
                mock_audio.assert_called_once()
    
    @patch('openai.Completion.create')
    @patch('streamlit.text_area', return_value="Hello world")
    def test_translation_with_gpt(self, mock_text_area, mock_openai):
        mock_openai.return_value = MagicMock(choices=[MagicMock(text="Hola mundo")])
        
        with patch('streamlit.text_area') as mock_translated_text_area:
            with patch('streamlit.form_submit_button', return_value=True):
                from app import main
                main()
                mock_translated_text_area.assert_called_with('Transcription', 'Hola mundo', label_visibility='collapsed')

    def test_ui_elements_presence(self):
        with patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.text_area') as mock_text_area, \
             patch('streamlit.header') as mock_header:
            from app import main
            main()
            mock_header.assert_called_with('HealthCare Translation APP')
            mock_selectbox.assert_called()
            mock_text_area.assert_called()

if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test-reports'))
