import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os
from dotenv import load_dotenv

load_dotenv()

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )

    def speak(self, text):
        """Convert text to speech and play it"""
        print("Aiva:", text)
        audio = self.tts_client.text_to_speech.convert(
            text=text,
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        play(audio)

    def get_voice_input(self):
        """Get voice input from microphone"""
        with sr.Microphone() as source:
            print("Ascult...")
            # Adjust for ambient noise with a longer sample
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Set more lenient energy threshold
            self.recognizer.energy_threshold = 300
            
            # Set longer phrase time limit
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1.0  # Wait longer for silence
            
            try:
                # Listen with longer timeout and phrase time limit
                audio = self.recognizer.listen(
                    source,
                    timeout=10,  # Wait up to 10 seconds for speech to start
                    phrase_time_limit=25  # Allow up to 25 seconds of speech
                )
                
                try:
                    text = self.recognizer.recognize_google(audio, language='ro-RO')
                    print(f"Ai spus: {text}")
                    return text
                except sr.UnknownValueError:
                    print("Îmi pare rău, nu am putut înțelege ce ai spus.")
                    return None
                except sr.RequestError as e:
                    print(f"Nu am putut obține rezultate; {e}")
                    return None
                    
            except sr.WaitTimeoutError:
                print("Nu am detectat niciun sunet în perioada de așteptare.")
                return None
            except Exception as e:
                print(f"Eroare în timpul ascultării: {str(e)}")
                return None 