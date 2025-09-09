#Extract the audio from video
import yt_dlp
import whisper
import re
import sqlite3


def download_audio(url, output_file="test_audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file.replace('.mp3', '.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': r'C:\ffmpeg-8.0-essentials_build\\bin\\ffmpeg.exe',  # Explicit path
        'cookiefile': 'cookies.txt',  # For age-restricted videos
        'verbose': True,  # Debug
        'noplaylist': True,  # Ensure single video
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_file
    except Exception as e:
        print(f"Error: {e}")
        return None
    
url = str(input("Enter the video url : "))
audio_file = download_audio(url)
if audio_file:
    print(f"Downloaded {audio_file} successfully!")
else:
    print("Audio download failed.")

#extract the transcript from audio
import whisper
model = whisper.load_model("base")
result = model.transcribe("test_audio.mp3")
with open("transcript.txt","w",encoding="utf-8") as f:
    f.write(result["text"])
print(f"Transcription completed. Transcript saved to transcript.txt")

#Text Cleaning and chunking

def clean_and_chunk_text(text , chunk_size = 500):
    cleaned_set = re.sub(r"\[\d+:\d+:\d+\.\d+.*?\]", "", text)  # removes timestamps
    cleaned_text = re.sub(r"\s+", " ", cleaned_set).strip()
    chunks = [cleaned_text[i:i+chunk_size] for i in range(0, len(cleaned_text), chunk_size)]
    return chunks
#SQLite Storage Database
def store_transcript(video_id,language , chunks):
    conn = sqlite3.connect('transcripts.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS transcripts (video_id  TEXT , language TEXT , chunk_id INTEGER , text TEXT )")
    for i , chunk in enumerate(chunks):
        cursor.execute("INSERT INTO transcripts (video_id , language , chunk_id , text) VALUES (? , ?, ? ,?)", (video_id , language , i , chunk))
    conn.commit()
    conn.close()

#Retrieve the data from database
def retrieve_transcript(video_id , language):
    conn = sqlite3.connect('transcript.db')
    cursor = conn.cursor()
    cursor.execute("Select text from transcripts where video_id = ? and language = ? order by chunk_id", (video_id , language))
    result = cursor.fetchall()
    conn.close()
    text_chunks = [chunk[0] for chunk in result]
    return " ".join(text_chunks)
chunks_from_db = retrieve_transcript(url)
print(chunks_from_db)












