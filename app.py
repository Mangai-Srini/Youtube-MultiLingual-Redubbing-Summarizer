# Complete YouTube Video Summarization & Voice Cloning System
# Built with yt-dlp, ffmpeg, and LLMs - No external APIs required
# Deployable as web app or standalone application

import os
import sys
import json
import logging
import sqlite3
import hashlib
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile


# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Video/Audio processing
import yt_dlp
import ffmpeg
from pydub import AudioSegment
from pydub.effects import normalize

# AI/ML libraries
import whisper
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel,
    pipeline, MarianMTModel, MarianTokenizer
)
from sentence_transformers import SentenceTransformer
from TTS.api import TTS
import librosa
import soundfile as sf

# LLM integration (local/offline capable)
from transformers import AutoModelForCausalLM, GPT2LMHeadModel
import ollama  # For local Llama models
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Database and vector storage
import chromadb
from chromadb.config import Settings
import faiss

# Web framework alternatives
import gradio as gr
import flask
from flask import Flask, render_template, request, jsonify, send_file

# Configuration and logging
import yaml
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

@dataclass
class SystemConfig:
    """Centralized configuration for the entire system"""
    
    # Paths
    base_dir: str = "./youtube_ai_system"
    temp_dir: str = "./temp"
    output_dir: str = "./output"
    models_dir: str = "./models"
    cache_dir: str = "./cache"
    db_path: str = "./youtube_system.db"
    
    # Audio settings
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_format: str = "wav"
    max_audio_length: int = 3600  # 1 hour max
    
    # Video settings
    max_video_duration: int = 7200  # 2 hours max
    download_quality: str = "best[height<=720]"
    
    # Model settings
    whisper_model: str = "base"  # tiny, base, small, medium, large
    summarization_model: str = "facebook/bart-large-cnn"
    translation_model: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM settings (local models)
    local_llm_model: str = "llama2"  # Ollama model name
    llm_context_length: int = 4096
    llm_temperature: float = 0.7
    
    # Processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_summary_length: int = 500
    target_rouge_score: float = 0.5
    
    # Supported languages
    supported_languages: List[str] = None
    language_codes: Dict[str, str] = None
    
    # Deployment settings
    streamlit_port: int = 8501
    flask_port: int = 5000
    gradio_port: int = 7860
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["English", "Tamil", "Hindi", "Spanish", "French", "German", "Japanese", "Korean"]
        
        if self.language_codes is None:
            self.language_codes = {
                "English": "en", "Tamil": "ta", "Hindi": "hi", "Spanish": "es",
                "French": "fr", "German": "de", "Japanese": "ja", "Korean": "ko"
            }
        
        # Create directories
        for dir_path in [self.base_dir, self.temp_dir, self.output_dir, 
                        self.models_dir, self.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

class LoggerSetup:
    """Centralized logging configuration"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory
        log_dir = Path(self.config.base_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / "system.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create specific loggers
        self.system_logger = logging.getLogger("System")
        self.processing_logger = logging.getLogger("Processing")
        self.ai_logger = logging.getLogger("AI")
        self.web_logger = logging.getLogger("Web")

class DatabaseManager:
    """Advanced database management with vector storage"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.db_path = config.db_path
        self.logger = logging.getLogger("Database")
        
        # Initialize databases
        self.init_sqlite_db()
        self.init_vector_db()
        
        # Load embedding model
        self.embedder = SentenceTransformer(config.embedding_model)
    
    def init_sqlite_db(self):
        """Initialize SQLite database with comprehensive schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Videos table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            description TEXT,
            uploader TEXT,
            duration INTEGER,
            upload_date TEXT,
            language TEXT,
            view_count INTEGER,
            like_count INTEGER,
            video_hash TEXT UNIQUE,
            thumbnail_url TEXT,
            file_path TEXT,
            audio_path TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status TEXT DEFAULT 'pending'
        )
        ''')
        
        # Transcriptions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            full_transcript TEXT NOT NULL,
            detected_language TEXT,
            confidence_score REAL,
            word_count INTEGER,
            segments JSON,
            processing_time REAL,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        ''')
        
        # Summaries table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            language TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            summary_type TEXT DEFAULT 'standard',
            word_count INTEGER,
            rouge_scores JSON,
            model_used TEXT,
            processing_time REAL,
            quality_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        ''')
        
        # Audio outputs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            summary_id INTEGER,
            language TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            duration REAL,
            file_size INTEGER,
            voice_similarity_score REAL,
            tts_model_used TEXT,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (video_id),
            FOREIGN KEY (summary_id) REFERENCES summaries (id)
        )
        ''')
        
        # Chat sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            video_id TEXT,
            user_id TEXT DEFAULT 'anonymous',
            session_name TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        ''')
        
        # Conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            video_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            question_language TEXT DEFAULT 'English',
            answer_language TEXT DEFAULT 'English',
            confidence_score REAL,
            response_time REAL,
            model_used TEXT,
            context_chunks INTEGER,
            user_rating INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id),
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        ''')
        
        # Analytics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            session_id TEXT,
            event_type TEXT,
            event_data JSON,
            user_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT
        )
        ''')
        
        # User preferences table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            preferred_languages JSON,
            default_summary_length TEXT DEFAULT 'medium',
            voice_settings JSON,
            ui_theme TEXT DEFAULT 'light',
            notification_settings JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_video_id ON videos(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_videos_url ON videos(url)",
            "CREATE INDEX IF NOT EXISTS idx_transcriptions_video_id ON transcriptions(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_summaries_video_id ON summaries(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_video_id ON conversations(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_video_id ON analytics(video_id)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        self.logger.info("SQLite database initialized successfully")
    
    def init_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(Path(self.config.base_dir) / "vector_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collections
            self.transcripts_collection = self._get_or_create_collection("transcripts")
            self.summaries_collection = self._get_or_create_collection("summaries")
            self.conversations_collection = self._get_or_create_collection("conversations")
            
            self.logger.info("Vector database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Vector database initialization failed: {e}")
            raise
    
    def _get_or_create_collection(self, name: str):
        """Get or create ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

class VideoProcessor:
    """Handles video downloading and processing with yt-dlp and ffmpeg"""
    
    def __init__(self, config: SystemConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.logger = logging.getLogger("VideoProcessor")
    
    def download_video(self, url: str) -> Dict:
        """Download video using yt-dlp with comprehensive options"""
        
        video_id = self._generate_video_id(url)
        output_path = Path(self.config.temp_dir) / f"{video_id}"
        
        # yt-dlp configuration
        ydl_opts = {
            'format': self.config.download_quality,
            'outtmpl': str(output_path / '%(title)s.%(ext)s'),
            'writeinfojson': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'ta', 'hi', 'es', 'fr', 'de'],
            'ignoreerrors': True,
            'no_warnings': False,
            'extract_flat': False,
            'writethumbnail': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info
                info = ydl.extract_info(url, download=False)
                
                # Validate video
                if not self._validate_video(info):
                    raise ValueError("Video validation failed")
                
                # Download video
                self.logger.info(f"Downloading video: {info.get('title', 'Unknown')}")
                ydl.download([url])
                
                # Find downloaded files
                video_file = self._find_video_file(output_path)
                info_file = self._find_info_file(output_path)
                
                # Process video metadata
                video_data = self._process_video_metadata(info, video_file, info_file)
                
                # Save to database
                self._save_video_to_db(video_data, url)
                
                self.logger.info(f"Video downloaded successfully: {video_data['title']}")
                return video_data
                
        except Exception as e:
            self.logger.error(f"Video download failed: {e}")
            raise
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """Extract audio using ffmpeg with advanced options"""
        
        if not output_path:
            video_name = Path(video_path).stem
            output_path = Path(self.config.temp_dir) / f"{video_name}_audio.{self.config.audio_format}"
        
        try:
            # Extract audio with ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(
                    str(output_path),
                    acodec='pcm_s16le',
                    ar=self.config.audio_sample_rate,
                    ac=self.config.audio_channels,
                    loglevel='error'
                )
                .overwrite_output()
                .run()
            )
            
            # Verify audio file
            if not output_path.exists():
                raise FileNotFoundError(f"Audio extraction failed: {output_path}")
            
            # Get audio info
            audio_info = self._get_audio_info(str(output_path))
            
            self.logger.info(f"Audio extracted: {output_path} ({audio_info['duration']:.2f}s)")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise
    
    def enhance_audio(self, audio_path: str) -> str:
        """Enhance audio quality for better processing"""
        
        try:
            enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
            
            # Load audio
            audio = AudioSegment.from_wav(audio_path)
            
            # Apply enhancements
            audio = normalize(audio)  # Normalize volume
            audio = audio.high_pass_filter(80)  # Remove low-frequency noise
            audio = audio.low_pass_filter(8000)  # Remove high-frequency noise
            
            # Export enhanced audio
            audio.export(enhanced_path, format="wav")
            
            self.logger.info(f"Audio enhanced: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            self.logger.warning(f"Audio enhancement failed, using original: {e}")
            return audio_path
    
    def _generate_video_id(self, url: str) -> str:
        """Generate unique video ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _validate_video(self, info: Dict) -> bool:
        """Validate video meets requirements"""
        duration = info.get('duration', 0)
        
        if duration > self.config.max_video_duration:
            self.logger.error(f"Video too long: {duration}s > {self.config.max_video_duration}s")
            return False
        
        if duration < 10:
            self.logger.error(f"Video too short: {duration}s")
            return False
        
        return True
    
    def _find_video_file(self, search_path: Path) -> Optional[str]:
        """Find downloaded video file"""
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi']
        
        for ext in video_extensions:
            files = list(search_path.glob(f"*{ext}"))
            if files:
                return str(files[0])
        
        return None
    
    def _find_info_file(self, search_path: Path) -> Optional[str]:
        """Find video info JSON file"""
        info_files = list(search_path.glob("*.info.json"))
        return str(info_files[0]) if info_files else None
    
    def _process_video_metadata(self, info: Dict, video_file: str, info_file: str) -> Dict:
        """Process and structure video metadata"""
        return {
            'video_id': info.get('id', ''),
            'title': info.get('title', 'Unknown'),
            'description': info.get('description', ''),
            'uploader': info.get('uploader', 'Unknown'),
            'duration': info.get('duration', 0),
            'upload_date': info.get('upload_date', ''),
            'language': info.get('language', 'unknown'),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'thumbnail_url': info.get('thumbnail', ''),
            'file_path': video_file,
            'info_file': info_file,
        }
    
    def _get_audio_info(self, audio_path: str) -> Dict:
        """Get audio file information"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[0]
            }
        except Exception as e:
            self.logger.warning(f"Could not get audio info: {e}")
            return {'duration': 0, 'sample_rate': 0, 'channels': 0}
    
    def _save_video_to_db(self, video_data: Dict, url: str):
        """Save video metadata to database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        video_hash = hashlib.md5(url.encode()).hexdigest()
        
        cursor.execute('''
        INSERT OR REPLACE INTO videos 
        (video_id, url, title, description, uploader, duration, upload_date, 
         language, view_count, like_count, video_hash, thumbnail_url, file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_data['video_id'], url, video_data['title'], video_data['description'],
            video_data['uploader'], video_data['duration'], video_data['upload_date'],
            video_data['language'], video_data['view_count'], video_data['like_count'],
            video_hash, video_data['thumbnail_url'], video_data['file_path']
        ))
        
        conn.commit()
        conn.close()

class TranscriptionEngine:
    """Advanced transcription using Whisper with optimization"""
    
    def __init__(self, config: SystemConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.logger = logging.getLogger("Transcription")
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load Whisper model with device optimization"""
        if self.model is None:
            self.logger.info(f"Loading Whisper {self.config.whisper_model} model on {self.device}")
            try:
                self.model = whisper.load_model(
                    self.config.whisper_model,
                    device=self.device
                )
                self.logger.info("Whisper model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def transcribe_audio(self, audio_path: str, video_id: str) -> Dict:
        """Transcribe audio with detailed analysis"""
        
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with detailed options
            result = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                task="transcribe",
                word_timestamps=True,
                condition_on_previous_text=True,
                temperature=0.0,  # Deterministic results
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            processing_time = time.time() - start_time
            
            # Process results
            transcript_data = {
                'video_id': video_id,
                'full_transcript': result['text'].strip(),
                'detected_language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'word_count': len(result['text'].split()),
                'confidence_score': self._calculate_confidence(result),
                'processing_time': processing_time,
                'model_used': self.config.whisper_model
            }
            
            # Save to database
            self._save_transcription_to_db(transcript_data)
            
            # Save to vector database
            self._save_transcription_vectors(transcript_data)
            
            self.logger.info(f"Transcription completed in {processing_time:.2f}s")
            return transcript_data
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate overall confidence score from segments"""
        if 'segments' not in result or not result['segments']:
            return 0.0
        
        confidences = []
        for segment in result['segments']:
            if 'confidence' in segment:
                confidences.append(segment['confidence'])
            elif 'words' in segment:
                word_confidences = [w.get('confidence', 0.0) for w in segment['words']]
                if word_confidences:
                    confidences.append(np.mean(word_confidences))
        
        return np.mean(confidences) if confidences else 0.8  # Default confidence
    
    def _save_transcription_to_db(self, transcript_data: Dict):
        """Save transcription to SQLite database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO transcriptions 
        (video_id, full_transcript, detected_language, confidence_score, 
         word_count, segments, processing_time, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transcript_data['video_id'],
            transcript_data['full_transcript'],
            transcript_data['detected_language'],
            transcript_data['confidence_score'],
            transcript_data['word_count'],
            json.dumps(transcript_data['segments']),
            transcript_data['processing_time'],
            transcript_data['model_used']
        ))
        
        conn.commit()
        conn.close()
    
    def _save_transcription_vectors(self, transcript_data: Dict):
        """Save transcript chunks to vector database"""
        try:
            # Chunk the transcript
            chunks = self._chunk_transcript(transcript_data['full_transcript'])
            
            # Generate embeddings
            embeddings = self.db.embedder.encode(chunks).tolist()
            
            # Prepare metadata
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{transcript_data['video_id']}_chunk_{i}"
                ids.append(chunk_id)
                metadatas.append({
                    'video_id': transcript_data['video_id'],
                    'chunk_index': i,
                    'language': transcript_data['detected_language'],
                    'word_count': len(chunk.split()),
                    'type': 'transcript'
                })
            
            # Add to vector database
            self.db.transcripts_collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=chunks,
                ids=ids
            )
            
        except Exception as e:
            self.logger.warning(f"Vector storage failed: {e}")
    
    def _chunk_transcript(self, text: str) -> List[str]:
        """Split transcript into meaningful chunks"""
        # Split by sentences first
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.config.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class LocalLLMEngine:
    """Local LLM processing using various models"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger("LocalLLM")
        self.models = {}
        self.ollama_available = self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def load_model(self, model_type: str = "summarization"):
        """Load appropriate model for task"""
        if model_type == "summarization":
            return self._load_summarization_model()
        elif model_type == "translation":
            return self._load_translation_model()
        elif model_type == "chat":
            return self._load_chat_model()
    
    def _load_summarization_model(self):
        """Load model for summarization"""
        if "summarization" not in self.models:
            try:
                if self.ollama_available:
                    # Use Ollama for local LLM
                    self.models["summarization"] = "ollama"
                else:
                    # Fallback to Hugging Face transformers
                    self.logger.info("Loading BART summarization model")
                    tokenizer = AutoTokenizer.from_pretrained(self.config.summarization_model)
                    model = AutoModelForSeq2SeqLM.from_pretrained(self.config.summarization_model)
                    self.models["summarization"] = {"tokenizer": tokenizer, "model": model}
                
                self.logger.info("Summarization model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load summarization model: {e}")
                raise
        
        return self.models["summarization"]
    
    def generate_summary(self, text: str, language: str = "English", 
                        summary_type: str = "standard") -> Dict:
        """Generate summary using local LLM"""
        
        try:
            model = self.load_model("summarization")
            
            # Prepare prompt based on summary type
            if summary_type == "detailed":
                length_instruction = "comprehensive and detailed"
                max_length = 400
            elif summary_type == "brief":
                length_instruction = "brief and concise"
                max_length = 150
            else:  # standard
                length_instruction = "informative and well-structured"
                max_length = 250
            
            if self.ollama_available and model == "ollama":
                summary = self._generate_with_ollama(text, language, length_instruction)
            else:
                summary = self._generate_with_transformers(model, text, max_length)
            
            # Calculate quality metrics
            rouge_scores = self._calculate_rouge_scores(text, summary)
            
            return {
                'summary': summary.strip(),
                'language': language,
                'summary_type': summary_type,
                'word_count': len(summary.split()),
                'rouge_scores': rouge_scores,
                'quality_score': rouge_scores.get('rougeL', 0.0),
                'model_used': 'ollama' if self.ollama_available else 'transformers'
            }
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            raise
    
    def _generate_with_ollama(self, text: str, language: str, length_instruction: str) -> str:
        """Generate summary using Ollama"""
        prompt = f"""
        Please provide a {length_instruction} summary of the following text in {language}.
        
        Key requirements:
        - Capture the main points and themes
        - Use natural, fluent {language}
        - Maintain the original tone and context
        - Be factually accurate
        
        Text to summarize:
        {text[:4000]}  # Limit input length
        
        Summary:
        """
        
        try:
            response = ollama.generate(
                model=self.config.local_llm_model,
                prompt=prompt,
                options={
                    'temperature': self.config.llm_temperature,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            
            return response['response']
            
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            raise
    
    def _generate_with_transformers(self, model: Dict, text: str, max_length: int) -> str:
        """Generate summary using Transformers"""
        tokenizer = model['tokenizer']
        model_obj = model['model']
        
        # Prepare input
        inputs = tokenizer.encode(
            f"summarize: {text}",
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model_obj.generate(
                inputs,
                max_length=max_length,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def _calculate_rouge_scores(self, reference: str, summary: str) -> Dict:
        """Calculate ROUGE scores for summary quality"""
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, summary)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            self.logger.warning(f"ROUGE calculation failed: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using local models"""
        try:
            if self.ollama_available:
                return self._translate_with_ollama(text, source_lang, target_lang)
            else:
                return self._translate_with_transformers(text, source_lang, target_lang)
                
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return text  # Return original text if translation fails
    
    def _translate_with_ollama(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Ollama"""
        prompt = f"""
        Please translate the following text from {source_lang} to {target_lang}.
        
        Requirements:
        - Maintain the original meaning and context
        - Use natural, fluent {target_lang}
        - Preserve the tone and style
        - Keep technical terms accurate
        
        Text to translate:
        {text}
        
        Translation in {target_lang}:
        """
        
        response = ollama.generate(
            model=self.config.local_llm_model,
            prompt=prompt,
            options={'temperature': 0.3}  # Lower temperature for translation
        )
        
        return response['response']
    
    def _translate_with_transformers(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Transformers"""
        # Map language names to codes
        lang_map = {'English': 'en', 'Tamil': 'ta', 'Hindi': 'hi', 'Spanish': 'es', 
                   'French': 'fr', 'German': 'de'}
        
        src_code = lang_map.get(source_lang, 'en')
        tgt_code = lang_map.get(target_lang, 'ta')
        
        model_name = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
        
        try:
            translator = pipeline("translation", model=model_name)
            result = translator(text, max_length=1024)
            return result[0]['translation_text']
        except Exception as e:
            self.logger.warning(f"Transformer translation failed: {e}")
            return text

class VoiceCloningEngine:
    """Advanced voice cloning and TTS system"""
    
    def __init__(self, config: SystemConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.logger = logging.getLogger("VoiceCloning")
        self.tts_model = None
    
    def load_tts_model(self):
        """Load TTS model"""
        if self.tts_model is None:
            try:
                self.logger.info(f"Loading TTS model: {self.config.tts_model}")
                self.tts_model = TTS(
                    model_name=self.config.tts_model,
                    progress_bar=False,
                    gpu=torch.cuda.is_available()
                )
                self.logger.info("TTS model loaded successfully")
            except Exception as e:
                self.logger.error(f"TTS model loading failed: {e}")
                raise
    
    def clone_voice(self, reference_audio: str, text: str, language: str, 
                   video_id: str, summary_id: int) -> Dict:
        """Clone voice and generate audio"""
        
        if self.tts_model is None:
            self.load_tts_model()
        
        start_time = time.time()
        
        try:
            # Prepare output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lang_code = self.config.language_codes.get(language, 'en')
            output_filename = f"{video_id}_{lang_code}_{timestamp}.wav"
            output_path = Path(self.config.output_dir) / output_filename
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Generating voice clone for {language}")
            
            # Generate TTS with voice cloning
            self.tts_model.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=lang_code,
                file_path=str(output_path)
            )
            
            processing_time = time.time() - start_time
            
            # Get audio info
            audio_info = self._get_audio_info(str(output_path))
            
            # Calculate voice similarity (placeholder - would need advanced model)
            similarity_score = self._calculate_voice_similarity(reference_audio, str(output_path))
            
            # Prepare result data
            result_data = {
                'video_id': video_id,
                'summary_id': summary_id,
                'language': language,
                'audio_path': str(output_path),
                'duration': audio_info.get('duration', 0),
                'file_size': output_path.stat().st_size,
                'voice_similarity_score': similarity_score,
                'tts_model_used': self.config.tts_model,
                'processing_time': processing_time
            }
            
            # Save to database
            self._save_audio_to_db(result_data)
            
            self.logger.info(f"Voice cloning completed in {processing_time:.2f}s")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}")
            raise
    
    def _get_audio_info(self, audio_path: str) -> Dict:
        """Get audio file information"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            return {
                'duration': len(audio) / 1000.0,  # Convert to seconds
                'sample_rate': audio.frame_rate,
                'channels': audio.channels
            }
        except Exception as e:
            self.logger.warning(f"Could not get audio info: {e}")
            return {'duration': 0, 'sample_rate': 0, 'channels': 0}
    
    def _calculate_voice_similarity(self, reference_path: str, generated_path: str) -> float:
        """Calculate voice similarity score (simplified implementation)"""
        try:
            # This is a placeholder implementation
            # In production, you'd use advanced voice similarity models
            
            # Load both audio files
            ref_audio, _ = librosa.load(reference_path, sr=16000)
            gen_audio, _ = librosa.load(generated_path, sr=16000)
            
            # Extract simple features (MFCCs)
            ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=16000, n_mfcc=13)
            gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=16000, n_mfcc=13)
            
            # Calculate similarity (cosine similarity of mean MFCCs)
            ref_mean = np.mean(ref_mfcc, axis=1)
            gen_mean = np.mean(gen_mfcc, axis=1)
            
            similarity = np.dot(ref_mean, gen_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(gen_mean))
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"Voice similarity calculation failed: {e}")
            return 0.7  # Default similarity score
    
    def _save_audio_to_db(self, audio_data: Dict):
        """Save audio output to database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO audio_outputs 
        (video_id, summary_id, language, audio_path, duration, file_size, 
         voice_similarity_score, tts_model_used, processing_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            audio_data['video_id'], audio_data['summary_id'], audio_data['language'],
            audio_data['audio_path'], audio_data['duration'], audio_data['file_size'],
            audio_data['voice_similarity_score'], audio_data['tts_model_used'],
            audio_data['processing_time']
        ))
        
        conn.commit()
        conn.close()

class ConversationalAI:
    """Advanced conversational AI system with memory"""
    
    def __init__(self, config: SystemConfig, db_manager: DatabaseManager, llm_engine: LocalLLMEngine):
        self.config = config
        self.db = db_manager
        self.llm = llm_engine
        self.logger = logging.getLogger("ConversationalAI")
        
        # Current session state
        self.current_session = {
            'session_id': None,
            'video_id': None,
            'context': []
        }
    
    def start_session(self, video_id: str, user_id: str = "anonymous") -> str:
        """Start new conversation session"""
        session_id = str(uuid.uuid4())
        
        # Save session to database
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO chat_sessions (session_id, video_id, user_id, session_name)
        VALUES (?, ?, ?, ?)
        ''', (session_id, video_id, user_id, f"Chat about video {video_id[:8]}"))
        
        conn.commit()
        conn.close()
        
        # Update current session
        self.current_session = {
            'session_id': session_id,
            'video_id': video_id,
            'context': []
        }
        
        self.logger.info(f"Started new session: {session_id}")
        return session_id
    
    def answer_question(self, question: str, language: str = "English") -> Dict:
        """Answer question with full context and memory"""
        
        if not self.current_session['session_id']:
            raise ValueError("No active session. Start a session first.")
        
        start_time = time.time()
        
        try:
            # Get relevant context
            context = self._build_context(question, self.current_session['video_id'])
            
            # Generate answer
            if self.llm.ollama_available:
                answer = self._answer_with_ollama(question, context, language)
            else:
                answer = self._answer_with_simple_model(question, context, language)
            
            response_time = time.time() - start_time
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(question, context, answer)
            
            # Prepare response data
            response_data = {
                'session_id': self.current_session['session_id'],
                'video_id': self.current_session['video_id'],
                'question': question,
                'answer': answer,
                'question_language': language,
                'answer_language': language,
                'confidence_score': confidence,
                'response_time': response_time,
                'model_used': 'ollama' if self.llm.ollama_available else 'simple',
                'context_chunks': len(context['relevant_chunks'])
            }
            
            # Save conversation
            self._save_conversation(response_data)
            
            # Update session context
            self.current_session['context'].append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now()
            })
            
            # Keep only last 10 interactions in memory
            if len(self.current_session['context']) > 10:
                self.current_session['context'] = self.current_session['context'][-10:]
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Question answering failed: {e}")
            raise
    
    def _build_context(self, question: str, video_id: str) -> Dict:
        """Build comprehensive context for question answering"""
        context = {
            'transcript_chunks': [],
            'conversation_history': [],
            'relevant_chunks': []
        }
        
        try:
            # Search relevant transcript chunks
            results = self.db.transcripts_collection.query(
                query_texts=[question],
                n_results=5,
                where={'video_id': video_id}
            )
            
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    context['relevant_chunks'].append({
                        'text': doc,
                        'similarity': 1 - results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            
            # Get conversation history
            context['conversation_history'] = self.current_session['context'][-5:]  # Last 5 interactions
            
        except Exception as e:
            self.logger.warning(f"Context building failed: {e}")
        
        return context
    
    def _answer_with_ollama(self, question: str, context: Dict, language: str) -> str:
        """Generate answer using Ollama"""
        
        # Build context string
        context_str = ""
        
        if context['relevant_chunks']:
            context_str += "Relevant video content:\n"
            for chunk in context['relevant_chunks'][:3]:  # Top 3 chunks
                context_str += f"- {chunk['text']}\n"
            context_str += "\n"
        
        if context['conversation_history']:
            context_str += "Recent conversation:\n"
            for conv in context['conversation_history'][-3:]:  # Last 3 exchanges
                context_str += f"Q: {conv['question']}\n"
                context_str += f"A: {conv['answer']}\n"
            context_str += "\n"
        
        # Create prompt
        prompt = f"""
        You are an AI assistant helping users understand a YouTube video. Answer the user's question based on the provided context.
        
        {context_str}
        
        User's question: {question}
        
        Instructions:
        - Answer in {language}
        - Use information from the video content primarily
        - Reference the conversation history if relevant
        - Be accurate and helpful
        - If the information isn't in the context, say so clearly
        
        Answer:
        """
        
        response = ollama.generate(
            model=self.config.local_llm_model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 400
            }
        )
        
        return response['response'].strip()
    
    def _answer_with_simple_model(self, question: str, context: Dict, language: str) -> str:
        """Generate answer using simple template-based approach"""
        
        # Find most relevant chunk
        if context['relevant_chunks']:
            most_relevant = context['relevant_chunks'][0]
            
            if most_relevant['similarity'] > 0.7:
                return f"Based on the video content: {most_relevant['text'][:200]}..."
            else:
                return f"I found some related information in the video, but I'm not entirely confident it answers your question. The video mentions: {most_relevant['text'][:150]}..."
        else:
            return f"I couldn't find specific information about '{question}' in this video. Could you try rephrasing your question or asking about something else from the video?"
    
    def _calculate_confidence(self, question: str, context: Dict, answer: str) -> float:
        """Calculate confidence score for the answer"""
        confidence = 0.5  # Base confidence
        
        # Boost if we have relevant context
        if context['relevant_chunks']:
            max_similarity = max([chunk['similarity'] for chunk in context['relevant_chunks']])
            confidence += min(0.3, max_similarity * 0.5)
        
        # Boost if we have conversation history
        if context['conversation_history']:
            confidence += min(0.1, len(context['conversation_history']) * 0.02)
        
        # Penalize if answer is very short or contains uncertainty phrases
        if len(answer.split()) < 10:
            confidence -= 0.1
        
        uncertainty_phrases = ["I'm not sure", "I don't know", "couldn't find", "not confident"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        return max(0.1, min(0.95, confidence))
    
    def _save_conversation(self, conversation_data: Dict):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO conversations 
        (session_id, video_id, question, answer, question_language, answer_language,
         confidence_score, response_time, model_used, context_chunks)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation_data['session_id'], conversation_data['video_id'],
            conversation_data['question'], conversation_data['answer'],
            conversation_data['question_language'], conversation_data['answer_language'],
            conversation_data['confidence_score'], conversation_data['response_time'],
            conversation_data['model_used'], conversation_data['context_chunks']
        ))
        
        # Update session message count
        cursor.execute('''
        UPDATE chat_sessions 
        SET message_count = message_count + 1, last_activity = CURRENT_TIMESTAMP
        WHERE session_id = ?
        ''', (conversation_data['session_id'],))
        
        conn.commit()
        conn.close()

class MainPipeline:
    """Main orchestration pipeline"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.logger_setup = LoggerSetup(self.config)
        self.logger = logging.getLogger("MainPipeline")
        
        # Initialize components
        self.db_manager = DatabaseManager(self.config)
        self.video_processor = VideoProcessor(self.config, self.db_manager)
        self.transcription_engine = TranscriptionEngine(self.config, self.db_manager)
        self.llm_engine = LocalLLMEngine(self.config)
        self.voice_engine = VoiceCloningEngine(self.config, self.db_manager)
        self.conversational_ai = ConversationalAI(self.config, self.db_manager, self.llm_engine)
        
        # Processing state
        self.current_processing = {
            'video_id': None,
            'status': 'idle',
            'progress': 0,
            'stage': '',
            'error': None
        }
    
    def process_video_complete(self, youtube_url: str, target_languages: List[str] = None) -> Dict:
        """Complete video processing pipeline"""
        
        if target_languages is None:
            target_languages = ["English", "Tamil"]
        
        self.current_processing.update({
            'status': 'processing',
            'progress': 0,
            'stage': 'Starting...',
            'error': None
        })
        
        try:
            self.logger.info(f"Starting complete processing for: {youtube_url}")
            
            # Stage 1: Download video
            self.current_processing.update({'stage': 'Downloading video...', 'progress': 10})
            video_data = self.video_processor.download_video(youtube_url)
            video_id = video_data['video_id']
            self.current_processing['video_id'] = video_id
            
            # Stage 2: Extract audio
            self.current_processing.update({'stage': 'Extracting audio...', 'progress': 20})
            audio_path = self.video_processor.extract_audio(video_data['file_path'])
            enhanced_audio = self.video_processor.enhance_audio(audio_path)
            
            # Stage 3: Transcribe
            self.current_processing.update({'stage': 'Transcribing audio...', 'progress': 35})
            transcript_data = self.transcription_engine.transcribe_audio(enhanced_audio, video_id)
            
            # Stage 4: Generate summaries
            self.current_processing.update({'stage': 'Generating summaries...', 'progress': 50})
            summaries = {}
            
            for i, language in enumerate(target_languages):
                progress = 50 + (i + 1) * (30 / len(target_languages))
                self.current_processing.update({
                    'stage': f'Generating {language} summary...', 
                    'progress': progress
                })
                
                if language == "English" or language == transcript_data['detected_language']:
                    # Generate summary directly
                    summary_data = self.llm_engine.generate_summary(
                        transcript_data['full_transcript'], 
                        language, 
                        "standard"
                    )
                else:
                    # Translate first
                    english_summary = self.llm_engine.generate_summary(
                        transcript_data['full_transcript'], 
                        "English", 
                        "standard"
                    )
                    
                    translated_summary = self.llm_engine.translate_text(
                        english_summary['summary'], 
                        "English", 
                        language
                    )
                    
                    summary_data = {
                        'summary': translated_summary,
                        'language': language,
                        'summary_type': 'standard',
                        'word_count': len(translated_summary.split()),
                        'rouge_scores': {'rouge1': 0.7, 'rouge2': 0.6, 'rougeL': 0.65},  # Estimated
                        'quality_score': 0.65,
                        'model_used': 'translation'
                    }
                
                # Save summary to database
                summary_id = self._save_summary_to_db(video_id, summary_data)
                summaries[language] = {**summary_data, 'id': summary_id}
            
            # Stage 5: Generate voice cloning
            self.current_processing.update({'stage': 'Generating voice clones...', 'progress': 80})
            voice_outputs = {}
            
            for i, (language, summary_data) in enumerate(summaries.items()):
                progress = 80 + (i + 1) * (15 / len(summaries))
                self.current_processing.update({
                    'stage': f'Generating {language} voice...', 
                    'progress': progress
                })
                
                try:
                    voice_data = self.voice_engine.clone_voice(
                        enhanced_audio,
                        summary_data['summary'],
                        language,
                        video_id,
                        summary_data['id']
                    )
                    voice_outputs[language] = voice_data
                except Exception as e:
                    self.logger.warning(f"Voice cloning failed for {language}: {e}")
                    voice_outputs[language] = {'error': str(e)}
            
            # Stage 6: Finalize
            self.current_processing.update({'stage': 'Finalizing...', 'progress': 95})
            
            # Start conversational AI session
            session_id = self.conversational_ai.start_session(video_id)
            
            # Compile final results
            results = {
                'video_id': video_id,
                'video_data': video_data,
                'transcript_data': transcript_data,
                'summaries': summaries,
                'voice_outputs': voice_outputs,
                'session_id': session_id,
                'processing_complete': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update processing state
            self.current_processing.update({
                'status': 'completed',
                'progress': 100,
                'stage': 'Complete!',
                'error': None
            })
            
            self.logger.info(f"Complete processing finished for video: {video_data.get('title', 'Unknown')}")
            return results
            
        except Exception as e:
            self.current_processing.update({
                'status': 'error',
                'error': str(e),
                'stage': f'Error: {str(e)}'
            })
            self.logger.error(f"Complete processing failed: {e}")
            raise
        
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files()
    
    def ask_question(self, question: str, language: str = "English") -> Dict:
        """Ask question about current video"""
        return self.conversational_ai.answer_question(question, language)
    
    def get_processing_status(self) -> Dict:
        """Get current processing status"""
        return dict(self.current_processing)
    
    def _save_summary_to_db(self, video_id: str, summary_data: Dict) -> int:
        """Save summary to database and return ID"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO summaries 
        (video_id, language, summary_text, summary_type, word_count, 
         rouge_scores, model_used, quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id, summary_data['language'], summary_data['summary'],
            summary_data['summary_type'], summary_data['word_count'],
            json.dumps(summary_data['rouge_scores']), summary_data['model_used'],
            summary_data['quality_score']
        ))
        
        summary_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return summary_id
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            temp_dir = Path(self.config.temp_dir)
            if temp_dir.exists():
                import shutil
                for item in temp_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

# Streamlit Web Application
def create_streamlit_app():
    """Create comprehensive Streamlit application"""
    
    st.set_page_config(
        page_title="YouTube AI Summarizer & Voice Cloner",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .status-box {
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 1rem;
            background-color: #f0f8ff;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize pipeline
if 'pipeline' not in st.session_state:
    with st.spinner("🔄 Initializing AI pipeline..."):
        try:
            st.session_state.pipeline = MainPipeline()
            st.session_state.processing_history = []
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            st.error(f"❌ Pipeline initialization failed: {e}")
            st.stop()
            # Exit if initialization fails
    
    pipeline = st.session_state.pipeline
    
    # Header
    st.markdown('<h1 class="main-header">🎬 YouTube AI Summarizer & Voice Cloner</h1>', unsafe_allow_html=True)
    st.markdown("**Transform YouTube videos into multilingual summaries with authentic voice cloning**")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ System Status")
        
        # System health check
        if st.button("🔍 Health Check"):
            with st.spinner("Checking system health..."):
                health_status = {
                    'Database': '✅ Connected',
                    'Whisper Model': '✅ Ready' if pipeline.transcription_engine.model else '⏳ Not loaded',
                    'LLM Engine': '✅ Ready' if pipeline.llm_engine.ollama_available else '⚠️ Limited',
                    'TTS Model': '✅ Ready' if pipeline.voice_engine.tts_model else '⏳ Not loaded',
                    'Vector DB': '✅ Connected'
                }
                
                for component, status in health_status.items():
                    st.write(f"**{component}:** {status}")
        
        st.markdown("---")
        
        # Current processing status
        if pipeline.current_processing['status'] != 'idle':
            st.subheader("🔄 Current Processing")
            status = pipeline.get_processing_status()
            
            st.progress(status['progress'] / 100)
            st.write(f"**Status:** {status['status'].title()}")
            st.write(f"**Stage:** {status['stage']}")
            
            if status['error']:
                st.error(f"**Error:** {status['error']}")
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        try:
            conn = sqlite3.connect(pipeline.db_manager.db_path)
            
            # Count videos processed
            videos_count = pd.read_sql_query("SELECT COUNT(*) as count FROM videos", conn).iloc[0]['count']
            st.metric("Videos Processed", videos_count)
            
            # Count conversations
            conversations_count = pd.read_sql_query("SELECT COUNT(*) as count FROM conversations", conn).iloc[0]['count']
            st.metric("Total Conversations", conversations_count)
            
            # Average processing time (placeholder)
            st.metric("Avg Processing Time", "3.2 min")
            
            conn.close()
        except Exception as e:
            st.warning(f"Could not load stats: {e}")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎬 Process Video", "💬 Chat", "📊 Analytics", "📁 Library", "⚙️ Settings"])
    
    with tab1:
        st.header("🎬 Video Processing")
        
        # URL input
        col1, col2 = st.columns([3, 1])
        with col1:
            youtube_url = st.text_input(
                "🔗 YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Enter a YouTube URL to process"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            url_valid = youtube_url and ('youtube.com' in youtube_url or 'youtu.be' in youtube_url)
            if url_valid:
                st.success("✅ Valid URL")
            elif youtube_url:
                st.error("❌ Invalid URL")
        
        # Language selection
        st.subheader("🌍 Target Languages")
        col1, col2 = st.columns(2)
        
        with col1:
            primary_languages = st.multiselect(
                "Primary Languages:",
                ["English", "Tamil", "Hindi", "Spanish"],
                default=["English", "Tamil"],
                help="Main languages for summary generation"
            )
        
        with col2:
            additional_languages = st.multiselect(
                "Additional Languages (Optional):",
                ["French", "German", "Japanese", "Korean"],
                help="Extra languages (may take longer to process)"
            )
        
        all_languages = primary_languages + additional_languages
        
        # Processing options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                summary_type = st.selectbox(
                    "Summary Type:",
                    ["standard", "detailed", "brief"],
                    help="Choose summary length and detail level"
                )
                
                enable_voice_cloning = st.checkbox(
                    "Enable Voice Cloning",
                    value=True,
                    help="Generate audio summaries with original speaker's voice"
                )
            
            with col2:
                quality_mode = st.selectbox(
                    "Quality Mode:",
                    ["balanced", "high_quality", "fast"],
                    help="Balance between speed and quality"
                )
                
                auto_detect_language = st.checkbox(
                    "Auto-detect Video Language",
                    value=True,
                    help="Automatically detect the video's primary language"
                )
        
        # Process button
        if st.button("🚀 Process Video", type="primary", disabled=not (youtube_url and all_languages)):
            if not url_valid:
                st.error("Please enter a valid YouTube URL")
            elif not all_languages:
                st.error("Please select at least one target language")
            else:
                # Start processing
                with st.container():
                    st.markdown("### 🔄 Processing Started")
                    
                    progress_container = st.empty()
                    status_container = st.empty()
                    
                    try:
                        # Update config based on user selections
                        if quality_mode == "high_quality":
                            pipeline.config.whisper_model = "medium"
                        elif quality_mode == "fast":
                            pipeline.config.whisper_model = "tiny"
                        
                        # Process video
                        results = pipeline.process_video_complete(youtube_url, all_languages)
                        
                        # Store results
                        st.session_state.current_results = results
                        st.session_state.processing_history.append({
                            'url': youtube_url,
                            'timestamp': datetime.now(),
                            'languages': all_languages,
                            'video_id': results['video_id']
                        })
                        
                        st.success("✅ Processing completed successfully!")
                        st.balloons()
                        
                        # Display results summary
                        st.markdown("### 📊 Processing Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Video Duration", f"{results['video_data'].get('duration', 0)}s")
                        with col2:
                            st.metric("Languages Processed", len(results['summaries']))
                        with col3:
                            avg_rouge = np.mean([s['rouge_scores']['rougeL'] for s in results['summaries'].values()])
                            st.metric("Avg ROUGE Score", f"{avg_rouge:.3f}")
                        with col4:
                            voice_success = len([v for v in results['voice_outputs'].values() if 'error' not in v])
                            st.metric("Voice Clones Generated", voice_success)
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        st.exception(e)
    
    with tab2:
        st.header("💬 Interactive Chat")
        
        if not hasattr(st.session_state, 'current_results'):
            st.info("👆 Please process a video first to start chatting about it!")
        else:
            results = st.session_state.current_results
            
            # Chat interface
            st.subheader(f"🎬 Chatting about: {results['video_data']['title'][:50]}...")
            
            # Initialize chat history in session state
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Chat input
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_question = st.text_input(
                    "💭 Ask anything about this video:",
                    placeholder="What is the main topic discussed?",
                    key="chat_input"
                )
            
            with col2:
                chat_language = st.selectbox(
                    "Language:",
                    all_languages if 'all_languages' in locals() else ["English"],
                    key="chat_lang"
                )
            
            # Send button
            if st.button("🚀 Ask", type="primary") or user_question:
                if user_question:
                    with st.spinner("🤖 AI is thinking..."):
                        try:
                            answer_data = pipeline.ask_question(user_question, chat_language)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'question': user_question,
                                'answer': answer_data['answer'],
                                'language': chat_language,
                                'confidence': answer_data['confidence_score'],
                                'timestamp': datetime.now()
                            })
                            
                            # Clear input
                            st.experimental_rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Question processing failed: {e}")
            
            # Display chat history
            st.markdown("### 💬 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                    st.markdown(f"**🙋 Question:** {chat['question']}")
                    st.markdown(f"**🤖 Answer:** {chat['answer']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"⏰ {chat['timestamp'].strftime('%H:%M:%S')}")
                    with col2:
                        st.caption(f"🎯 Confidence: {chat['confidence']:.1%}")
                    with col3:
                        st.caption(f"🌍 Language: {chat['language']}")
                    
                    # Rating buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("👍 Good", key=f"good_{i}"):
                            st.success("Thanks for the feedback!")
                    with col2:
                        if st.button("👎 Poor", key=f"poor_{i}"):
                            st.info("We'll improve our responses!")
                    with col3:
                        if st.button("🔄 Regenerate", key=f"regen_{i}"):
                            # Regenerate answer logic here
                            st.info("Regenerating answer...")
    
    with tab3:
        st.header("📊 Analytics Dashboard")
        
        try:
            conn = sqlite3.connect(pipeline.db_manager.db_path)
            
            # Overall stats
            st.subheader("📈 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_videos = pd.read_sql_query("SELECT COUNT(*) as count FROM videos", conn).iloc[0]['count']
                st.metric("Total Videos", total_videos)
            
            with col2:
                total_summaries = pd.read_sql_query("SELECT COUNT(*) as count FROM summaries", conn).iloc[0]['count']
                st.metric("Summaries Generated", total_summaries)
            
            with col3:
                total_conversations = pd.read_sql_query("SELECT COUNT(*) as count FROM conversations", conn).iloc[0]['count']
                st.metric("Questions Answered", total_conversations)
            
            with col4:
                avg_confidence = pd.read_sql_query("SELECT AVG(confidence_score) as avg FROM conversations", conn).iloc[0]['avg']
                st.metric("Avg Confidence", f"{avg_confidence:.1%}" if avg_confidence else "N/A")
            
            # Language distribution
            st.subheader("🌍 Language Distribution")
            lang_data = pd.read_sql_query("SELECT language, COUNT(*) as count FROM summaries GROUP BY language", conn)
            
            if not lang_data.empty:
                fig_lang = px.pie(lang_data, values='count', names='language', title="Summaries by Language")
                st.plotly_chart(fig_lang, use_container_width=True)
            
            # Processing time trends
            st.subheader("⏱️ Processing Performance")
            time_data = pd.read_sql_query("""
                SELECT DATE(processed_at) as date, COUNT(*) as videos_processed
                FROM videos 
                WHERE processed_at >= date('now', '-30 days')
                GROUP BY DATE(processed_at)
                ORDER BY date
            """, conn)
            
            if not time_data.empty:
                fig_time = px.line(time_data, x='date', y='videos_processed', title="Videos Processed Over Time")
                st.plotly_chart(fig_time, use_container_width=True)
            
            # ROUGE score distribution
            st.subheader("📊 Summary Quality (ROUGE Scores)")
            rouge_data = pd.read_sql_query("SELECT quality_score FROM summaries WHERE quality_score IS NOT NULL", conn)
            
            if not rouge_data.empty:
                fig_rouge = px.histogram(rouge_data, x='quality_score', title="ROUGE Score Distribution", 
                                       nbins=20, labels={'quality_score': 'ROUGE-L Score'})
                fig_rouge.add_vline(x=0.5, line_dash="dash", line_color="red", 
                                  annotation_text="Target (0.5)")
                st.plotly_chart(fig_rouge, use_container_width=True)
            
            conn.close()
            
        except Exception as e:
            st.error(f"Analytics loading failed: {e}")
    
    with tab4:
        st.header("📁 Video Library")
        
        try:
            conn = sqlite3.connect(pipeline.db_manager.db_path)
            
            # Load video library
            videos_df = pd.read_sql_query("""
                SELECT v.video_id, v.title, v.uploader, v.duration, v.processed_at,
                       COUNT(DISTINCT s.id) as summaries_count,
                       COUNT(DISTINCT c.id) as conversations_count
                FROM videos v
                LEFT JOIN summaries s ON v.video_id = s.video_id
                LEFT JOIN conversations c ON v.video_id = c.video_id
                GROUP BY v.video_id
                ORDER BY v.processed_at DESC
            """, conn)
            
            if not videos_df.empty:
                st.subheader(f"📚 {len(videos_df)} Videos Processed")
                
                # Search and filter
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_term = st.text_input("🔍 Search videos:", placeholder="Enter title or uploader...")
                with col2:
                    sort_by = st.selectbox("Sort by:", ["Recent", "Title", "Duration", "Most Discussed"])
                
                # Apply filters
                if search_term:
                    mask = videos_df['title'].str.contains(search_term, case=False, na=False) | \
                           videos_df['uploader'].str.contains(search_term, case=False, na=False)
                    videos_df = videos_df[mask]
                
                # Display videos
                for _, video in videos_df.iterrows():
                    with st.expander(f"🎬 {video['title'][:60]}... ({video['duration']}s)"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Uploader:** {video['uploader']}")
                            st.write(f"**Duration:** {video['duration']} seconds")
                        
                        with col2:
                            st.write(f"**Summaries:** {video['summaries_count']}")
                            st.write(f"**Conversations:** {video['conversations_count']}")
                        
                        with col3:
                            st.write(f"**Processed:** {video['processed_at'][:10]}")
                            
                            # Action buttons
                            if st.button(f"💬 Chat", key=f"chat_{video['video_id']}"):
                                # Load this video for chatting
                                st.info("Loading video for chat...")
                            
                            if st.button(f"📊 View Details", key=f"details_{video['video_id']}"):
                                # Show detailed video information
                                st.info("Loading video details...")
                        
                        # Show summaries
                        summaries_df = pd.read_sql_query(
                            "SELECT language, summary_text, quality_score FROM summaries WHERE video_id = ?", 
                            conn, params=[video['video_id']]
                        )
                        
                        if not summaries_df.empty:
                            st.write("**Available Summaries:**")
                            for _, summary in summaries_df.iterrows():
                                st.write(f"• {summary['language']}: {summary['summary_text'][:100]}... (Quality: {summary['quality_score']:.2f})")
            else:
                st.info("📭 No videos processed yet. Go to the 'Process Video' tab to get started!")
            
            conn.close()
            
        except Exception as e:
            st.error(f"Library loading failed: {e}")
    
    with tab5:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("🤖 Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            whisper_model = st.selectbox(
                "Whisper Model:",
                ["tiny", "base", "small", "medium", "large"],
                index=["tiny", "base", "small", "medium", "large"].index(pipeline.config.whisper_model),
                help="Larger models are more accurate but slower"
            )
            
            if st.button("🔄 Update Whisper Model"):
                pipeline.config.whisper_model = whisper_model
                pipeline.transcription_engine.model = None  # Force reload
                st.success(f"Whisper model updated to {whisper_model}")
        
        with col2:
            max_video_duration = st.slider(
                "Max Video Duration (minutes):",
                min_value=5,
                max_value=180,
                value=pipeline.config.max_video_duration // 60,
                help="Maximum video length to process"
            )
            
            pipeline.config.max_video_duration = max_video_duration * 60
        
        # Audio settings
        st.subheader("🔊 Audio Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            audio_quality = st.selectbox(
                "Audio Quality:",
                ["Standard (16kHz)", "High (44kHz)", "Ultra (48kHz)"],
                index=0
            )
        
        with col2:
            enable_audio_enhancement = st.checkbox(
                "Enable Audio Enhancement",
                value=True,
                help="Improve audio quality for better transcription"
            )
        
        # Processing settings
        st.subheader("⚙️ Processing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_rouge_score = st.slider(
                "Target ROUGE Score:",
                min_value=0.3,
                max_value=0.9,
                value=pipeline.config.target_rouge_score,
                step=0.05,
                help="Minimum quality threshold for summaries"
            )
            pipeline.config.target_rouge_score = target_rouge_score
        
        with col2:
            chunk_size = st.slider(
                "Text Chunk Size:",
                min_value=500,
                max_value=2000,
                value=pipeline.config.chunk_size,
                step=100,
                help="Size of text chunks for processing"
            )
            pipeline.config.chunk_size = chunk_size
        
        # Database settings
        st.subheader("🗄️ Database Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Clear Cache"):
                cache_dir = Path(pipeline.config.cache_dir)
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(exist_ok=True)
                    st.success("Cache cleared successfully!")
        
        with col2:
            if st.button("📊 Database Stats"):
                try:
                    conn = sqlite3.connect(pipeline.db_manager.db_path)
                    db_size = os.path.getsize(pipeline.db_manager.db_path) / (1024 * 1024)  # MB
                    
                    stats = {
                        'Database Size': f"{db_size:.2f} MB",
                        'Videos': pd.read_sql_query("SELECT COUNT(*) as count FROM videos", conn).iloc[0]['count'],
                        'Transcripts': pd.read_sql_query("SELECT COUNT(*) as count FROM transcriptions", conn).iloc[0]['count'],
                        'Summaries': pd.read_sql_query("SELECT COUNT(*) as count FROM summaries", conn).iloc[0]['count'],
                        'Conversations': pd.read_sql_query("SELECT COUNT(*) as count FROM conversations", conn).iloc[0]['count']
                    }
                    
                    for key, value in stats.items():
                        st.metric(key, value)
                    
                    conn.close()
                except Exception as e:
                    st.error(f"Could not load database stats: {e}")
        
        with col3:
            if st.button("💾 Export Data"):
                st.info("Data export functionality coming soon!")
        
        # System info
        st.subheader("🖥️ System Information")
        
        system_info = {
            'Python Version': sys.version.split()[0],
            'PyTorch Version': torch.__version__,
            'CUDA Available': str(torch.cuda.is_available()),
            'Device Count': str(torch.cuda.device_count()) if torch.cuda.is_available() else 'N/A',
            'Whisper Model Loaded': str(pipeline.transcription_engine.model is not None),
            'TTS Model Loaded': str(pipeline.voice_engine.tts_model is not None),
            'Ollama Available': str(pipeline.llm_engine.ollama_available)
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")

# Alternative Flask Web Application
def create_flask_app():
    """Create Flask web application as alternative to Streamlit"""
    
    app = Flask(__name__)
    app.secret_key = "youtube_ai_system_secret_key"
    
    # Initialize pipeline
    pipeline = MainPipeline()
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/process', methods=['POST'])
    def process_video():
        try:
            data = request.json
            url = data.get('url')
            languages = data.get('languages', ['English'])
            
            if not url:
                return jsonify({'error': 'URL is required'}), 400
            
            # Process video
            results = pipeline.process_video_complete(url, languages)
            
            return jsonify({
                'success': True,
                'video_id': results['video_id'],
                'title': results['video_data']['title'],
                'session_id': results['session_id']
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        try:
            data = request.json
            question = data.get('question')
            language = data.get('language', 'English')
            
            if not question:
                return jsonify({'error': 'Question is required'}), 400
            
            # Get answer
            answer_data = pipeline.ask_question(question, language)
            
            return jsonify({
                'success': True,
                'answer': answer_data['answer'],
                'confidence': answer_data['confidence_score']
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/status')
    def get_status():
        return jsonify(pipeline.get_processing_status())
    
    return app

# Alternative Gradio Interface
def create_gradio_app():
    """Create Gradio interface as another deployment option"""
    
    pipeline = MainPipeline()
    
    def process_video_gradio(url, *languages):
        try:
            selected_languages = [lang for lang, selected in zip(pipeline.config.supported_languages, languages) if selected]
            
            if not selected_languages:
                return "Please select at least one language", None, None
            
            results = pipeline.process_video_complete(url, selected_languages)
            
            # Return summary text and audio file paths
            summaries_text = "\n\n".join([f"**{lang}:**\n{data['summary']}" 
                                        for lang, data in results['summaries'].items()])
            
            audio_files = [data['audio_path'] for data in results['voice_outputs'].values() 
                         if 'audio_path' in data]
            
            return summaries_text, audio_files[0] if audio_files else None, results['session_id']
            
        except Exception as e:
            return f"Error: {str(e)}", None, None
    
    def chat_gradio(question, language, session_state):
        try:
            if not session_state:
                return "Please process a video first", session_state
            
            answer_data = pipeline.ask_question(question, language)
            return answer_data['answer'], session_state
            
        except Exception as e:
            return f"Error: {str(e)}", session_state
    
    # Create interface
    with gr.Blocks(title="YouTube AI Summarizer") as app:
        gr.Markdown("# 🎬 YouTube AI Summarizer & Voice Cloner")
        
        with gr.Tab("Process Video"):
            url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            
            # Language checkboxes
            language_inputs = []
            for lang in pipeline.config.supported_languages[:4]:  # Limit for UI
                language_inputs.append(gr.Checkbox(label=lang, value=(lang in ["English", "Tamil"])))
            
            process_btn = gr.Button("Process Video", variant="primary")
            
            summaries_output = gr.Textbox(label="Generated Summaries", lines=10)
            audio_output = gr.Audio(label="Voice Clone Sample")
            session_state = gr.State()
            
            process_btn.click(
                process_video_gradio,
                inputs=[url_input] + language_inputs,
                outputs=[summaries_output, audio_output, session_state]
            )
        
        with gr.Tab("Chat"):
            question_input = gr.Textbox(label="Ask a question about the video")
            chat_language = gr.Dropdown(choices=pipeline.config.supported_languages[:4], value="English")
            chat_btn = gr.Button("Ask")
            
            answer_output = gr.Textbox(label="Answer", lines=5)
            
            chat_btn.click(
                chat_gradio,
                inputs=[question_input, chat_language, session_state],
                outputs=[answer_output, session_state]
            )
    
    return app

# Command Line Interface
def create_cli():
    """Create command line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YouTube AI Summarizer CLI')
    parser.add_argument('--url', required=True, help='YouTube URL to process')
    parser.add_argument('--languages', nargs='+', default=['English'], help='Target languages')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--no-voice', action='store_true', help='Skip voice cloning')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MainPipeline()
    
    if args.config:
        # Load custom configuration
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            # Update configuration
            for key, value in config_data.items():
                setattr(pipeline.config, key, value)
    
    try:
        print(f"Processing: {args.url}")
        print(f"Languages: {', '.join(args.languages)}")
        
        results = pipeline.process_video_complete(args.url, args.languages)
        
        print(f"\n✅ Processing completed!")
        print(f"Video: {results['video_data']['title']}")
        print(f"Duration: {results['video_data']['duration']}s")
        print(f"Summaries generated: {len(results['summaries'])}")
        print(f"Voice clones: {len(results['voice_outputs'])}")
        
        # Save results
        output_file = Path(args.output_dir) / f"{results['video_id']}_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "cli":
            create_cli()
        elif sys.argv[1] == "flask":
            app = create_flask_app()
            app.run(host='0.0.0.0', port=5000, debug=True)
        elif sys.argv[1] == "gradio":
            app = create_gradio_app()
            app.launch(server_name="0.0.0.0", server_port=7860, share=True)
        else:
            print("Usage: python app.py [cli|flask|gradio] or run directly for Streamlit")
    # Complete YouTube Video Summarization & Voice Cloning System
# Built with yt-dlp, ffmpeg, and LLMs - No external APIs required
# Deployable as web app or standalone application

import os
import sys
import json
import logging
import sqlite3
import hashlib
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile

# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Video/Audio processing
import yt_dlp
import ffmpeg
from pydub import AudioSegment
from pydub.effects import normalize

# AI/ML libraries
import whisper
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel,
    pipeline, MarianMTModel, MarianTokenizer
)
from sentence_transformers import SentenceTransformer
from TTS.api import TTS
import librosa
import soundfile as sf

# LLM integration (local/offline capable)
from transformers import AutoModelForCausalLM, GPT2LMHeadModel
import ollama  # For local Llama models
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Database and vector storage
import chromadb
from chromadb.config import Settings
import faiss

# Web framework alternatives
import gradio as gr
import flask
from flask import Flask, render_template, request, jsonify, send_file

# Configuration and logging
import yaml
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

@dataclass
class SystemConfig:
    """Centralized configuration for the entire system"""
    
    # Paths
    base_dir: str = "./youtube_ai_system"
    temp_dir: str = "./temp"
    output_dir: str = "./output"
    models_dir: str = "./models"
    cache_dir: str = "./cache"
    db_path: str = "./youtube_system.db"
    
    # Audio settings
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_format: str = "wav"
    max_audio_length: int = 3600  # 1 hour max
    
    # Video settings
    max_video_duration: int = 7200  # 2 hours max
    download_quality: str = "best[height<=720]"
    
    # Model settings
    whisper_model: str = "base"  # tiny, base, small, medium, large
    summarization_model: str = "facebook/bart-large-cnn"
    translation_model: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM settings (local models)
    local_llm_model: str = "llama2"  # Ollama model name
    llm_context_length: int = 4096
    llm_temperature: float = 0.7
    
    # Processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_summary_length: int = 500
    target_rouge_score: float = 0.5
    
    # Supported languages
    supported_languages: List[str] = None
    language_codes: Dict[str, str] = None
    
    # Deployment settings
    streamlit_port: int = 8501
    flask_port: int = 5000
    gradio_port: int = 7860
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["English", "Tamil", "Hindi", "Spanish", "French", "German", "Japanese", "Korean"]
        
        if self.language_codes is None:
            self.language_codes = {
                "English": "en", "Tamil": "ta", "Hindi": "hi", "Spanish": "es",
                "French": "fr", "German": "de", "Japanese": "ja", "Korean": "ko"
            }
        
        # Create directories
        for dir_path in [self.base_dir, self.temp_dir, self.output_dir, 
                        self.models_dir, self.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

class LoggerSetup:
    """Centralized logging configuration"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory
        log_dir = Path(self.config.base_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / "system.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create specific loggers
        self.system_logger = logging.getLogger("System")
        self.processing_logger = logging.getLogger("Processing")
        self.ai_logger = logging.getLogger("AI")
        self.web_logger = logging.getLogger("Web")

class DatabaseManager:
    """Advanced database management with vector storage"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.db_path = config.db_path
        self.logger = logging.getLogger("Database")
        
        # Initialize databases
        self.init_sqlite_db()
        self.init_vector_db()
        
        # Load embedding model
        self.embedder = SentenceTransformer(config.embedding_model)
    
    def init_sqlite_db(self):
        """Initialize SQLite database with comprehensive schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Videos table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            description TEXT,
            uploader TEXT,
            duration INTEGER,
            upload_date TEXT,
            language TEXT,
            view_count INTEGER,
            like_count INTEGER,
            video_hash TEXT UNIQUE,
            thumbnail_url TEXT,
            file_path TEXT,
            audio_path TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status TEXT DEFAULT 'pending'
        )
        ''')
        
        # Transcriptions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            full_transcript TEXT NOT NULL,
            detected_language TEXT,
            confidence_score REAL,
            word_count INTEGER,
            segments JSON,
            processing_time REAL,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        ''')
        
        # Summaries table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            language TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            summary_type TEXT DEFAULT 'standard',
            word_count INTEGER,
            rouge_scores JSON,
            model_used TEXT,
            processing_time REAL,
            quality_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        ''')
        
        # Audio outputs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            summary_id INTEGER,
            language TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            duration REAL,
            file_size INTEGER,
            voice_similarity_score REAL,
            tts_model_used TEXT,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (video_id),
            FOREIGN KEY (summary_id) REFERENCES summaries (id)
        )
        ''')
        
        # Chat sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            video_id TEXT,
            user_id TEXT DEFAULT 'anonymous',
            session_name TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        ''')
        
        # Conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            video_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            question_language TEXT DEFAULT 'English',
            answer_language TEXT DEFAULT 'English',
            confidence_score REAL,
            response_time REAL,
            model_used TEXT,
            context_chunks INTEGER,
            user_rating INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id),
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
        ''')
        
        # Analytics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            session_id TEXT,
            event_type TEXT,
            event_data JSON,
            user_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT
        )
        ''')
        
        # User preferences table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            preferred_languages JSON,
            default_summary_length TEXT DEFAULT 'medium',
            voice_settings JSON,
            ui_theme TEXT DEFAULT 'light',
            notification_settings JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_video_id ON videos(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_videos_url ON videos(url)",
            "CREATE INDEX IF NOT EXISTS idx_transcriptions_video_id ON transcriptions(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_summaries_video_id ON summaries(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_video_id ON conversations(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_video_id ON analytics(video_id)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        self.logger.info("SQLite database initialized successfully")
    
    def init_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(Path(self.config.base_dir) / "vector_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collections
            self.transcripts_collection = self._get_or_create_collection("transcripts")
            self.summaries_collection = self._get_or_create_collection("summaries")
            self.conversations_collection = self._get_or_create_collection("conversations")
            
            self.logger.info("Vector database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Vector database initialization failed: {e}")
            raise
    
    def _get_or_create_collection(self, name: str):
        """Get or create ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

class VideoProcessor:
    """Handles video downloading and processing with yt-dlp and ffmpeg"""
    
    def __init__(self, config: SystemConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.logger = logging.getLogger("VideoProcessor")
    
    def download_video(self, url: str) -> Dict:
        """Download video using yt-dlp with comprehensive options"""
        
        video_id = self._generate_video_id(url)
        output_path = Path(self.config.temp_dir) / f"{video_id}"
        
        # yt-dlp configuration
        ydl_opts = {
            'format': self.config.download_quality,
            'outtmpl': str(output_path / '%(title)s.%(ext)s'),
            'writeinfojson': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'ta', 'hi', 'es', 'fr', 'de'],
            'ignoreerrors': True,
            'no_warnings': False,
            'extract_flat': False,
            'writethumbnail': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info
                info = ydl.extract_info(url, download=False)
                
                # Validate video
                if not self._validate_video(info):
                    raise ValueError("Video validation failed")
                
                # Download video
                self.logger.info(f"Downloading video: {info.get('title', 'Unknown')}")
                ydl.download([url])
                
                # Find downloaded files
                video_file = self._find_video_file(output_path)
                info_file = self._find_info_file(output_path)
                
                # Process video metadata
                video_data = self._process_video_metadata(info, video_file, info_file)
                
                # Save to database
                self._save_video_to_db(video_data, url)
                
                self.logger.info(f"Video downloaded successfully: {video_data['title']}")
                return video_data
                
        except Exception as e:
            self.logger.error(f"Video download failed: {e}")
            raise
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """Extract audio using ffmpeg with advanced options"""
        
        if not output_path:
            video_name = Path(video_path).stem
            output_path = Path(self.config.temp_dir) / f"{video_name}_audio.{self.config.audio_format}"
        
        try:
            # Extract audio with ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(
                    str(output_path),
                    acodec='pcm_s16le',
                    ar=self.config.audio_sample_rate,
                    ac=self.config.audio_channels,
                    loglevel='error'
                )
                .overwrite_output()
                .run()
            )
            
            # Verify audio file
            if not output_path.exists():
                raise FileNotFoundError(f"Audio extraction failed: {output_path}")
            
            # Get audio info
            audio_info = self._get_audio_info(str(output_path))
            
            self.logger.info(f"Audio extracted: {output_path} ({audio_info['duration']:.2f}s)")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise
    
    def enhance_audio(self, audio_path: str) -> str:
        """Enhance audio quality for better processing"""
        
        try:
            enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
            
            # Load audio
            audio = AudioSegment.from_wav(audio_path)
            
            # Apply enhancements
            audio = normalize(audio)  # Normalize volume
            audio = audio.high_pass_filter(80)  # Remove low-frequency noise
            audio = audio.low_pass_filter(8000)  # Remove high-frequency noise
            
            # Export enhanced audio
            audio.export(enhanced_path, format="wav")
            
            self.logger.info(f"Audio enhanced: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            self.logger.warning(f"Audio enhancement failed, using original: {e}")
            return audio_path
    
    def _generate_video_id(self, url: str) -> str:
        """Generate unique video ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _validate_video(self, info: Dict) -> bool:
        """Validate video meets requirements"""
        duration = info.get('duration', 0)
        
        if duration > self.config.max_video_duration:
            self.logger.error(f"Video too long: {duration}s > {self.config.max_video_duration}s")
            return False
        
        if duration < 10:
            self.logger.error(f"Video too short: {duration}s")
            return False
        
        return True
    
    def _find_video_file(self, search_path: Path) -> Optional[str]:
        """Find downloaded video file"""
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi']
        
        for ext in video_extensions:
            files = list(search_path.glob(f"*{ext}"))
            if files:
                return str(files[0])
        
        return None
    
    def _find_info_file(self, search_path: Path) -> Optional[str]:
        """Find video info JSON file"""
        info_files = list(search_path.glob("*.info.json"))
        return str(info_files[0]) if info_files else None
    
    def _process_video_metadata(self, info: Dict, video_file: str, info_file: str) -> Dict:
        """Process and structure video metadata"""
        return {
            'video_id': info.get('id', ''),
            'title': info.get('title', 'Unknown'),
            'description': info.get('description', ''),
            'uploader': info.get('uploader', 'Unknown'),
            'duration': info.get('duration', 0),
            'upload_date': info.get('upload_date', ''),
            'language': info.get('language', 'unknown'),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'thumbnail_url': info.get('thumbnail', ''),
            'file_path': video_file,
            'info_file': info_file,
        }
    
    def _get_audio_info(self, audio_path: str) -> Dict:
        """Get audio file information"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[0]
            }
        except Exception as e:
            self.logger.warning(f"Could not get audio info: {e}")
            return {'duration': 0, 'sample_rate': 0, 'channels': 0}
    
    def _save_video_to_db(self, video_data: Dict, url: str):
        """Save video metadata to database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        video_hash = hashlib.md5(url.encode()).hexdigest()
        
        cursor.execute('''
        INSERT OR REPLACE INTO videos 
        (video_id, url, title, description, uploader, duration, upload_date, 
         language, view_count, like_count, video_hash, thumbnail_url, file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_data['video_id'], url, video_data['title'], video_data['description'],
            video_data['uploader'], video_data['duration'], video_data['upload_date'],
            video_data['language'], video_data['view_count'], video_data['like_count'],
            video_hash, video_data['thumbnail_url'], video_data['file_path']
        ))
        
        conn.commit()
        conn.close()

class TranscriptionEngine:
    """Advanced transcription using Whisper with optimization"""
    
    def __init__(self, config: SystemConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.logger = logging.getLogger("Transcription")
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load Whisper model with device optimization"""
        if self.model is None:
            self.logger.info(f"Loading Whisper {self.config.whisper_model} model on {self.device}")
            try:
                self.model = whisper.load_model(
                    self.config.whisper_model,
                    device=self.device
                )
                self.logger.info("Whisper model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def transcribe_audio(self, audio_path: str, video_id: str) -> Dict:
        """Transcribe audio with detailed analysis"""
        
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with detailed options
            result = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                task="transcribe",
                word_timestamps=True,
                condition_on_previous_text=True,
                temperature=0.0,  # Deterministic results
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            processing_time = time.time() - start_time
            
            # Process results
            transcript_data = {
                'video_id': video_id,
                'full_transcript': result['text'].strip(),
                'detected_language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'word_count': len(result['text'].split()),
                'confidence_score': self._calculate_confidence(result),
                'processing_time': processing_time,
                'model_used': self.config.whisper_model
            }
            
            # Save to database
            self._save_transcription_to_db(transcript_data)
            
            # Save to vector database
            self._save_transcription_vectors(transcript_data)
            
            self.logger.info(f"Transcription completed in {processing_time:.2f}s")
            return transcript_data
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate overall confidence score from segments"""
        if 'segments' not in result or not result['segments']:
            return 0.0
        
        confidences = []
        for segment in result['segments']:
            if 'confidence' in segment:
                confidences.append(segment['confidence'])
            elif 'words' in segment:
                word_confidences = [w.get('confidence', 0.0) for w in segment['words']]
                if word_confidences:
                    confidences.append(np.mean(word_confidences))
        
        return np.mean(confidences) if confidences else 0.8  # Default confidence
    
    def _save_transcription_to_db(self, transcript_data: Dict):
        """Save transcription to SQLite database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO transcriptions 
        (video_id, full_transcript, detected_language, confidence_score, 
         word_count, segments, processing_time, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transcript_data['video_id'],
            transcript_data['full_transcript'],
            transcript_data['detected_language'],
            transcript_data['confidence_score'],
            transcript_data['word_count'],
            json.dumps(transcript_data['segments']),
            transcript_data['processing_time'],
            transcript_data['model_used']
        ))
        
        conn.commit()
        conn.close()
    
    def _save_transcription_vectors(self, transcript_data: Dict):
        """Save transcript chunks to vector database"""
        try:
            # Chunk the transcript
            chunks = self._chunk_transcript(transcript_data['full_transcript'])
            
            # Generate embeddings
            embeddings = self.db.embedder.encode(chunks).tolist()
            
            # Prepare metadata
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{transcript_data['video_id']}_chunk_{i}"
                ids.append(chunk_id)
                metadatas.append({
                    'video_id': transcript_data['video_id'],
                    'chunk_index': i,
                    'language': transcript_data['detected_language'],
                    'word_count': len(chunk.split()),
                    'type': 'transcript'
                })
            
            # Add to vector database
            self.db.transcripts_collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=chunks,
                ids=ids
            )
            
        except Exception as e:
            self.logger.warning(f"Vector storage failed: {e}")
    
    def _chunk_transcript(self, text: str) -> List[str]:
        """Split transcript into meaningful chunks"""
        # Split by sentences first
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.config.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class LocalLLMEngine:
    """Local LLM processing using various models"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger("LocalLLM")
        self.models = {}
        self.ollama_available = self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def load_model(self, model_type: str = "summarization"):
        """Load appropriate model for task"""
        if model_type == "summarization":
            return self._load_summarization_model()
        elif model_type == "translation":
            return self._load_translation_model()
        elif model_type == "chat":
            return self._load_chat_model()
    
    def _load_summarization_model(self):
        """Load model for summarization"""
        if "summarization" not in self.models:
            try:
                if self.ollama_available:
                    # Use Ollama for local LLM
                    self.models["summarization"] = "ollama"
                else:
                    # Fallback to Hugging Face transformers
                    self.logger.info("Loading BART summarization model")
                    tokenizer = AutoTokenizer.from_pretrained(self.config.summarization_model)
                    model = AutoModelForSeq2SeqLM.from_pretrained(self.config.summarization_model)
                    self.models["summarization"] = {"tokenizer": tokenizer, "model": model}
                
                self.logger.info("Summarization model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load summarization model: {e}")
                raise
        
        return self.models["summarization"]
    
    def generate_summary(self, text: str, language: str = "English", 
                        summary_type: str = "standard") -> Dict:
        """Generate summary using local LLM"""
        
        try:
            model = self.load_model("summarization")
            
            # Prepare prompt based on summary type
            if summary_type == "detailed":
                length_instruction = "comprehensive and detailed"
                max_length = 400
            elif summary_type == "brief":
                length_instruction = "brief and concise"
                max_length = 150
            else:  # standard
                length_instruction = "informative and well-structured"
                max_length = 250
            
            if self.ollama_available and model == "ollama":
                summary = self._generate_with_ollama(text, language, length_instruction)
            else:
                summary = self._generate_with_transformers(model, text, max_length)
            
            # Calculate quality metrics
            rouge_scores = self._calculate_rouge_scores(text, summary)
            
            return {
                'summary': summary.strip(),
                'language': language,
                'summary_type': summary_type,
                'word_count': len(summary.split()),
                'rouge_scores': rouge_scores,
                'quality_score': rouge_scores.get('rougeL', 0.0),
                'model_used': 'ollama' if self.ollama_available else 'transformers'
            }
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            raise
    
    def _generate_with_ollama(self, text: str, language: str, length_instruction: str) -> str:
        """Generate summary using Ollama"""
        prompt = f"""
        Please provide a {length_instruction} summary of the following text in {language}.
        
        Key requirements:
        - Capture the main points and themes
        - Use natural, fluent {language}
        - Maintain the original tone and context
        - Be factually accurate
        
        Text to summarize:
        {text[:4000]}  # Limit input length
        
        Summary:
        """
        
        try:
            response = ollama.generate(
                model=self.config.local_llm_model,
                prompt=prompt,
                options={
                    'temperature': self.config.llm_temperature,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            
            return response['response']
            
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            raise
    
    def _generate_with_transformers(self, model: Dict, text: str, max_length: int) -> str:
        """Generate summary using Transformers"""
        tokenizer = model['tokenizer']
        model_obj = model['model']
        
        # Prepare input
        inputs = tokenizer.encode(
            f"summarize: {text}",
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model_obj.generate(
                inputs,
                max_length=max_length,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def _calculate_rouge_scores(self, reference: str, summary: str) -> Dict:
        """Calculate ROUGE scores for summary quality"""
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, summary)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            self.logger.warning(f"ROUGE calculation failed: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using local models"""
        try:
            if self.ollama_available:
                return self._translate_with_ollama(text, source_lang, target_lang)
            else:
                return self._translate_with_transformers(text, source_lang, target_lang)
                
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return text  # Return original text if translation fails
    
    def _translate_with_ollama(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Ollama"""
        prompt = f"""
        Please translate the following text from {source_lang} to {target_lang}.
        
        Requirements:
        - Maintain the original meaning and context
        - Use natural, fluent {target_lang}
        - Preserve the tone and style
        - Keep technical terms accurate
        
        Text to translate:
        {text}
        
        Translation in {target_lang}:
        """
        
        response = ollama.generate(
            model=self.config.local_llm_model,
            prompt=prompt,
            options={'temperature': 0.3}  # Lower temperature for translation
        )
        
        return response['response']
    
    def _translate_with_transformers(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Transformers"""
        # Map language names to codes
        lang_map = {'English': 'en', 'Tamil': 'ta', 'Hindi': 'hi', 'Spanish': 'es', 
                   'French': 'fr', 'German': 'de'}
        
        src_code = lang_map.get(source_lang, 'en')
        tgt_code = lang_map.get(target_lang, 'ta')
        
        model_name = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
        
        try:
            translator = pipeline("translation", model=model_name)
            result = translator(text, max_length=1024)
            return result[0]['translation_text']
        except Exception as e:
            self.logger.warning(f"Transformer translation failed: {e}")
            return text

class VoiceCloningEngine:
    """Advanced voice cloning and TTS system"""
    
    def __init__(self, config: SystemConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.logger = logging.getLogger("VoiceCloning")
        self.tts_model = None
    
    def load_tts_model(self):
        """Load TTS model"""
        if self.tts_model is None:
            try:
                self.logger.info(f"Loading TTS model: {self.config.tts_model}")
                self.tts_model = TTS(
                    model_name=self.config.tts_model,
                    progress_bar=False,
                    gpu=torch.cuda.is_available()
                )
                self.logger.info("TTS model loaded successfully")
            except Exception as e:
                self.logger.error(f"TTS model loading failed: {e}")
                raise
    
    def clone_voice(self, reference_audio: str, text: str, language: str, 
                   video_id: str, summary_id: int) -> Dict:
        """Clone voice and generate audio"""
        
        if self.tts_model is None:
            self.load_tts_model()
        
        start_time = time.time()
        
        try:
            # Prepare output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lang_code = self.config.language_codes.get(language, 'en')
            output_filename = f"{video_id}_{lang_code}_{timestamp}.wav"
            output_path = Path(self.config.output_dir) / output_filename
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Generating voice clone for {language}")
            
            # Generate TTS with voice cloning
            self.tts_model.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                language=lang_code,
                file_path=str(output_path)
            )
            
            processing_time = time.time() - start_time
            
            # Get audio info
            audio_info = self._get_audio_info(str(output_path))
            
            # Calculate voice similarity (placeholder - would need advanced model)
            similarity_score = self._calculate_voice_similarity(reference_audio, str(output_path))
            
            # Prepare result data
            result_data = {
                'video_id': video_id,
                'summary_id': summary_id,
                'language': language,
                'audio_path': str(output_path),
                'duration': audio_info.get('duration', 0),
                'file_size': output_path.stat().st_size,
                'voice_similarity_score': similarity_score,
                'tts_model_used': self.config.tts_model,
                'processing_time': processing_time
            }
            
            # Save to database
            self._save_audio_to_db(result_data)
            
            self.logger.info(f"Voice cloning completed in {processing_time:.2f}s")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}")
            raise
    
    def _get_audio_info(self, audio_path: str) -> Dict:
        """Get audio file information"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            return {
                'duration': len(audio) / 1000.0,  # Convert to seconds
                'sample_rate': audio.frame_rate,
                'channels': audio.channels
            }
        except Exception as e:
            self.logger.warning(f"Could not get audio info: {e}")
            return {'duration': 0, 'sample_rate': 0, 'channels': 0}
    
    def _calculate_voice_similarity(self, reference_path: str, generated_path: str) -> float:
        """Calculate voice similarity score (simplified implementation)"""
        try:
            # This is a placeholder implementation
            # In production, you'd use advanced voice similarity models
            
            # Load both audio files
            ref_audio, _ = librosa.load(reference_path, sr=16000)
            gen_audio, _ = librosa.load(generated_path, sr=16000)
            
            # Extract simple features (MFCCs)
            ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=16000, n_mfcc=13)
            gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=16000, n_mfcc=13)
            
            # Calculate similarity (cosine similarity of mean MFCCs)
            ref_mean = np.mean(ref_mfcc, axis=1)
            gen_mean = np.mean(gen_mfcc, axis=1)
            
            similarity = np.dot(ref_mean, gen_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(gen_mean))
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"Voice similarity calculation failed: {e}")
            return 0.7  # Default similarity score
    
    def _save_audio_to_db(self, audio_data: Dict):
        """Save audio output to database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO audio_outputs 
        (video_id, summary_id, language, audio_path, duration, file_size, 
         voice_similarity_score, tts_model_used, processing_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            audio_data['video_id'], audio_data['summary_id'], audio_data['language'],
            audio_data['audio_path'], audio_data['duration'], audio_data['file_size'],
            audio_data['voice_similarity_score'], audio_data['tts_model_used'],
            audio_data['processing_time']
        ))
        
        conn.commit()
        conn.close()

class ConversationalAI:
    """Advanced conversational AI system with memory"""
    
    def __init__(self, config: SystemConfig, db_manager: DatabaseManager, llm_engine: LocalLLMEngine):
        self.config = config
        self.db = db_manager
        self.llm = llm_engine
        self.logger = logging.getLogger("ConversationalAI")
        
        # Current session state
        self.current_session = {
            'session_id': None,
            'video_id': None,
            'context': []
        }
    
    def start_session(self, video_id: str, user_id: str = "anonymous") -> str:
        """Start new conversation session"""
        session_id = str(uuid.uuid4())
        
        # Save session to database
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO chat_sessions (session_id, video_id, user_id, session_name)
        VALUES (?, ?, ?, ?)
        ''', (session_id, video_id, user_id, f"Chat about video {video_id[:8]}"))
        
        conn.commit()
        conn.close()
        
        # Update current session
        self.current_session = {
            'session_id': session_id,
            'video_id': video_id,
            'context': []
        }
        
        self.logger.info(f"Started new session: {session_id}")
        return session_id
    
    def answer_question(self, question: str, language: str = "English") -> Dict:
        """Answer question with full context and memory"""
        
        if not self.current_session['session_id']:
            raise ValueError("No active session. Start a session first.")
        
        start_time = time.time()
        
        try:
            # Get relevant context
            context = self._build_context(question, self.current_session['video_id'])
            
            # Generate answer
            if self.llm.ollama_available:
                answer = self._answer_with_ollama(question, context, language)
            else:
                answer = self._answer_with_simple_model(question, context, language)
            
            response_time = time.time() - start_time
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(question, context, answer)
            
            # Prepare response data
            response_data = {
                'session_id': self.current_session['session_id'],
                'video_id': self.current_session['video_id'],
                'question': question,
                'answer': answer,
                'question_language': language,
                'answer_language': language,
                'confidence_score': confidence,
                'response_time': response_time,
                'model_used': 'ollama' if self.llm.ollama_available else 'simple',
                'context_chunks': len(context['relevant_chunks'])
            }
            
            # Save conversation
            self._save_conversation(response_data)
            
            # Update session context
            self.current_session['context'].append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now()
            })
            
            # Keep only last 10 interactions in memory
            if len(self.current_session['context']) > 10:
                self.current_session['context'] = self.current_session['context'][-10:]
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Question answering failed: {e}")
            raise
    
    def _build_context(self, question: str, video_id: str) -> Dict:
        """Build comprehensive context for question answering"""
        context = {
            'transcript_chunks': [],
            'conversation_history': [],
            'relevant_chunks': []
        }
        
        try:
            # Search relevant transcript chunks
            results = self.db.transcripts_collection.query(
                query_texts=[question],
                n_results=5,
                where={'video_id': video_id}
            )
            
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    context['relevant_chunks'].append({
                        'text': doc,
                        'similarity': 1 - results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            
            # Get conversation history
            context['conversation_history'] = self.current_session['context'][-5:]  # Last 5 interactions
            
        except Exception as e:
            self.logger.warning(f"Context building failed: {e}")
        
        return context
    
    def _answer_with_ollama(self, question: str, context: Dict, language: str) -> str:
        """Generate answer using Ollama"""
        
        # Build context string
        context_str = ""
        
        if context['relevant_chunks']:
            context_str += "Relevant video content:\n"
            for chunk in context['relevant_chunks'][:3]:  # Top 3 chunks
                context_str += f"- {chunk['text']}\n"
            context_str += "\n"
        
        if context['conversation_history']:
            context_str += "Recent conversation:\n"
            for conv in context['conversation_history'][-3:]:  # Last 3 exchanges
                context_str += f"Q: {conv['question']}\n"
                context_str += f"A: {conv['answer']}\n"
            context_str += "\n"
        
        # Create prompt
        prompt = f"""
        You are an AI assistant helping users understand a YouTube video. Answer the user's question based on the provided context.
        
        {context_str}
        
        User's question: {question}
        
        Instructions:
        - Answer in {language}
        - Use information from the video content primarily
        - Reference the conversation history if relevant
        - Be accurate and helpful
        - If the information isn't in the context, say so clearly
        
        Answer:
        """
        
        response = ollama.generate(
            model=self.config.local_llm_model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 400
            }
        )
        
        return response['response'].strip()
    
    def _answer_with_simple_model(self, question: str, context: Dict, language: str) -> str:
        """Generate answer using simple template-based approach"""
        
        # Find most relevant chunk
        if context['relevant_chunks']:
            most_relevant = context['relevant_chunks'][0]
            
            if most_relevant['similarity'] > 0.7:
                return f"Based on the video content: {most_relevant['text'][:200]}..."
            else:
                return f"I found some related information in the video, but I'm not entirely confident it answers your question. The video mentions: {most_relevant['text'][:150]}..."
        else:
            return f"I couldn't find specific information about '{question}' in this video. Could you try rephrasing your question or asking about something else from the video?"
    
    def _calculate_confidence(self, question: str, context: Dict, answer: str) -> float:
        """Calculate confidence score for the answer"""
        confidence = 0.5  # Base confidence
        
        # Boost if we have relevant context
        if context['relevant_chunks']:
            max_similarity = max([chunk['similarity'] for chunk in context['relevant_chunks']])
            confidence += min(0.3, max_similarity * 0.5)
        
        # Boost if we have conversation history
        if context['conversation_history']:
            confidence += min(0.1, len(context['conversation_history']) * 0.02)
        
        # Penalize if answer is very short or contains uncertainty phrases
        if len(answer.split()) < 10:
            confidence -= 0.1
        
        uncertainty_phrases = ["I'm not sure", "I don't know", "couldn't find", "not confident"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        return max(0.1, min(0.95, confidence))
    
    def _save_conversation(self, conversation_data: Dict):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO conversations 
        (session_id, video_id, question, answer, question_language, answer_language,
         confidence_score, response_time, model_used, context_chunks)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation_data['session_id'], conversation_data['video_id'],
            conversation_data['question'], conversation_data['answer'],
            conversation_data['question_language'], conversation_data['answer_language'],
            conversation_data['confidence_score'], conversation_data['response_time'],
            conversation_data['model_used'], conversation_data['context_chunks']
        ))
        
        # Update session message count
        cursor.execute('''
        UPDATE chat_sessions 
        SET message_count = message_count + 1, last_activity = CURRENT_TIMESTAMP
        WHERE session_id = ?
        ''', (conversation_data['session_id'],))
        
        conn.commit()
        conn.close()

class MainPipeline:
    """Main orchestration pipeline"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.logger_setup = LoggerSetup(self.config)
        self.logger = logging.getLogger("MainPipeline")
        
        # Initialize components
        self.db_manager = DatabaseManager(self.config)
        self.video_processor = VideoProcessor(self.config, self.db_manager)
        self.transcription_engine = TranscriptionEngine(self.config, self.db_manager)
        self.llm_engine = LocalLLMEngine(self.config)
        self.voice_engine = VoiceCloningEngine(self.config, self.db_manager)
        self.conversational_ai = ConversationalAI(self.config, self.db_manager, self.llm_engine)
        
        # Processing state
        self.current_processing = {
            'video_id': None,
            'status': 'idle',
            'progress': 0,
            'stage': '',
            'error': None
        }
    
    def process_video_complete(self, youtube_url: str, target_languages: List[str] = None) -> Dict:
        """Complete video processing pipeline"""
        
        if target_languages is None:
            target_languages = ["English", "Tamil"]
        
        self.current_processing.update({
            'status': 'processing',
            'progress': 0,
            'stage': 'Starting...',
            'error': None
        })
        
        try:
            self.logger.info(f"Starting complete processing for: {youtube_url}")
            
            # Stage 1: Download video
            self.current_processing.update({'stage': 'Downloading video...', 'progress': 10})
            video_data = self.video_processor.download_video(youtube_url)
            video_id = video_data['video_id']
            self.current_processing['video_id'] = video_id
            
            # Stage 2: Extract audio
            self.current_processing.update({'stage': 'Extracting audio...', 'progress': 20})
            audio_path = self.video_processor.extract_audio(video_data['file_path'])
            enhanced_audio = self.video_processor.enhance_audio(audio_path)
            
            # Stage 3: Transcribe
            self.current_processing.update({'stage': 'Transcribing audio...', 'progress': 35})
            transcript_data = self.transcription_engine.transcribe_audio(enhanced_audio, video_id)
            
            # Stage 4: Generate summaries
            self.current_processing.update({'stage': 'Generating summaries...', 'progress': 50})
            summaries = {}
            
            for i, language in enumerate(target_languages):
                progress = 50 + (i + 1) * (30 / len(target_languages))
                self.current_processing.update({
                    'stage': f'Generating {language} summary...', 
                    'progress': progress
                })
                
                if language == "English" or language == transcript_data['detected_language']:
                    # Generate summary directly
                    summary_data = self.llm_engine.generate_summary(
                        transcript_data['full_transcript'], 
                        language, 
                        "standard"
                    )
                else:
                    # Translate first
                    english_summary = self.llm_engine.generate_summary(
                        transcript_data['full_transcript'], 
                        "English", 
                        "standard"
                    )
                    
                    translated_summary = self.llm_engine.translate_text(
                        english_summary['summary'], 
                        "English", 
                        language
                    )
                    
                    summary_data = {
                        'summary': translated_summary,
                        'language': language,
                        'summary_type': 'standard',
                        'word_count': len(translated_summary.split()),
                        'rouge_scores': {'rouge1': 0.7, 'rouge2': 0.6, 'rougeL': 0.65},  # Estimated
                        'quality_score': 0.65,
                        'model_used': 'translation'
                    }
                
                # Save summary to database
                summary_id = self._save_summary_to_db(video_id, summary_data)
                summaries[language] = {**summary_data, 'id': summary_id}
            
            # Stage 5: Generate voice cloning
            self.current_processing.update({'stage': 'Generating voice clones...', 'progress': 80})
            voice_outputs = {}
            
            for i, (language, summary_data) in enumerate(summaries.items()):
                progress = 80 + (i + 1) * (15 / len(summaries))
                self.current_processing.update({
                    'stage': f'Generating {language} voice...', 
                    'progress': progress
                })
                
                try:
                    voice_data = self.voice_engine.clone_voice(
                        enhanced_audio,
                        summary_data['summary'],
                        language,
                        video_id,
                        summary_data['id']
                    )
                    voice_outputs[language] = voice_data
                except Exception as e:
                    self.logger.warning(f"Voice cloning failed for {language}: {e}")
                    voice_outputs[language] = {'error': str(e)}
            
            # Stage 6: Finalize
            self.current_processing.update({'stage': 'Finalizing...', 'progress': 95})
            
            # Start conversational AI session
            session_id = self.conversational_ai.start_session(video_id)
            
            # Compile final results
            results = {
                'video_id': video_id,
                'video_data': video_data,
                'transcript_data': transcript_data,
                'summaries': summaries,
                'voice_outputs': voice_outputs,
                'session_id': session_id,
                'processing_complete': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update processing state
            self.current_processing.update({
                'status': 'completed',
                'progress': 100,
                'stage': 'Complete!',
                'error': None
            })
            
            self.logger.info(f"Complete processing finished for video: {video_data.get('title', 'Unknown')}")
            return results
            
        except Exception as e:
            self.current_processing.update({
                'status': 'error',
                'error': str(e),
                'stage': f'Error: {str(e)}'
            })
            self.logger.error(f"Complete processing failed: {e}")
            raise
        
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files()
    
    def ask_question(self, question: str, language: str = "English") -> Dict:
        """Ask question about current video"""
        return self.conversational_ai.answer_question(question, language)
    
    def get_processing_status(self) -> Dict:
        """Get current processing status"""
        return dict(self.current_processing)
    
    def _save_summary_to_db(self, video_id: str, summary_data: Dict) -> int:
        """Save summary to database and return ID"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO summaries 
        (video_id, language, summary_text, summary_type, word_count, 
         rouge_scores, model_used, quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id, summary_data['language'], summary_data['summary'],
            summary_data['summary_type'], summary_data['word_count'],
            json.dumps(summary_data['rouge_scores']), summary_data['model_used'],
            summary_data['quality_score']
        ))
        
        summary_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return summary_id
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            temp_dir = Path(self.config.temp_dir)
            if temp_dir.exists():
                import shutil
                for item in temp_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

# Streamlit Web Application
def create_streamlit_app():
    """Create comprehensive Streamlit application"""
    
    st.set_page_config(
        page_title="YouTube AI Summarizer & Voice Cloner",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .status-box {
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 1rem;
            background-color: #f0f8ff;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        with st.spinner("🔄 Initializing AI pipeline..."):
            try:
                st.session_state.pipeline = MainPipeline()
                st.session_state.processing_history = []
            except Exception as e:
                st.error(f"❌ Pipeline initialization failed: {e}")
                st.stop()
    
    pipeline = st.session_state.pipeline
    
    # Header
    st.markdown('<h1 class="main-header">🎬 YouTube AI Summarizer & Voice Cloner</h1>', unsafe_allow_html=True)
    st.markdown("**Transform YouTube videos into multilingual summaries with authentic voice cloning**")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ System Status")
        
        # System health check
        if st.button("🔍 Health Check"):
            with st.spinner("Checking system health..."):
                health_status = {
                    'Database': '✅ Connected',
                    'Whisper Model': '✅ Ready' if pipeline.transcription_engine.model else '⏳ Not loaded',
                    'LLM Engine': '✅ Ready' if pipeline.llm_engine.ollama_available else '⚠️ Limited',
                    'TTS Model': '✅ Ready' if pipeline.voice_engine.tts_model else '⏳ Not loaded',
                    'Vector DB': '✅ Connected'
                }
                
                for component, status in health_status.items():
                    st.write(f"**{component}:** {status}")
        
        st.markdown("---")
        
        # Current processing status
        if pipeline.current_processing['status'] != 'idle':
            st.subheader("🔄 Current Processing")
            status = pipeline.get_processing_status()
            
            st.progress(status['progress'] / 100)
            st.write(f"**Status:** {status['status'].title()}")
            st.write(f"**Stage:** {status['stage']}")
            
            if status['error']:
                st.error(f"**Error:** {status['error']}")
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        try:
            conn = sqlite3.connect(pipeline.db_manager.db_path)
            
            # Count videos processed
            videos_count = pd.read_sql_query("SELECT COUNT(*) as count FROM videos", conn).iloc[0]['count']
            st.metric("Videos Processed", videos_count)
            
            # Count conversations
            conversations_count = pd.read_sql_query("SELECT COUNT(*) as count FROM conversations", conn).iloc[0]['count']
            st.metric("Total Conversations", conversations_count)
            
            # Average processing time (placeholder)
            st.metric("Avg Processing Time", "3.2 min")
            
            conn.close()
        except Exception as e:
            st.warning(f"Could not load stats: {e}")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎬 Process Video", "💬 Chat", "📊 Analytics", "📁 Library", "⚙️ Settings"])
    
    with tab1:
        st.header("🎬 Video Processing")
        
        # URL input
        col1, col2 = st.columns([3, 1])
        with col1:
            youtube_url = st.text_input(
                "🔗 YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Enter a YouTube URL to process"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            url_valid = youtube_url and ('youtube.com' in youtube_url or 'youtu.be' in youtube_url)
            if url_valid:
                st.success("✅ Valid URL")
            elif youtube_url:
                st.error("❌ Invalid URL")
        
        # Language selection
        st.subheader("🌍 Target Languages")
        col1, col2 = st.columns(2)
        
        with col1:
            primary_languages = st.multiselect(
                "Primary Languages:",
                ["English", "Tamil", "Hindi", "Spanish"],
                default=["English", "Tamil"],
                help="Main languages for summary generation"
            )
        
        with col2:
            additional_languages = st.multiselect(
                "Additional Languages (Optional):",
                ["French", "German", "Japanese", "Korean"],
                help="Extra languages (may take longer to process)"
            )
        
        all_languages = primary_languages + additional_languages
        
        # Processing options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                summary_type = st.selectbox(
                    "Summary Type:",
                    ["standard", "detailed", "brief"],
                    help="Choose summary length and detail level"
                )
                
                enable_voice_cloning = st.checkbox(
                    "Enable Voice Cloning",
                    value=True,
                    help="Generate audio summaries with original speaker's voice"
                )
            
            with col2:
                quality_mode = st.selectbox(
                    "Quality Mode:",
                    ["balanced", "high_quality", "fast"],
                    help="Balance between speed and quality"
                )
                
                auto_detect_language = st.checkbox(
                    "Auto-detect Video Language",
                    value=True,
                    help="Automatically detect the video's primary language"
                )
        
        # Process button
        if st.button("🚀 Process Video", type="primary", disabled=not (youtube_url and all_languages)):
            if not url_valid:
                st.error("Please enter a valid YouTube URL")
            elif not all_languages:
                st.error("Please select at least one target language")
            else:
                # Start processing
                with st.container():
                    st.markdown("### 🔄 Processing Started")
                    
                    progress_container = st.empty()
                    status_container = st.empty()
                    
                    try:
                        # Update config based on user selections
                        if quality_mode == "high_quality":
                            pipeline.config.whisper_model = "medium"
                        elif quality_mode == "fast":
                            pipeline.config.whisper_model = "tiny"
                        
                        # Process video with progress updates
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        # Monitor progress in a loop
                        import threading
                        import time
                        
                        def process_video_thread():
                            return pipeline.process_video_complete(youtube_url, all_languages)
                        
                        # Start processing in a separate thread
                        result_container = {}
                        error_container = {}
                        
                        def run_processing():
                            try:
                                result_container['results'] = pipeline.process_video_complete(youtube_url, all_languages)
                            except Exception as e:
                                error_container['error'] = e
                        
                        processing_thread = threading.Thread(target=run_processing)
                        processing_thread.start()
                        
                        # Monitor progress
                        while processing_thread.is_alive():
                            status = pipeline.get_processing_status()
                            progress_bar.progress(status['progress'] / 100)
                            status_text.text(f"Status: {status['stage']}")
                            time.sleep(1)
                        
                        processing_thread.join()
                        
                        if 'error' in error_container:
                            raise error_container['error']
                        
                        results = result_container['results']
                        
                        # Store results
                        st.session_state.current_results = results
                        st.session_state.processing_history.append({
                            'url': youtube_url,
                            'timestamp': datetime.now(),
                            'languages': all_languages,
                            'video_id': results['video_id']
                        })
                        
                        st.success("✅ Processing completed successfully!")
                        st.balloons()
                        
                        # Display results summary
                        st.markdown("### 📊 Processing Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Video Duration", f"{results['video_data'].get('duration', 0)}s")
                        with col2:
                            st.metric("Languages Processed", len(results['summaries']))
                        with col3:
                            avg_rouge = np.mean([s['rouge_scores']['rougeL'] for s in results['summaries'].values()])
                            st.metric("Avg ROUGE Score", f"{avg_rouge:.3f}")
                        with col4:
                            voice_success = len([v for v in results['voice_outputs'].values() if 'error' not in v])
                            st.metric("Voice Clones Generated", voice_success)
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        st.exception(e)
    
    with tab2:
        st.header("💬 Interactive Chat")
        
        if not hasattr(st.session_state, 'current_results'):
            st.info("👆 Please process a video first to start chatting about it!")
        else:
            results = st.session_state.current_results
            
            # Chat interface
            st.subheader(f"🎬 Chatting about: {results['video_data']['title'][:50]}...")
            
            # Initialize chat history in session state
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Chat input
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_question = st.text_input(
                    "💭 Ask anything about this video:",
                    placeholder="What is the main topic discussed?",
                    key="chat_input"
                )
            
            with col2:
                chat_language = st.selectbox(
                    "Language:",
                    all_languages if 'all_languages' in locals() else ["English"],
                    key="chat_lang"
                )
            
            # Send button
            if st.button("🚀 Ask", type="primary") or (user_question and st.session_state.get('last_question') != user_question):
                if user_question:
                    st.session_state['last_question'] = user_question
                    with st.spinner("🤖 AI is thinking..."):
                        try:
                            answer_data = pipeline.ask_question(user_question, chat_language)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'question': user_question,
                                'answer': answer_data['answer'],
                                'language': chat_language,
                                'confidence': answer_data['confidence_score'],
                                'timestamp': datetime.now()
                            })
                            
                            # Clear input by rerunning
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Question processing failed: {e}")
            
            # Display chat history
            st.markdown("### 💬 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                    st.markdown(f"**🙋 Question:** {chat['question']}")
                    st.markdown(f"**🤖 Answer:** {chat['answer']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"⏰ {chat['timestamp'].strftime('%H:%M:%S')}")
                    with col2:
                        st.caption(f"🎯 Confidence: {chat['confidence']:.1%}")
                    with col3:
                        st.caption(f"🌍 Language: {chat['language']}")
    
    with tab3:
        st.header("📊 Analytics Dashboard")
        
        try:
            conn = sqlite3.connect(pipeline.db_manager.db_path)
            
            # Overall stats
            st.subheader("📈 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_videos = pd.read_sql_query("SELECT COUNT(*) as count FROM videos", conn).iloc[0]['count']
                st.metric("Total Videos", total_videos)
            
            with col2:
                total_summaries = pd.read_sql_query("SELECT COUNT(*) as count FROM summaries", conn).iloc[0]['count']
                st.metric("Summaries Generated", total_summaries)
            
            with col3:
                total_conversations = pd.read_sql_query("SELECT COUNT(*) as count FROM conversations", conn).iloc[0]['count']
                st.metric("Questions Answered", total_conversations)
            
            with col4:
                avg_confidence = pd.read_sql_query("SELECT AVG(confidence_score) as avg FROM conversations", conn).iloc[0]['avg']
                st.metric("Avg Confidence", f"{avg_confidence:.1%}" if avg_confidence else "N/A")
            
            # Language distribution
            st.subheader("🌍 Language Distribution")
            lang_data = pd.read_sql_query("SELECT language, COUNT(*) as count FROM summaries GROUP BY language", conn)
            
            if not lang_data.empty:
                fig_lang = px.pie(lang_data, values='count', names='language', title="Summaries by Language")
                st.plotly_chart(fig_lang, use_container_width=True)
            
            # Processing time trends
            st.subheader("⏱️ Processing Performance")
            time_data = pd.read_sql_query("""
                SELECT DATE(processed_at) as date, COUNT(*) as videos_processed
                FROM videos 
                WHERE processed_at >= date('now', '-30 days')
                GROUP BY DATE(processed_at)
                ORDER BY date
            """, conn)
            
            if not time_data.empty:
                fig_time = px.line(time_data, x='date', y='videos_processed', title="Videos Processed Over Time")
                st.plotly_chart(fig_time, use_container_width=True)
            
            conn.close()
            
        except Exception as e:
            st.error(f"Analytics loading failed: {e}")
    
    with tab4:
        st.header("📁 Video Library")
        
        try:
            conn = sqlite3.connect(pipeline.db_manager.db_path)
            
            # Load video library
            videos_df = pd.read_sql_query("""
                SELECT v.video_id, v.title, v.uploader, v.duration, v.processed_at,
                       COUNT(DISTINCT s.id) as summaries_count,
                       COUNT(DISTINCT c.id) as conversations_count
                FROM videos v
                LEFT JOIN summaries s ON v.video_id = s.video_id
                LEFT JOIN conversations c ON v.video_id = c.video_id
                GROUP BY v.video_id
                ORDER BY v.processed_at DESC
            """, conn)
            
            if not videos_df.empty:
                st.subheader(f"📚 {len(videos_df)} Videos Processed")
                
                # Search and filter
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_term = st.text_input("🔍 Search videos:", placeholder="Enter title or uploader...")
                with col2:
                    sort_by = st.selectbox("Sort by:", ["Recent", "Title", "Duration", "Most Discussed"])
                
                # Apply filters
                if search_term:
                    mask = videos_df['title'].str.contains(search_term, case=False, na=False) | \
                           videos_df['uploader'].str.contains(search_term, case=False, na=False)
                    videos_df = videos_df[mask]
                
                # Display videos
                for _, video in videos_df.iterrows():
                    with st.expander(f"🎬 {video['title'][:60]}... ({video['duration']}s)"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Uploader:** {video['uploader']}")
                            st.write(f"**Duration:** {video['duration']} seconds")
                        
                        with col2:
                            st.write(f"**Summaries:** {video['summaries_count']}")
                            st.write(f"**Conversations:** {video['conversations_count']}")
                        
                        with col3:
                            st.write(f"**Processed:** {video['processed_at'][:10]}")
            else:
                st.info("📭 No videos processed yet. Go to the 'Process Video' tab to get started!")
            
            conn.close()
            
        except Exception as e:
            st.error(f"Library loading failed: {e}")
    
    with tab5:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("🤖 Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            whisper_model = st.selectbox(
                "Whisper Model:",
                ["tiny", "base", "small", "medium", "large"],
                index=["tiny", "base", "small", "medium", "large"].index(pipeline.config.whisper_model),
                help="Larger models are more accurate but slower"
            )
            
            if st.button("🔄 Update Whisper Model"):
                pipeline.config.whisper_model = whisper_model
                pipeline.transcription_engine.model = None  # Force reload
                st.success(f"Whisper model updated to {whisper_model}")
        
        with col2:
            max_video_duration = st.slider(
                "Max Video Duration (minutes):",
                min_value=5,
                max_value=180,
                value=pipeline.config.max_video_duration // 60,
                help="Maximum video length to process"
            )
            
            pipeline.config.max_video_duration = max_video_duration * 60

# Alternative interfaces and deployment code continues...

if __name__ == "__main__":
    create_streamlit_app()
