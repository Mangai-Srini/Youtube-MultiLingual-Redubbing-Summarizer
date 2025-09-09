#Importing the necesary libraries
import os
import sqlite3
import yt_dlp
import ffmpeg
import whisper
import streamlit as st
from datetime import datetime,timedelta
import tempfile
from pathlib import Path
import json
from typing import List,Dict,Optional,Tuple
import logging
import hashlib
import uuid
import asyncio
import time
import threading
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile

# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

#video/audio processing
import yt_dlp
import ffmpeg
from pydub import AudioSegment
from pydub.effects import normalize

#Ai/Ml libraries used
import whisper
import torch
from transformers import (
    AutoTokenizer,AutoModelforSeq2SeqLM,
    AutoModel,pipeline,MarianMTModel,MarianTokenizer
)
from sentence_transformers import SentenceTransformer
from TTS.api import TTS
import librosa
import soundfile as sf

#LLM integration(offline/local capable)
from transformers import AutoModelForCausalLM,GPT2LMHeadModel
from ollama #loacal llama integration
from langchain.llms import Llamacpp
from langchain.callbacks.managers import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#Database/Vector storage
import chromadb
from chromadb.config import Settings
import faiss

#Web framework Alternatives
import gradio as gr
import flask
from flask import Flask,render_template,request,jsonify,send_file

#Configuration and logging
import yaml
from rogue_score import rogue_scorer
import nltk

print("All libraries imported successfully")





















