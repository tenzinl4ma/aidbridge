import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST', 'redis-16546.c212.ap-south-1-1.ec2.redns.redis-cloud.com')
REDIS_PORT = int(os.getenv('REDIS_PORT', 16546))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
REDIS_DB = int(os.getenv('REDIS_DB', 0))

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')