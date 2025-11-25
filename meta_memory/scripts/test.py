from pymilvus import connections
import time

print("Trying to connect to Milvus at 127.0.0.1:19530...")
try:
    connections.connect(host="127.0.0.1", port="19530", timeout=10)
    print("✅ Connected successfully!")
except Exception as e:
    print("❌ Failed to connect:", e)