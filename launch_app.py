import os
import time
import subprocess
import threading
from pyngrok import ngrok, conf
import streamlit as st

def launch_streamlit():
    """Launch the Streamlit app"""
    print("🚀 Starting Streamlit application...")
    
    # Set the path to the current directory
    app_path = os.path.join(os.getcwd(), "streamlit_app.py")
    
    # Launch Streamlit
    cmd = f"streamlit run {app_path} --server.port 8501 --server.address localhost"
    subprocess.run(cmd, shell=True)

def setup_ngrok():
    """Setup ngrok tunnel"""
    try:
        print("🌐 Setting up ngrok tunnel...")
        
        # Kill any existing ngrok processes
        ngrok.kill()
        
        # Create tunnel
        tunnel = ngrok.connect(8501, "http")
        public_url = tunnel.public_url
        
        print(f"\n{'='*60}")
        print(f"🎉 SUCCESS! Your app is now live and accessible worldwide!")
        print(f"{'='*60}")
        print(f"")
        print(f"🔗 PUBLIC URL: {public_url}")
        print(f"📱 LOCAL URL:  http://localhost:8501")
        print(f"")
        print(f"✨ Share this link with anyone to let them use your app!")
        print(f"🌍 The app is accessible from anywhere in the world")
        print(f"⚡ Real-time image tampering detection available online")
        print(f"")
        print(f"{'='*60}")
        print(f"💡 To stop the app, press Ctrl+C in the terminal")
        print(f"{'='*60}")
        
        return public_url
        
    except Exception as e:
        print(f"❌ Error setting up ngrok: {e}")
        print("📝 Make sure you have ngrok installed and configured")
        print("🔧 You can still access the app locally at http://localhost:8501")
        return None

def main():
    print(f"""
    
    🔍 AI IMAGE TAMPERING DETECTION - WEB LAUNCHER
    =============================================
    
    🎯 Quality-Based Authentication System
    🌐 With ngrok public URL sharing
    ⚡ Real-time analysis capabilities
    
    """)
    
    print("⏳ Initializing application...")
    
    # Setup ngrok in a separate thread first
    def setup_tunnel():
        time.sleep(3)  # Wait for Streamlit to start
        setup_ngrok()
    
    # Start ngrok setup in background
    tunnel_thread = threading.Thread(target=setup_tunnel)
    tunnel_thread.daemon = True
    tunnel_thread.start()
    
    # Launch Streamlit (this will block until app is closed)
    launch_streamlit()

if __name__ == "__main__":
    main()
