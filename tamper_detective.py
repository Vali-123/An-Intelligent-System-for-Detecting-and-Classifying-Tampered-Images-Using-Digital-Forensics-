#!/usr/bin/env python3
"""
🕵️ TAMPER DETECTIVE - Your Image Analysis Assistant
Fun interactive menu to access all tampering detection tools
"""

import os
import sys
import time
import random

class TamperDetective:
    def __init__(self):
        self.detective_quotes = [
            "🕵️ Elementary, my dear Watson! Let's catch some tampering!",
            "🔍 The game is afoot! Time to investigate these images!",
            "🎯 Every pixel tells a story... let's read it!",
            "🚀 Ready to unleash the power of AI detection!",
            "💡 Trust me, I'm a detective... a tamper detective!"
        ]
        
        self.tools = {
            '1': {'name': '🚀 Quick Scan', 'desc': 'Fast single image analysis with fun feedback', 'script': 'quick_scan.py'},
            '2': {'name': '📂 Folder Scan', 'desc': 'Batch analysis of entire folders', 'script': 'folder_scan.py'},
            '3': {'name': '🔄 Compare Images', 'desc': 'Side-by-side comparison of two images', 'script': 'compare_images.py'},
            '4': {'name': '🔬 Deep Analysis', 'desc': 'Detailed technical analysis', 'script': 'analyze_single_image.py'},
            '5': {'name': '🎮 Interactive Demo', 'desc': 'Original demo with menu options', 'script': 'demo.py'},
            '6': {'name': '🖥️ Streamlit App', 'desc': 'Web interface (if available)', 'script': 'streamlit_app.py'}
        }
    
    def show_banner(self):
        """Display fun detective banner"""
        print("="*70)
        print("🕵️  TAMPER DETECTIVE - Your AI Image Analysis Assistant  🕵️")
        print("="*70)
        print(random.choice(self.detective_quotes))
        print("-" * 70)
    
    def show_menu(self):
        """Display the main menu"""
        print("\n🎯 CHOOSE YOUR DETECTIVE TOOL:")
        print("─" * 50)
        
        for key, tool in self.tools.items():
            print(f"  {key}. {tool['name']}")
            print(f"     └─ {tool['desc']}")
            print()
        
        print("  0. 🚪 Exit Detective Mode")
        print("─" * 50)
    
    def run_tool(self, choice):
        """Run the selected tool"""
        if choice == '0':
            self.goodbye()
            return False
        
        if choice not in self.tools:
            print("❌ Invalid choice! Try again, detective.")
            return True
        
        tool = self.tools[choice]
        script = tool['script']
        
        print(f"\n🚀 Launching {tool['name']}...")
        print("─" * 50)
        
        # Check if script exists
        if not os.path.exists(script):
            print(f"❌ Tool not found: {script}")
            print("Make sure all detective tools are in the same folder!")
            input("\n📱 Press Enter to return to menu...")
            return True
        
        # Special handling for different tools
        if choice == '1':  # Quick Scan
            self.run_quick_scan()
        elif choice == '2':  # Folder Scan
            self.run_folder_scan()
        elif choice == '3':  # Compare Images
            self.run_compare_images()
        elif choice == '4':  # Deep Analysis
            self.run_deep_analysis()
        elif choice == '5':  # Interactive Demo
            os.system(f'python {script}')
        elif choice == '6':  # Streamlit App
            self.run_streamlit()
        
        input("\n📱 Press Enter to return to detective menu...")
        return True
    
    def run_quick_scan(self):
        """Launch quick scan with file picker"""
        print("🚀 QUICK SCAN MODE")
        print("Enter image path or drag & drop:")
        image_path = input("➤ ").strip().strip('"').strip("'")
        
        if image_path and os.path.exists(image_path):
            os.system(f'python quick_scan.py "{image_path}"')
        else:
            print("❌ Image not found! Detective mode aborted.")
    
    def run_folder_scan(self):
        """Launch folder scan with folder picker"""
        print("📂 FOLDER SCAN MODE")
        print("Enter folder path to investigate:")
        folder_path = input("➤ ").strip().strip('"').strip("'")
        
        if folder_path and os.path.exists(folder_path):
            os.system(f'python folder_scan.py "{folder_path}"')
        else:
            print("❌ Folder not found! Detective investigation cancelled.")
    
    def run_compare_images(self):
        """Launch image comparison"""
        print("🔄 IMAGE COMPARISON MODE")
        print("Enter path to first image:")
        image1 = input("Image 1 ➤ ").strip().strip('"').strip("'")
        
        if not image1 or not os.path.exists(image1):
            print("❌ First image not found!")
            return
        
        print("Enter path to second image:")
        image2 = input("Image 2 ➤ ").strip().strip('"').strip("'")
        
        if not image2 or not os.path.exists(image2):
            print("❌ Second image not found!")
            return
        
        os.system(f'python compare_images.py "{image1}" "{image2}"')
    
    def run_deep_analysis(self):
        """Launch deep analysis"""
        print("🔬 DEEP ANALYSIS MODE")
        print("Enter image path for detailed investigation:")
        image_path = input("➤ ").strip().strip('"').strip("'")
        
        if image_path and os.path.exists(image_path):
            os.system(f'python analyze_single_image.py "{image_path}"')
        else:
            print("❌ Image not found! Deep analysis cancelled.")
    
    def run_streamlit(self):
        """Try to launch Streamlit app"""
        print("🖥️ LAUNCHING WEB INTERFACE...")
        if os.path.exists('streamlit_app.py'):
            print("Starting web server... Check your browser!")
            os.system('streamlit run streamlit_app.py')
        else:
            print("❌ Web interface not available.")
            print("You can create one using Streamlit!")
    
    def show_tips(self):
        """Show helpful tips"""
        tips = [
            "💡 Tip: Drag and drop files instead of typing long paths!",
            "🎯 Tip: Higher confidence scores mean more likely tampering!",
            "🔍 Tip: Check multiple detection methods for best results!",
            "📊 Tip: Save reports when scanning folders for documentation!",
            "⚡ Tip: Quick Scan is perfect for single image checks!"
        ]
        
        print(f"\n{random.choice(tips)}")
    
    def goodbye(self):
        """Farewell message"""
        farewells = [
            "🕵️ Case closed! Great detective work!",
            "👋 Until next time, keep those images authentic!",
            "🎉 Another successful investigation! Stay vigilant!",
            "🔍 The truth is out there... and you found it!",
            "🌟 Keep fighting the good fight against fake images!"
        ]
        
        print("\n" + "="*70)
        print(random.choice(farewells))
        print("🎯 Tamper Detective signing off... 🕵️")
        print("="*70)
    
    def run(self):
        """Main detective loop"""
        self.show_banner()
        
        while True:
            self.show_menu()
            self.show_tips()
            
            print("\n🎯 What's your next move, detective?")
            choice = input("➤ ").strip()
            
            if not self.run_tool(choice):
                break

def main():
    # Check if we have the required files
    required_files = ['analyze_single_image.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nMake sure you're in the correct directory!")
        return
    
    # Start the detective interface
    detective = TamperDetective()
    detective.run()

if __name__ == "__main__":
    main()
