"""
Launch Interactive Web Interface
=================================
Launches a Gradio-based web interface for interactive sentiment analysis
with all explainability methods.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.interactive_interface import create_interface


def main():
    """Launch the interactive interface."""
    
    print("\n" + "="*70)
    print("🚀 Hindi Sentiment Analysis - Interactive Interface")
    print("="*70)
    print()
    print("📦 Loading models and initializing analyzers...")
    print("   This may take a few moments...")
    print()
    
    # Create and launch interface
    demo = create_interface()
    
    print("="*70)
    print("✅ Interface Ready!")
    print("="*70)
    print()
    print("🌐 Access the interface at:")
    print("   • Local: http://localhost:7860")
    print("   • Network: http://0.0.0.0:7860")
    print()
    print("💡 Features:")
    print("   • Real-time sentiment analysis")
    print("   • 5 explainability methods")
    print("   • Interactive visualizations")
    print("   • Support for Hindi (Devanagari) text")
    print()
    print("⌨  Press Ctrl+C to stop the server")
    print("="*70)
    print()
    
    # Launch with settings
    demo.launch(
        share=False,  # Set to True for public URL
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interface stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure you've trained the model first (run_phase1.py)")

