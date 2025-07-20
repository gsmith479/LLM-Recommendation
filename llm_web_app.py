import streamlit as st
import platform
import psutil
import subprocess
import threading
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="LLM Model Recommender",
    page_icon="🤖",
    layout="wide"
)

class LLMRecommenderWeb:
    def __init__(self):
        self.model_database = self.load_model_database()
        self.system_specs = {}

    def load_model_database(self):
        """Load database of GGUF models with their requirements"""
        return {
            "Llama 2 7B Chat GGUF": {
                "size": "7B",
                "min_ram": 8,
                "recommended_ram": 16,
                "min_vram": 4,
                "recommended_vram": 8,
                "model_file_size": 4.1,
                "quantization": "Q4_0",
                "use_case": "General chat, coding assistance",
                "performance": "Fast inference, good quality",
                "download_url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF"
            },
            "Code Llama 7B GGUF": {
                "size": "7B",
                "min_ram": 8,
                "recommended_ram": 16,
                "min_vram": 4,
                "recommended_vram": 8,
                "model_file_size": 4.1,
                "quantization": "Q4_0",
                "use_case": "Code generation, programming help",
                "performance": "Excellent for coding tasks",
                "download_url": "https://huggingface.co/TheBloke/CodeLlama-7B-GGUF"
            },
            "Mistral 7B Instruct GGUF": {
                "size": "7B",
                "min_ram": 6,
                "recommended_ram": 12,
                "min_vram": 3,
                "recommended_vram": 6,
                "model_file_size": 4.1,
                "quantization": "Q4_0",
                "use_case": "General purpose, instruction following",
                "performance": "Very fast, efficient",
                "download_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
            },
            "Llama 2 13B Chat GGUF": {
                "size": "13B",
                "min_ram": 16,
                "recommended_ram": 32,
                "min_vram": 8,
                "recommended_vram": 12,
                "model_file_size": 7.3,
                "quantization": "Q4_0",
                "use_case": "Advanced chat, complex reasoning",
                "performance": "Better quality than 7B models",
                "download_url": "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF"
            },
            "Code Llama 13B GGUF": {
                "size": "13B",
                "min_ram": 16,
                "recommended_ram": 32,
                "min_vram": 8,
                "recommended_vram": 12,
                "model_file_size": 7.3,
                "quantization": "Q4_0",
                "use_case": "Advanced coding, large codebases",
                "performance": "Superior code understanding",
                "download_url": "https://huggingface.co/TheBloke/CodeLlama-13B-GGUF"
            },
            "Llama 2 70B Chat GGUF": {
                "size": "70B",
                "min_ram": 64,
                "recommended_ram": 128,
                "min_vram": 24,
                "recommended_vram": 48,
                "model_file_size": 38.0,
                "quantization": "Q4_0",
                "use_case": "Professional applications, research",
                "performance": "Highest quality, slower inference",
                "download_url": "https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF"
            }
        }

    def detect_system_specs(self):
        """Detect system specifications - this would run on client side"""
        try:
            system_info = {
                'OS': f"{platform.system()} {platform.release()}",
                'CPU': platform.processor() or "Unknown",
                'CPU Cores': psutil.cpu_count(logical=False),
                'CPU Threads': psutil.cpu_count(logical=True),
                'Total RAM': round(psutil.virtual_memory().total / (1024**3), 2),
                'Available RAM': round(psutil.virtual_memory().available / (1024**3), 2),
            }
            
            # Basic GPU detection
            gpu_info = self.detect_gpu()
            system_info.update(gpu_info)
            
            return system_info
        except Exception as e:
            st.error(f"Error detecting system specs: {e}")
            return {}

    def detect_gpu(self):
        """Detect GPU information"""
        gpu_info = {'GPU': 'Unknown', 'VRAM': 0}
        
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "Name"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:
                        line = line.strip()
                        if line and 'Name' not in line:
                            gpu_info['GPU'] = line
                            break
        except:
            pass
        
        return gpu_info

    def get_recommendations(self, ram_gb, vram_gb):
        """Generate model recommendations"""
        recommended = []
        compatible = []
        
        for model_name, specs in self.model_database.items():
            if (ram_gb >= specs['recommended_ram'] and 
                vram_gb >= specs['recommended_vram']):
                recommended.append((model_name, specs))
            elif (ram_gb >= specs['min_ram'] and 
                  vram_gb >= specs['min_vram']):
                compatible.append((model_name, specs))
        
        return recommended, compatible

def main():
    st.title("🤖 LLM Model Recommender for LM Studio")
    st.markdown("Find the best GGUF models for your system specifications")
    
    app = LLMRecommenderWeb()
    
    # Sidebar for manual input
    st.sidebar.header("System Specifications")
    
    # Manual input option
    use_manual = st.sidebar.checkbox("Enter specifications manually")
    
    if use_manual:
        ram_gb = st.sidebar.number_input("Total RAM (GB)", min_value=1, max_value=256, value=16)
        vram_gb = st.sidebar.number_input("GPU VRAM (GB)", min_value=0, max_value=80, value=8)
        gpu_name = st.sidebar.text_input("GPU Name (optional)", value="Unknown")
        
        system_specs = {
            'Total RAM': ram_gb,
            'VRAM': vram_gb,
            'GPU': gpu_name
        }
    else:
        if st.sidebar.button("🔄 Detect System Specs"):
            with st.spinner("Detecting system specifications..."):
                system_specs = app.detect_system_specs()
                st.session_state.system_specs = system_specs
        
        system_specs = st.session_state.get('system_specs', {})
        ram_gb = system_specs.get('Total RAM', 0)
        vram_gb = system_specs.get('VRAM', 0)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 System Info", "🎯 Recommendations", "📚 Model Database"])
    
    with tab1:
        st.header("System Information")
        if system_specs:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total RAM", f"{system_specs.get('Total RAM', 0)} GB")
                st.metric("CPU Cores", system_specs.get('CPU Cores', 'Unknown'))
                st.metric("Operating System", system_specs.get('OS', 'Unknown'))
            
            with col2:
                st.metric("GPU VRAM", f"{system_specs.get('VRAM', 0)} GB")
                st.metric("GPU", system_specs.get('GPU', 'Unknown'))
                st.metric("CPU Threads", system_specs.get('CPU Threads', 'Unknown'))
        else:
            st.info("Click 'Detect System Specs' or use manual input to see system information.")
    
    with tab2:
        st.header("Model Recommendations")
        
        if ram_gb > 0:
            recommended, compatible = app.get_recommendations(ram_gb, vram_gb)
            
            if recommended:
                st.success("✅ Highly Recommended Models")
                for model_name, specs in recommended:
                    with st.expander(f"📦 {model_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Size:** {specs['size']} parameters")
                            st.write(f"**File Size:** {specs['model_file_size']} GB")
                            st.write(f"**Use Case:** {specs['use_case']}")
                        with col2:
                            st.write(f"**Performance:** {specs['performance']}")
                            st.write(f"**Quantization:** {specs['quantization']}")
                            st.link_button("Download", specs['download_url'])
            
            if compatible:
                st.warning("⚠️ Compatible Models (May run slower)")
                for model_name, specs in compatible:
                    with st.expander(f"📦 {model_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Size:** {specs['size']} parameters")
                            st.write(f"**File Size:** {specs['model_file_size']} GB")
                        with col2:
                            st.write(f"**Use Case:** {specs['use_case']}")
                            st.link_button("Download", specs['download_url'])
            
            if not recommended and not compatible:
                st.error("❌ No compatible models found for your system specifications.")
                st.info("Consider upgrading your RAM or GPU for better model compatibility.")
        
        else:
            st.info("Please provide system specifications to get recommendations.")
    
    with tab3:
        st.header("Complete Model Database")
        
        # Group models by size
        small_models = []
        medium_models = []
        large_models = []
        
        for model_name, specs in app.model_database.items():
            if "7B" in specs['size']:
                small_models.append((model_name, specs))
            elif "13B" in specs['size']:
                medium_models.append((model_name, specs))
            else:
                large_models.append((model_name, specs))
        
        # Display categories
        categories = [
            ("🚀 Small Models (7B Parameters)", small_models),
            ("⚡ Medium Models (13B Parameters)", medium_models),
            ("🏋️ Large Models (30B+ Parameters)", large_models)
        ]
        
        for category_name, models in categories:
            if models:
                st.subheader(category_name)
                for model_name, specs in models:
                    with st.expander(f"📦 {model_name}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Size:** {specs['size']}")
                            st.write(f"**File Size:** {specs['model_file_size']} GB")
                            st.write(f"**Quantization:** {specs['quantization']}")
                        with col2:
                            st.write(f"**Min RAM:** {specs['min_ram']} GB")
                            st.write(f"**Min VRAM:** {specs['min_vram']} GB")
                            st.write(f"**Recommended RAM:** {specs['recommended_ram']} GB")
                        with col3:
                            st.write(f"**Use Case:** {specs['use_case']}")
                            st.write(f"**Performance:** {specs['performance']}")
                            st.link_button("Download", specs['download_url'])

if __name__ == "__main__":
    main()