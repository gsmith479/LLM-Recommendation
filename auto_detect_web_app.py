import streamlit as st
import platform
import psutil
import subprocess
import re
from datetime import datetime

# Page config
st.set_page_config(
    page_title="LLM Model Recommender",
    page_icon="🤖",
    layout="wide"
)

class SystemDetector:
    """Class to automatically detect system specifications"""
    
    @staticmethod
    def get_total_ram_gb():
        """Get total system RAM in GB"""
        try:
            ram_bytes = psutil.virtual_memory().total
            ram_gb = round(ram_bytes / (1024**3))
            return ram_gb
        except Exception as e:
            st.warning(f"Could not detect RAM: {e}")
            return 16  # Default fallback
    
    @staticmethod
    def get_gpu_info():
        """Get GPU information including VRAM"""
        gpu_info = {"name": "Unknown", "vram_gb": 0}
        
        try:
            # Try nvidia-smi first (most reliable for NVIDIA GPUs)
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    if len(parts) >= 2:
                        gpu_info["name"] = parts[0].strip()
                        gpu_info["vram_gb"] = round(int(parts[1]) / 1024)  # Convert MB to GB
                        return gpu_info
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback methods for different operating systems
        system = platform.system()
        
        if system == "Windows":
            gpu_info.update(SystemDetector._get_windows_gpu())
        elif system == "Darwin":  # macOS
            gpu_info.update(SystemDetector._get_macos_gpu())
        elif system == "Linux":
            gpu_info.update(SystemDetector._get_linux_gpu())
        
        return gpu_info
    
    @staticmethod
    def _get_windows_gpu():
        """Get GPU info on Windows using wmic"""
        try:
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterRAM'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip() and 'AdapterRAM' not in line:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            vram_bytes = int(parts[0]) if parts[0].isdigit() else 0
                            gpu_name = ' '.join(parts[1:])
                            if vram_bytes > 0:
                                return {"name": gpu_name, "vram_gb": round(vram_bytes / (1024**3))}
        except Exception:
            pass
        
        return {"name": "Unknown Windows GPU", "vram_gb": 4}
    
    @staticmethod
    def _get_macos_gpu():
        """Get GPU info on macOS using system_profiler"""
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse the output to find GPU name and VRAM
                lines = result.stdout.split('\n')
                gpu_name = "Unknown Mac GPU"
                vram_gb = 0
                
                for i, line in enumerate(lines):
                    if 'Chipset Model:' in line:
                        gpu_name = line.split(':')[1].strip()
                    elif 'VRAM (Total):' in line:
                        vram_match = re.search(r'(\d+)\s*MB', line)
                        if vram_match:
                            vram_gb = round(int(vram_match.group(1)) / 1024)
                    elif 'Metal Support:' in line and vram_gb == 0:
                        # For integrated GPUs, estimate based on system RAM
                        vram_gb = max(2, SystemDetector.get_total_ram_gb() // 8)
                
                return {"name": gpu_name, "vram_gb": vram_gb}
        except Exception:
            pass
        
        return {"name": "Unknown Mac GPU", "vram_gb": 8}
    
    @staticmethod
    def _get_linux_gpu():
        """Get GPU info on Linux using lspci"""
        try:
            # Try lspci first
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA' in line or 'Display' in line:
                        # Extract GPU name
                        if 'NVIDIA' in line.upper():
                            gpu_name = "NVIDIA " + line.split('NVIDIA')[1].split('[')[0].strip()
                            return {"name": gpu_name, "vram_gb": 8}  # Default estimate
                        elif 'AMD' in line.upper() or 'ATI' in line.upper():
                            gpu_name = line.split(':')[2].strip() if ':' in line else "AMD GPU"
                            return {"name": gpu_name, "vram_gb": 6}  # Default estimate
        except Exception:
            pass
        
        return {"name": "Unknown Linux GPU", "vram_gb": 4}
    
    @staticmethod
    def get_cpu_info():
        """Get CPU information"""
        try:
            cpu_name = platform.processor() or "Unknown CPU"
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            return {
                "name": cpu_name,
                "cores": cpu_cores,
                "threads": cpu_threads
            }
        except Exception:
            return {"name": "Unknown CPU", "cores": 4, "threads": 8}

class LLMRecommenderWeb:
    def __init__(self):
        self.model_database = self.load_model_database()
        self.detector = SystemDetector()

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
            },
            "Code Llama 34B GGUF": {
                "size": "34B",
                "min_ram": 32,
                "recommended_ram": 64,
                "min_vram": 16,
                "recommended_vram": 24,
                "model_file_size": 19.0,
                "quantization": "Q4_0",
                "use_case": "Enterprise coding, complex projects",
                "performance": "Excellent code quality",
                "download_url": "https://huggingface.co/TheBloke/CodeLlama-34B-GGUF"
            },
            "Orca 2 13B GGUF": {
                "size": "13B",
                "min_ram": 16,
                "recommended_ram": 32,
                "min_vram": 8,
                "recommended_vram": 12,
                "model_file_size": 7.3,
                "quantization": "Q4_0",
                "use_case": "Reasoning, math, science",
                "performance": "Strong analytical capabilities",
                "download_url": "https://huggingface.co/microsoft/Orca-2-13b"
            },
            "Zephyr 7B Beta GGUF": {
                "size": "7B",
                "min_ram": 6,
                "recommended_ram": 12,
                "min_vram": 3,
                "recommended_vram": 6,
                "model_file_size": 4.1,
                "quantization": "Q4_0",
                "use_case": "Helpful assistant, aligned responses",
                "performance": "Well-aligned, helpful",
                "download_url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF"
            }
        }

    def get_recommendations(self, ram_gb, vram_gb):
        """Generate model recommendations"""
        recommended = []
        compatible = []
        not_recommended = []
        
        for model_name, specs in self.model_database.items():
            if (ram_gb >= specs['recommended_ram'] and 
                vram_gb >= specs['recommended_vram']):
                recommended.append((model_name, specs))
            elif (ram_gb >= specs['min_ram'] and 
                  vram_gb >= specs['min_vram']):
                compatible.append((model_name, specs))
            else:
                not_recommended.append((model_name, specs))
        
        # Sort by model size
        recommended.sort(key=lambda x: x[1]['model_file_size'])
        compatible.sort(key=lambda x: x[1]['model_file_size'])
        not_recommended.sort(key=lambda x: x[1]['model_file_size'])
        
        return recommended, compatible, not_recommended

def main():
    st.title("🤖 LLM Model Recommender for LM Studio")
    st.markdown("Find the best GGUF models for your system specifications")
    
    app = LLMRecommenderWeb()
    
    # Auto-detect system specs on first load
    if 'auto_detected' not in st.session_state:
        with st.spinner("🔍 Auto-detecting system specifications..."):
            st.session_state.detected_ram = app.detector.get_total_ram_gb()
            st.session_state.detected_gpu_info = app.detector.get_gpu_info()
            st.session_state.detected_cpu_info = app.detector.get_cpu_info()
            st.session_state.auto_detected = True
    
    # Sidebar for system specs
    st.sidebar.header("🔧 System Specifications")
    
    # Show auto-detection status
    if st.session_state.auto_detected:
        st.sidebar.success("✅ System auto-detected!")
        if st.sidebar.button("🔄 Re-detect System"):
            with st.spinner("🔍 Re-detecting system..."):
                st.session_state.detected_ram = app.detector.get_total_ram_gb()
                st.session_state.detected_gpu_info = app.detector.get_gpu_info()
                st.session_state.detected_cpu_info = app.detector.get_cpu_info()
                st.rerun()
    
    # System specs inputs with auto-detected defaults
    ram_gb = st.sidebar.number_input(
        "Total RAM (GB)", 
        min_value=1, 
        max_value=256, 
        value=st.session_state.get('detected_ram', 16),
        help="Auto-detected from your system. Adjust if needed."
    )
    
    vram_gb = st.sidebar.number_input(
        "GPU VRAM (GB)", 
        min_value=0, 
        max_value=80, 
        value=st.session_state.get('detected_gpu_info', {}).get('vram_gb', 8),
        help="Auto-detected from your GPU. Enter 0 if using CPU only."
    )
    
    gpu_name = st.sidebar.text_input(
        "GPU Name", 
        value=st.session_state.get('detected_gpu_info', {}).get('name', ''),
        help="Auto-detected GPU name. Edit if incorrect."
    )
    
    # Show detection details
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ **Detected System Info**")
    
    if st.session_state.get('auto_detected'):
        cpu_info = st.session_state.get('detected_cpu_info', {})
        st.sidebar.markdown(f"**💻 OS:** {platform.system()} {platform.release()}")
        st.sidebar.markdown(f"**🔧 CPU:** {cpu_info.get('cores', 'Unknown')} cores / {cpu_info.get('threads', 'Unknown')} threads")
        st.sidebar.markdown(f"**💾 RAM:** {st.session_state.get('detected_ram', 'Unknown')} GB")
        st.sidebar.markdown(f"**🎮 GPU:** {gpu_name or 'Not detected'}")
        st.sidebar.markdown(f"**📊 VRAM:** {vram_gb} GB")
    
    # Manual detection help
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Manual Detection:")
    with st.sidebar.expander("How to verify specs manually"):
        st.markdown("""
        **Windows:**
        - RAM: Task Manager → Performance → Memory
        - GPU: Task Manager → Performance → GPU → Dedicated GPU memory
        
        **Mac:**
        - RAM: Apple Menu → About This Mac
        - GPU: Apple Menu → About This Mac → System Report
        
        **Linux:**
        - RAM: `free -h` or `lsmem`
        - GPU: `nvidia-smi` or `lspci | grep VGA`
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "📊 System Summary", "🔍 Detection Details", "📚 Model Database"])
    
    with tab1:
        st.header("Model Recommendations")
        
        if ram_gb > 0:
            recommended, compatible, not_recommended = app.get_recommendations(ram_gb, vram_gb)
            
            # System summary at top
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💾 Total RAM", f"{ram_gb} GB")
            with col2:
                st.metric("🎮 GPU VRAM", f"{vram_gb} GB")
            with col3:
                st.metric("🔧 GPU", gpu_name if gpu_name else "Not specified")
            
            st.markdown("---")
            
            # Recommendations
            if recommended:
                st.success("✅ **Highly Recommended Models**")
                st.markdown("These models will run smoothly on your system:")
                
                for model_name, specs in recommended:
                    with st.expander(f"📦 **{model_name}** ({specs['size']} • {specs['model_file_size']} GB)", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**🎯 Use Case:** {specs['use_case']}")
                            st.write(f"**⚡ Performance:** {specs['performance']}")
                            st.write(f"**📏 Quantization:** {specs['quantization']}")
                        with col2:
                            st.write(f"**💾 Min RAM:** {specs['min_ram']} GB")
                            st.write(f"**🎮 Min VRAM:** {specs['min_vram']} GB")
                            st.link_button("📥 Download from Hugging Face", specs['download_url'])
            
            if compatible:
                st.warning("⚠️ **Compatible Models** (May run slower)")
                st.markdown("These models meet minimum requirements but may be slower:")
                
                for model_name, specs in compatible:
                    with st.expander(f"📦 **{model_name}** ({specs['size']} • {specs['model_file_size']} GB)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**🎯 Use Case:** {specs['use_case']}")
                            st.write(f"**⚡ Performance:** {specs['performance']}")
                        with col2:
                            st.write(f"**💾 Requires:** {specs['min_ram']} GB RAM")
                            st.write(f"**🎮 Requires:** {specs['min_vram']} GB VRAM")
                            st.link_button("📥 Download", specs['download_url'])
                        
                        st.info("💡 This model may require CPU-only inference or slower performance")
            
            if not recommended and not compatible:
                st.error("❌ **No Compatible Models Found**")
                st.markdown("Unfortunately, none of the models in our database are compatible with your current specifications.")
                
                # Show what's needed for smallest model
                smallest_model = min(app.model_database.items(), key=lambda x: x[1]['min_ram'])
                model_name, specs = smallest_model
                
                st.info(f"💡 **To run the smallest model** ({model_name}):")
                st.write(f"• You would need at least **{specs['min_ram']} GB RAM** and **{specs['min_vram']} GB VRAM**")
                st.write(f"• Currently you have: **{ram_gb} GB RAM** and **{vram_gb} GB VRAM**")
            
            # Performance tips
            st.markdown("---")
            st.markdown("### 💡 **Performance Tips**")
            tips_col1, tips_col2 = st.columns(2)
            
            with tips_col1:
                st.markdown("""
                **🚀 For Better Performance:**
                • Close other applications when running models
                • Use quantized models (Q4_0) for faster loading
                • Enable GPU acceleration in LM Studio
                • Consider upgrading RAM for larger models
                """)
            
            with tips_col2:
                st.markdown("""
                **🔧 LM Studio Setup:**
                • Download GGUF format models only
                • Set context length based on available memory  
                • Use GPU layers if you have NVIDIA GPU
                • Monitor memory usage during inference
                """)
    
    with tab2:
        st.header("System Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💾 Total RAM", f"{ram_gb} GB")
            st.metric("🔧 GPU Model", gpu_name if gpu_name else "Not specified")
        
        with col2:
            st.metric("🎮 GPU VRAM", f"{vram_gb} GB")
            st.metric("📊 System Type", "GPU Accelerated" if vram_gb > 0 else "CPU Only")
        
        # Capability assessment
        st.markdown("---")
        st.markdown("### 🎯 **System Capability Assessment**")
        
        if ram_gb >= 64:
            st.success("🏆 **High-End System** - Can run any model including 70B+ parameters")
        elif ram_gb >= 32:
            st.success("🚀 **High Performance** - Great for 13B-34B parameter models")
        elif ram_gb >= 16:
            st.info("⚡ **Good Performance** - Perfect for 7B-13B parameter models")
        elif ram_gb >= 8:
            st.warning("📱 **Entry Level** - Best with 7B parameter models")
        else:
            st.error("⚠️ **Limited** - May struggle with most LLM models")
        
        if vram_gb >= 16:
            st.success("🎮 **Excellent GPU** - Full GPU acceleration available")
        elif vram_gb >= 8:
            st.info("🎯 **Good GPU** - GPU acceleration for most models")
        elif vram_gb >= 4:
            st.warning("⚡ **Basic GPU** - Limited GPU acceleration")
        else:
            st.info("💻 **CPU Mode** - Will use CPU-only inference")
    
    with tab3:
        st.header("🔍 System Detection Details")
        
        if st.session_state.get('auto_detected'):
            st.success("✅ System specifications were automatically detected!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💻 **System Information**")
                st.write(f"**Operating System:** {platform.system()} {platform.release()}")
                st.write(f"**Architecture:** {platform.machine()}")
                st.write(f"**Python Version:** {platform.python_version()}")
                
                cpu_info = st.session_state.get('detected_cpu_info', {})
                st.markdown("### 🔧 **CPU Details**")
                st.write(f"**Processor:** {cpu_info.get('name', 'Unknown')[:50]}...")
                st.write(f"**Physical Cores:** {cpu_info.get('cores', 'Unknown')}")
                st.write(f"**Logical Threads:** {cpu_info.get('threads', 'Unknown')}")
            
            with col2:
                st.markdown("### 💾 **Memory Information**")
                try:
                    mem = psutil.virtual_memory()
                    st.write(f"**Total RAM:** {round(mem.total / (1024**3), 1)} GB")
                    st.write(f"**Available RAM:** {round(mem.available / (1024**3), 1)} GB")
                    st.write(f"**Used RAM:** {round(mem.used / (1024**3), 1)} GB")
                except Exception as e:
                    st.write(f"**Total RAM:** {ram_gb} GB (detected)")
                    st.write(f"**Memory details unavailable:** {str(e)}")
                
                gpu_info = st.session_state.get('detected_gpu_info', {})
                st.markdown("### 🎮 **GPU Information**")
                st.write(f"**GPU Name:** {gpu_info.get('name', 'Unknown')}")
                st.write(f"**VRAM:** {gpu_info.get('vram_gb', 0)} GB")
        
        else:
            st.warning("⚠️ System auto-detection has not run yet.")
            if st.button("🔍 Detect System Now"):
                with st.spinner("Detecting system specifications..."):
                    st.session_state.detected_ram = app.detector.get_total_ram_gb()
                    st.session_state.detected_gpu_info = app.detector.get_gpu_info()
                    st.session_state.detected_cpu_info = app.detector.get_cpu_info()
                    st.session_state.auto_detected = True
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚙️ **Detection Methods Used**")
        st.info("""
        **RAM Detection:** Uses `psutil.virtual_memory()` to get total system RAM
        
        **GPU Detection:** 
        1. First tries `nvidia-smi` for NVIDIA GPUs (most accurate)
        2. Falls back to OS-specific methods:
           - Windows: `wmic` commands
           - macOS: `system_profiler` 
           - Linux: `lspci` commands
        
        **CPU Detection:** Uses `platform` and `psutil` modules
        
        **Note:** Detection accuracy may vary depending on system configuration and available tools.
        """)
    
    with tab4:
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
            ("🚀 **Small Models (7B Parameters)**", small_models, "Perfect for most users, fast inference"),
            ("⚡ **Medium Models (13B Parameters)**", medium_models, "Better quality, moderate resource usage"),
            ("🏋️ **Large Models (30B+ Parameters)**", large_models, "Highest quality, requires powerful hardware")
        ]
        
        for category_name, models, description in categories:
            if models:
                st.markdown(f"### {category_name}")
                st.markdown(f"*{description}*")
                
                for model_name, specs in models:
                    with st.expander(f"📦 **{model_name}**"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**📏 Size:** {specs['size']}")
                            st.write(f"**💾 File Size:** {specs['model_file_size']} GB")
                            st.write(f"**🔧 Quantization:** {specs['quantization']}")
                        with col2:
                            st.write(f"**💾 Min RAM:** {specs['min_ram']} GB")
                            st.write(f"**🎮 Min VRAM:** {specs['min_vram']} GB")
                            st.write(f"**💾 Recommended RAM:** {specs['recommended_ram']} GB")
                        with col3:
                            st.write(f"**🎯 Use Case:** {specs['use_case']}")
                            st.write(f"**⚡ Performance:** {specs['performance']}")
                            st.link_button("📥 Download", specs['download_url'])
                
                st.markdown("---")
        
        # GGUF format info
        st.markdown("### 🔧 **About GGUF Format**")
        st.info("""
        **GGUF** (GPT-Generated Unified Format) is the recommended format for LM Studio:
        
        • **Quantization reduces file size and memory usage**
        • **Q4_0**: 4-bit quantization - good balance of size and quality
        • **Q5_1**: 5-bit quantization - better quality, larger size  
        • **Q8_0**: 8-bit quantization - highest quality, largest size
        
        💡 **Tip**: Start with Q4_0 quantization for the best balance of performance and quality.
        """)

if __name__ == "__main__":
    main()