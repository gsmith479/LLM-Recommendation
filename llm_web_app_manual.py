import streamlit as st
import platform
from datetime import datetime

# Page config
st.set_page_config(
    page_title="LLM Model Recommender",
    page_icon="🤖",
    layout="wide"
)

class LLMRecommenderWeb:
    def __init__(self):
        self.model_database = self.load_model_database()

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
    
    # Sidebar for system specs input
    st.sidebar.header("🔧 System Specifications")
    st.sidebar.markdown("Enter your system specifications below:")
    
    ram_gb = st.sidebar.number_input(
        "Total RAM (GB)", 
        min_value=1, 
        max_value=256, 
        value=16,
        help="How much total RAM does your system have?"
    )
    
    vram_gb = st.sidebar.number_input(
        "GPU VRAM (GB)", 
        min_value=0, 
        max_value=80, 
        value=8,
        help="How much VRAM does your GPU have? Enter 0 if using CPU only."
    )
    
    gpu_name = st.sidebar.text_input(
        "GPU Name (optional)", 
        value="",
        placeholder="e.g., RTX 4090, RTX 3080, etc."
    )
    
    # Add some helpful info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 How to find your specs:")
    st.sidebar.markdown("""
    **Windows:**
    - RAM: Task Manager → Performance → Memory
    - GPU: Task Manager → Performance → GPU
    
    **Mac:**
    - RAM: Apple Menu → About This Mac
    - GPU: Apple Menu → About This Mac
    
    **Linux:**
    - RAM: `free -h` command
    - GPU: `nvidia-smi` or `lspci | grep VGA`
    """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 System Summary", "📚 Model Database"])
    
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
