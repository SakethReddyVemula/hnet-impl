import streamlit as st
import torch
import os
import glob
from hnet_impl import HNetLM, HNetConfig, ByteTokenizer, completion_sync
from hnet_impl.sampling import colorize_prefill, aggregate_bytes_to_utf8
from torch import nested, Tensor as TT
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from contextlib import nullcontext
# Mock NJT and other torchisms if not directly importable or if they depend on distributed setup not present in valid inference
# However, hnet_impl seems to handle imports gracefully.

# --- Helper Functions ---

@st.cache_resource
def load_model(checkpoint_path, model_dim, model_arch):
    """Loads the H-Net model from a checkpoint."""
    # Create config
    # Parse model_dim and model_arch from strings if needed, or assume they are lists
    # Expecting inputs like [256, 256] and ["m2", "T4"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Re-create config
    c = HNetConfig.create_reasonable_config(D=model_dim, arch=model_arch)
    with device:
        m = HNetLM(c)
    
    # Load weights
    try:
        # Map location to device to avoid CUDA OOM if loading multiple models or if on CPU
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only=False per existing code pattern
        
        # Sanitize state_dict to handle DTensors if present
        new_state_dict = {}
        for k, v in state_dict.items():
            # If v is a DTensor (DistributedTensor), we need to extract the local tensor.
            # Since the shapes match (per user error log), it implies we have the full tensor data 
            # likely wrapped or we are just running into type checks.
            # Forcing it to a standard tensor by creating a new one or using to_local() if available.
            
            # Check if it's a DTensor by name or type check if possible, 
            # but simpler generic approach for torch objects:
            if hasattr(v, 'to_local'):
                # It's a DTensor
                v = v.to_local()
            
            # Even if to_local() returns a Tensor, it might still have some distributed attributes 
            # if we are not careful, or maybe it's fine. 
            # However, looking at the error: "got mixed torch.Tensor and DTensor"
            # We want purely torch.Tensor.
            
            if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter):
                 # converting to parameter might happen inside load_state_dict
                 pass

            new_state_dict[k] = v

        m.load_state_dict(new_state_dict)
        m.eval()
        return m, c
    except Exception as e:
        st.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None

def get_checkpoints(directory):
    """List all .pt files in the directory sorted by modification time or name."""
    if not os.path.exists(directory):
        return []
    files = glob.glob(os.path.join(directory, "*.pt"))
    # Sort by epoch number if possible, else by name
    # assuming format "checkpoint_epoch_N.pt"
    def sort_key(f):
        try:
            return int(f.split("_epoch_")[-1].split(".")[0])
        except:
            return 0
    return sorted(files, key=sort_key)

# Styles for visualization
CSS = """
<style>
.segment-text {
    font-family: monospace;
    font-size: 16px;
    padding: 2px 0;
    line-height: 2.0;
}
.seg-chunk {
    border-radius: 4px;
    padding: 2px 1px;
    margin: 0 1px;
}
</style>
"""

# Map termcolor colors to CSS colors
COLOR_MAP = {
    "light_yellow": "#FFD700", # Gold
    "light_green": "#90EE90",  # LightGreen
    "default": "#FFFFFF"       # White
}
BG_MAP = {
    "on_black": "transparent",
    "on_dark_grey": "#444444", # DarkGray
    "default": "transparent"
}

def html_colorize(text, color, on_color, attrs):
    """Generates HTML span for a character/segment."""
    css_color = COLOR_MAP.get(color, "white")
    css_bg = BG_MAP.get(on_color, "transparent")
    
    style_str = f"color: {css_color}; background-color: {css_bg};"
    
    classes = "segment-text seg-chunk"
    if "underline" in attrs:
        style_str += " text-decoration: underline;"
    if "blink" in attrs:
        # Blink is annoying in web, maybe just bold or border?
        style_str += " border: 1px solid red;" 
    if "dark" in attrs:
        style_str += " opacity: 0.6;"
    
    return f'<span class="{classes}" style="{style_str}">{text}</span>'

def visualize_segmentation(model, text):
    """Runs the model on text and returns HTML representation of segments."""
    if not text:
        return ""
        
    tokenizer = ByteTokenizer()
    device = next(model.parameters()).device
    
    # Encode
    tokens = tokenizer.encode([text])[0]
    # Remove BOS if added by tokenizer, although colorize_byte_prefill usually handles raw text.
    # checking sampling.py: colorize_byte_prefill takes str -> encode -> [1:].cuda()
    # So we should follow that.
    
    iids = tokens[1:].to(device) # skip BOS
    
    # Use the logic from colorize_prefill but capturing output instead of printing
    html_output = ["<div class='segment-container'>"]
    
    # Need to access model attributes and methods
    # Using the sampling logic
    try:
        from hnet_impl.sampling import colorize_prefill, aggregate_bytes_to_utf8, COLOR_CYCLE, HIGHLIGHT_CYCLE, get_termcolor_from_boundaries
        
        # We need to access some internals or rely on existing function generator
        # The existing 'colorize_prefill' yields (char, dict(color, on_color, attrs))
        
        with model.sampling_mode():
             # We need to replicate colorize_prefill logic because we can't easily hook into it 
             # if we want to change how it prints, BUT colorize_prefill yields token+style, 
             # so we can just iterate it!
            
             gen = colorize_prefill(model, iids, aggregate_bytes_to_utf8)
             
             for char, style in gen:
                 # style is dict(color=..., on_color=..., attrs=...)
                 html_fragment = html_colorize(
                     char, 
                     style.get('color'), 
                     style.get('on_color'), 
                     style.get('attrs', [])
                 )
                 html_output.append(html_fragment)
                 
    except Exception as e:
        return f"<div style='color:red'>Visualization Error: {str(e)}</div>"
        
    html_output.append("</div>")
    return "".join(html_output)


# --- Main App ---

def main():
    st.set_page_config(page_title="H-Net Visualizer", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    
    st.title("H-Net Segmentation Visualizer")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    default_lang = "eng"
    lang_code = st.sidebar.text_input("Language Code", value=default_lang)
    
    # Hugging Face Config
    st.sidebar.subheader("Hugging Face Source")
    hf_repo = st.sidebar.text_input("HF Repo ID", value=os.environ.get("HF_REPO_ID", ""))
    hf_token = st.sidebar.text_input("HF Token", value=os.environ.get("HF_TOKEN", ""), type="password")
    
    use_local = st.sidebar.checkbox("Use Local Checkpoints Only", value=not hf_repo)

    # Model Config (Defaults from script)
    st.sidebar.subheader("Model Architecture")
    dim_str = st.sidebar.text_input("Model Dimensions (D)", value="256 256")
    arch_str = st.sidebar.text_input("Model Architecture", value="m2 T4")
    
    try:
        model_dims = [int(x) for x in dim_str.split()]
        model_arch = arch_str.split()
    except:
        st.error("Invalid model configuration format")
        return

    # Checkpoint Discovery
    checkpoints = []
    
    if use_local:
        base_dir = os.path.abspath("checkpoints/hnets")
        ckpt_dir = st.sidebar.text_input("Local Checkpoint Directory", value=os.path.join(base_dir, lang_code))
        checkpoints = get_checkpoints(ckpt_dir)
        if not checkpoints:
            st.sidebar.warning(f"No checkpoints found in {ckpt_dir}")
    else:
        if hf_repo:
            try:
                from huggingface_hub import HfApi, hf_hub_download
                api = HfApi(token=hf_token if hf_token else None)
                # List files in the subfolder
                # Assuming checkpoints are like {lang_code}/checkpoint_epoch_X.pt or just checkpoint_epoch_X.pt depending on how they were uploaded.
                # The script says: export HF_SUBFOLDER="${LANG_CODE}"
                # upload_checkpoint(..., subfolder=...) -> path_in_repo = subfolder/filename
                
                subfolder = lang_code
                files = api.list_repo_files(repo_id=hf_repo)
                
                # Filter for checkpoints in the subfolder
                ckpt_files = [f for f in files if f.startswith(subfolder + "/") and f.endswith(".pt")]
                
                if not ckpt_files:
                    st.sidebar.warning(f"No .pt files found in {hf_repo}/{subfolder}")
                else:
                    # Sort them
                    def sort_key_hf(f):
                        try:
                            # expecting "eng/checkpoint_epoch_1.pt"
                            return int(f.split("_epoch_")[-1].split(".")[0])
                        except:
                            return 0
                    checkpoints = sorted(ckpt_files, key=sort_key_hf)
                    st.sidebar.success(f"Found {len(checkpoints)} remote checkpoints")
                    
            except Exception as e:
                st.sidebar.error(f"Error fetching from HF: {e}")
        else:
            st.sidebar.info("Enter HF Repo ID to fetch checkpoints.")

    # Selection
    if checkpoints:
        selected_ckpt_name = st.sidebar.selectbox(
            "Select Checkpoint", 
            options=checkpoints,
            format_func=lambda x: os.path.basename(x)
        )
        
        # Prepare the actual path to load
        selected_ckpt_path = None
        if use_local:
            selected_ckpt_path = selected_ckpt_name # It's already the full path from get_checkpoints
        else:
            # Download if remote
            if st.sidebar.button(f"Load {os.path.basename(selected_ckpt_name)}"):
                 try:
                     with st.spinner("Downloading from Hugging Face..."):
                         selected_ckpt_path = hf_hub_download(
                             repo_id=hf_repo,
                             filename=selected_ckpt_name,
                             token=hf_token if hf_token else None
                         )
                         st.session_state.current_ckpt_path = selected_ckpt_path
                 except Exception as e:
                     st.error(f"Download failed: {e}")
            
            # Use cached path if available
            if 'current_ckpt_path' in st.session_state and st.session_state.current_ckpt_path:
                 # Verify acts on the same file (simple check)
                 if os.path.basename(st.session_state.current_ckpt_path) == os.path.basename(selected_ckpt_name):
                     selected_ckpt_path = st.session_state.current_ckpt_path
    else:
        selected_ckpt_path = None
        
    # Main Content
    input_text = st.text_area("Input Text", value="Hello world. This is a test sentence for segmentation.", height=150)
    
    if st.button("Visualize"):
        if selected_ckpt_path and input_text:
            with st.spinner(f"Loading model..."):
                model, _ = load_model(selected_ckpt_path, model_dims, model_arch)
            
            if model:
                with st.spinner("Processing..."):
                    html_vis = visualize_segmentation(model, input_text)
                    st.markdown("### Segmentation Result")
                    st.markdown(html_vis, unsafe_allow_html=True)
        elif not selected_ckpt_path and not use_local:
             st.warning("Please load the checkpoint first.")

    # Animation Mode
    st.markdown("---")
    st.header("Checkpoint Animation")
    
    if checkpoints and input_text:
        run_anim = st.checkbox("Enable Animation Dashboard")
        
        if run_anim:
            # Slider
            ckpt_idx = st.slider("Checkpoint Index", 0, len(checkpoints)-1, 0)
            target_ckpt_name = checkpoints[ckpt_idx]
            
            st.write(f"Viewing: **{os.path.basename(target_ckpt_name)}**")
            
            # For animation, we likely need to auto-download. This might be slow.
            # warn user?
            # Or just do it.
            
            if 'anim_cached_idx' not in st.session_state:
                st.session_state.anim_cached_idx = -1
                st.session_state.anim_model = None

            # Load only if changed
            if st.session_state.anim_cached_idx != ckpt_idx:
                path_to_load = target_ckpt_name
                if not use_local:
                    try:
                        with st.spinner(f"Downloading {os.path.basename(target_ckpt_name)}..."):
                             path_to_load = hf_hub_download(
                                 repo_id=hf_repo,
                                 filename=target_ckpt_name,
                                 token=hf_token if hf_token else None
                             )
                    except Exception as e:
                        st.error(f"Download failed: {e}")
                        path_to_load = None

                if path_to_load:
                    with st.spinner("Switching checkpoint..."):
                        m, _ = load_model(path_to_load, model_dims, model_arch)
                        st.session_state.anim_model = m
                        st.session_state.anim_cached_idx = ckpt_idx
            
            if st.session_state.anim_model:
                html_vis = visualize_segmentation(st.session_state.anim_model, input_text)
                st.markdown(html_vis, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
