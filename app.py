import streamlit as st
from transformers import pipeline

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    classifier = pipeline('text-classification', model='fake_news')
    return classifier

# Title and description
st.title("📰 Fake News Detection")
st.markdown("---")
st.write("Enter a news headline or text to check if it's likely to be fake or real news.")

# Load classifier
classifier = load_model()

# Input section
st.subheader("Input Text")
user_input = st.text_area(
    "Paste your news headline or text here:",
    height=150,
    placeholder="Enter text to analyze..."
)

# Prediction section
if user_input:
    if st.button("🔍 Check News", type="primary"):
        with st.spinner("Analyzing..."):
            result = classifier(user_input)
        
        # Display results
        st.subheader("Results")
        
        label = result[0]['label']
        score = result[0]['score']
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Classification", label.upper())
        
        with col2:
            st.metric("Confidence", f"{score:.2%}")
        
        # Visual indicator
        if label.lower() == "fake":
            st.warning(f"⚠️ **Likely Fake News** (Confidence: {score:.2%})")
        else:
            st.success(f"✅ **Likely Real News** (Confidence: {score:.2%})")
        
        # Detailed information
        st.divider()
        st.subheader("Prediction Details")
        st.json({
            "Label": label,
            "Confidence Score": f"{score:.4f}",
            "Input Text": user_input[:100] + "..." if len(user_input) > 100 else user_input
        })

else:
    st.info("👆 Enter some text above to get started!")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <p><small>Built with DistilBERT | Fake News Detection Model</small></p>
    </div>
    """,
    unsafe_allow_html=True
)