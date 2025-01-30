import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer

# Custom CSS for styling
st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #4CAF50; text-align: center; }
.data-section { background: #f9f9f9; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

def load_deepseek_model():
    """Load model with version-pinned commit"""
    model_name = "deepseek-ai/DeepSeek-R1"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision="5dde110d1a9ee857b90a6710b7138f9130ce6fa0"  # Critical commit
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision="5dde110d1a9ee857b90a6710b7138f9130ce6fa0"
    )
    return model, tokenizer

def generate_business_insights(data_summary, kpis, model, tokenizer):
    """Generate AI-powered business analysis"""
    prompt = f"""Analyze this business data and KPIs:
    
    Data Summary:
    {data_summary}
    
    Key Metrics:
    {kpis}
    
    Provide a concise performance summary (Good/Bad) and actionable suggestions:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    st.title("Business Analysis Assistant")
    
    # Load model once
    if 'model' not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.model, st.session_state.tokenizer = load_deepseek_model()
    
    # File upload
    uploaded_file = st.file_uploader("Upload business data (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            # Load data
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Show data preview
            with st.expander("Raw Data Preview"):
                st.write(df.head())
            
            # Business context
            st.sidebar.header("Business Context")
            industry = st.sidebar.selectbox("Industry", ["Retail", "E-commerce", "Manufacturing"])
            goal = st.sidebar.selectbox("Primary Goal", ["Increase Revenue", "Reduce Costs"])
            
            # Auto-identify KPIs
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            kpis = numeric_cols + ["Custom Metric 1", "Custom Metric 2"]  # Add your logic
            
            # Dashboard
            st.subheader("Business Dashboard")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(df, x=df.index, y=numeric_cols[0], title=f"{numeric_cols[0]} Trend")
                st.plotly_chart(fig)
            with col2:
                fig = px.bar(df.tail(10), x=df.index, y=numeric_cols[1], title="Recent Performance")
                st.plotly_chart(fig)
            
            # Generate insights
            if st.button("Analyze Business Performance"):
                data_summary = df.describe().to_string()
                analysis = generate_business_insights(
                    data_summary, kpis, 
                    st.session_state.model, st.session_state.tokenizer
                )
                st.subheader("AI Analysis")
                st.write(analysis)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
