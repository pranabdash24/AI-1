import streamlit as st
import pandas as pd
import plotly.express as px
import torch  # <-- Missing import added here
from transformers import AutoModelForCausalLM, AutoTokenizer

# Custom CSS styling
st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #4CAF50; text-align: center; margin-bottom: 2rem; }
.data-section { background: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

def load_deepseek_model():
    """Load DeepSeek-R1 model with version pinning"""
    model_name = "deepseek-ai/DeepSeek-R1"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision="main",
        device_map="auto"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision="main",
        device_map="auto",
        torch_dtype=torch.float16  # Now using properly imported torch
    )
    return model, tokenizer

def analyze_data(df, industry, goal):
    """Auto-identify KPIs based on data and user inputs"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    kpis = {
        "Retail": ["Monthly Sales", "Customer Retention"],
        "E-commerce": ["Conversion Rate", "Cart Abandonment"],
        "Manufacturing": ["Production Yield", "Defect Rate"]
    }.get(industry, []) + numeric_cols
    return list(set(kpis))

def main():
    st.markdown("<h1 class='main-header'>Business Intelligence Assistant</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        with st.spinner("Loading AI engine..."):
            st.session_state.model, st.session_state.tokenizer = load_deepseek_model()

    # File upload section
    uploaded_file = st.file_uploader("Upload business data", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            # Load data
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            
            # Data preview
            with st.expander("📊 View Raw Data", expanded=True):
                st.dataframe(df.head(), use_container_width=True)

            # Business context
            st.sidebar.header("Business Context")
            industry = st.sidebar.selectbox("Industry Sector", ["Retail", "E-commerce", "Manufacturing"])
            goal = st.sidebar.selectbox("Primary Objective", ["Revenue Growth", "Cost Optimization", "Customer Experience"])

            # Analysis flow
            kpis = analyze_data(df, industry, goal)
            
            # Dashboard
            st.subheader("Performance Dashboard")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(df, x=df.columns[0], y=kpis[0], title=f"{kpis[0]} Trend")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(df.tail(10), x=df.columns[0], y=kpis[-1], title="Recent Performance")
                st.plotly_chart(fig, use_container_width=True)

            # AI Analysis
            if st.button("Generate Business Report"):
                data_summary = f"""
                Records: {len(df)}
                Time Period: {df.iloc[:, 0].min()} to {df.iloc[:, 0].max()}
                Key Metrics: {', '.join(kpis)}
                """
                
                analysis = st.session_state.model.generate(
                    **st.session_state.tokenizer(
                        f"Analyze this {industry} business data focusing on {goal}:\n{data_summary}",
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    ),
                    max_new_tokens=500
                )
                
                st.subheader("AI Business Report")
                st.write(st.session_state.tokenizer.decode(analysis[0], skip_special_tokens=True))

        except Exception as e:
            st.error(f"🚨 Error processing data: {str(e)}")

if __name__ == "__main__":
    main()
