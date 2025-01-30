import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import re
import numpy as np
from datetime import datetime

# Configure Google Gemini API
GOOGLE_API_KEY = st.secrets["general"]["API_KEY"]  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

def send_data_to_gemini(df, industry, goal):
    df_subset = df.head(100)
    data_json = df_subset.to_json(orient="records", date_format='iso')

    prompt = f"""
    You are a business intelligence assistant analyzing {industry} data for {goal}. 
    Generate Python code using Plotly Express wrapped in ```python blocks. Follow these steps:

    1. Identify major KPIs from this data:
    {data_json}
    
    2. For each KPI:
    - Create a visualization using Plotly Express
    - Use st.plotly_chart(fig) to display it
    - Add titles/axis labels
    - Use Streamlit layout components
    
    3. Provide a 3-sentence insight after each chart

    Format response EXACTLY like:
    ```
    # KPI 1 Visualization
    fig1 = px.line(df, x='...', y='...')
    st.plotly_chart(fig1)
    ```
    Insight: [Your insight here]

    ```
    # KPI 2 Visualization 
    fig2 = px.bar(df, x='...', y='...')
    st.plotly_chart(fig2)
    ```
    Insight: [Your insight here]

    5. At last a proper summary of the bussniss, SWOT, and future recommendations.
    """
    

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Business Intelligence Assistant</h1>", unsafe_allow_html=True)
    st.warning("⚠️ This app executes AI-generated code. Only use with trusted data sources.")

    uploaded_file = st.file_uploader("Upload business data", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            
            with st.expander("📊 View Raw Data", expanded=True):
                st.dataframe(df.head(100), use_container_width=True)

            st.sidebar.header("Business Context")
            industry = st.sidebar.selectbox("Industry Sector", 
                ["Retail", "E-commerce", "Manufacturing", "Healthcare", "Finance"])
            goal = st.sidebar.selectbox("Primary Objective", 
                ["Revenue Growth", "Cost Optimization", "Customer Experience", "Operational Efficiency"])

            if st.button("Analyze Data & Generate Report"):
                with st.spinner("Analyzing data with AI..."):
                    report = send_data_to_gemini(df, industry, goal)
                try:
                    # Improved parsing with flexible regex
                    code_blocks = re.findall(r'```python\n(.*?)\n```', report, re.DOTALL)
                    insights = re.findall(r'Insight:\s*(.*?)(?=\n\s*```|$)', report, re.DOTALL)

                    if not code_blocks:
                        st.error("No code found in response. Common reasons:")
                        st.markdown("""
                        - API key not configured
                        - Dataset too small/vague
                        - Gemini content policy restrictions
                        """)
                        return

                    exec_env = {
                        'df': df.head(100),
                        'st': st,
                        'px': px,
                        'pd': pd,
                        'np': np,
                        'datetime': datetime
                    }

                    for i, code in enumerate(code_blocks):
                        with st.container():
                            with st.expander(f"View KPI {i+1} Code"):
                                st.code(code, language='python')
                            
                            try:
                                exec(code, exec_env)
                                if i < len(insights):
                                    st.success(f"**Insight {i+1}:** {insights[i].strip()}")
                            except Exception as e:
                                st.error(f"Error executing KPI {i+1}: {str(e)}")

                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")

        except Exception as e:
            st.error(f"Data Error: {str(e)}")

if __name__ == "__main__":
    main()
