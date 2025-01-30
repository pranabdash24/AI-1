import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer  # Load model and tokenizer directly

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 12px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .data-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load DeepSeek-R1 model and tokenizer
def load_deepseek_model():
    model_name = "deepseek-ai/DeepSeek-R1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

# Function to generate business summary and suggestions using DeepSeek-R1
def generate_business_summary(data_summary, kpis, model, tokenizer):
    prompt = f"""
    You are a business analyst. Analyze the following data summary and KPIs, and provide a business summary, feedback, and suggestions:
    
    Data Summary:
    {data_summary}
    
    Key KPIs:
    {kpis}
    
    Provide a concise summary of the business performance (Good/Bad), feedback on the current scenario, and actionable suggestions for improvement.
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to identify KPIs based on the dataset and user inputs
def identify_kpis(df, industry, goal):
    kpis = []
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Industry-specific KPIs
    if industry == "Retail":
        kpis.extend(["Sales Growth", "Customer Retention Rate", "Average Transaction Value"])
    elif industry == "E-commerce":
        kpis.extend(["Conversion Rate", "Customer Acquisition Cost", "Cart Abandonment Rate"])
    elif industry == "Manufacturing":
        kpis.extend(["Production Efficiency", "Defect Rate", "Inventory Turnover"])

    # Goal-specific KPIs
    if goal == "Increase Revenue":
        kpis.append("Revenue Growth")
    elif goal == "Reduce Costs":
        kpis.append("Cost Reduction Percentage")
    elif goal == "Improve Customer Satisfaction":
        kpis.append("Net Promoter Score (NPS)")

    # Add numeric columns as potential KPIs
    kpis.extend(numeric_columns)

    return list(set(kpis))  # Remove duplicates

# Function to create a dashboard with relevant KPIs
def create_dashboard(df, kpis):
    st.write("### Business Dashboard")
    for kpi in kpis:
        if kpi in df.columns:
            st.write(f"#### {kpi}")
            fig = px.line(df, x=df.index, y=kpi, title=f"{kpi} Over Time")
            st.plotly_chart(fig)

# Main function to analyze and visualize the business data
def analyze_business_data(file, model, tokenizer):
    try:
        # Load the dataset
        if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            st.error("Unsupported file type. Please upload an Excel or CSV file.")
            return

        st.markdown("<div class='data-section'>", unsafe_allow_html=True)
        st.markdown("### Data Loaded Successfully", unsafe_allow_html=True)
        st.write("#### Data Preview:")
        st.write(df.head())

        # Q/A Session
        st.sidebar.write("### Business Information")
        industry = st.sidebar.selectbox(
            "What industry does your business belong to?",
            ["Retail", "E-commerce", "Manufacturing", "Healthcare", "Other"]
        )
        goal = st.sidebar.selectbox(
            "What is your primary business goal?",
            ["Increase Revenue", "Reduce Costs", "Improve Customer Satisfaction", "Expand Market Share"]
        )

        # Identify KPIs
        kpis = identify_kpis(df, industry, goal)
        st.write("### Identified KPIs")
        st.write(kpis)

        # Create Dashboard
        create_dashboard(df, kpis)

        # Generate Business Summary
        data_summary = df.describe().to_string()
        business_summary = generate_business_summary(data_summary, kpis, model, tokenizer)
        st.write("### Business Summary and Suggestions")
        st.write(business_summary)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit app
st.markdown("<h1 class='main-header'>Business Analysis App</h1>", unsafe_allow_html=True)
st.write("Upload your business data and answer a few questions to get insights and recommendations.")

# Load DeepSeek-R1 model and tokenizer
model, tokenizer = load_deepseek_model()

# File upload
uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    analyze_business_data(uploaded_file, model, tokenizer)