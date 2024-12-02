import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Any

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('ANTHROPIC_API_KEY')

class DataProcessor:
    """Handles all data processing operations"""
    
    @staticmethod
    def clean_percentage(value: Any) -> float:
        if pd.isna(value) or value == 0 or value == '0':
            return 0.0
        if isinstance(value, str):
            try:
                return float(value.strip('%')) / 100
            except ValueError:
                return 0.0
        return value if value <= 1 else value / 100

    @staticmethod
    def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        percentage_cols = ['CTR', 'Impr. (Abs. Top) %', 'Impr. (Top) %', 'Conv. rate']
        for col in percentage_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].apply(DataProcessor.clean_percentage)
        
        numeric_cols = ['Clicks', 'Impr.', 'Cost', 'Conversions', 'View-through conv.', 
                       'Avg. CPC', 'Cost / conv.']
        for col in numeric_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
        
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Day'])
        df_cleaned['CPC'] = df_cleaned['Cost'] / df_cleaned['Clicks'].replace(0, np.nan)
        df_cleaned['CPM'] = (df_cleaned['Cost'] / df_cleaned['Impr.'].replace(0, np.nan)) * 1000
        
        return df_cleaned

class PPCAnalyzer:
    def __init__(self):
        """Initialize PPC Analyzer with API key from environment"""
        if not API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
            
        self.llm = ChatAnthropic(
            api_key=API_KEY,
            model="claude-3-opus-20240229"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert PPC consultant analyzing Google Ads data.
                         Focus on providing actionable insights in these areas:
                         
                         1. Performance Analysis
                         - Campaign efficiency and ROI
                         - Geographic performance differences
                         - Quality score insights
                         - Conversion patterns
                         
                         2. Strategic Recommendations
                         - Specific optimization opportunities
                         - Budget allocation suggestions
                         - Testing recommendations
                         - Scaling opportunities
                         
                         3. Competitive Position
                         - Impression share analysis
                         - Market opportunity identification
                         - Bidding strategy recommendations
                         
                         Format as a professional consulting report with clear sections
                         and prioritized action items."""),
            ("user", "{analysis_data}")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def analyze_data(self, df: pd.DataFrame) -> Tuple[str, Dict[str, go.Figure]]:
        df_cleaned = DataProcessor.prepare_data(df)
        figures = self.create_visualizations(df_cleaned)
        analysis = self.generate_analysis(df_cleaned)
        return analysis, figures
    
    def create_visualizations(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        figures = {}
        
        # Performance Trends
        daily_metrics = df.groupby('Date').agg({
            'Cost': 'sum',
            'Clicks': 'sum',
            'Conversions': 'sum',
            'Impr.': 'sum'
        }).reset_index()
        
        fig_trends = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Cost', 'Daily Clicks', 'Daily Conversions', 'Daily Impressions')
        )
        
        fig_trends.add_trace(
            go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Cost'],
                      mode='lines+markers', name='Cost'),
            row=1, col=1
        )
        
        fig_trends.add_trace(
            go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Clicks'],
                      mode='lines+markers', name='Clicks'),
            row=1, col=2
        )
        
        fig_trends.add_trace(
            go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Conversions'],
                      mode='lines+markers', name='Conversions'),
            row=2, col=1
        )
        
        fig_trends.add_trace(
            go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Impr.'],
                      mode='lines+markers', name='Impressions'),
            row=2, col=2
        )
        
        fig_trends.update_layout(height=800, showlegend=True)
        figures['trends'] = fig_trends
        
        # Campaign Performance
        campaign_perf = df.groupby('Campaign').agg({
            'Cost': 'sum',
            'Clicks': 'sum',
            'Conversions': 'sum',
            'Impr.': 'sum'
        }).reset_index()
        
        fig_campaign = px.treemap(
            campaign_perf,
            path=[px.Constant("All Campaigns"), 'Campaign'],
            values='Cost',
            color='Clicks',
            title='Campaign Performance Overview'
        )
        
        figures['campaign_performance'] = fig_campaign
        
        return figures
    
    def generate_analysis(self, df: pd.DataFrame) -> str:
        summary_data = self.prepare_summary_data(df)
        response = self.chain.invoke({"analysis_data": summary_data})
        return response["text"]
    
    def prepare_summary_data(self, df: pd.DataFrame) -> str:
        overall_metrics = f"""
        Campaign Performance Analysis ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})
        
        Overall Performance:
        - Total Spend: ${df['Cost'].sum():,.2f}
        - Total Clicks: {df['Clicks'].sum():,}
        - Total Conversions: {df['Conversions'].sum():,}
        - Average CPC: ${df['Cost'].sum() / df['Clicks'].sum():.2f}
        - Overall CTR: {(df['Clicks'].sum() / df['Impr.'].sum() * 100):.2f}%
        
        Campaign Breakdown:
        {df.groupby('Campaign').agg({
            'Cost': 'sum',
            'Clicks': 'sum',
            'Conversions': 'sum',
            'Impr.': 'sum'
        }).round(2).to_string()}
        """
        return overall_metrics

def main():
    st.set_page_config(page_title="PPC Campaign Analysis Dashboard", layout="wide")
    
    st.title("PPC Campaign Analysis Dashboard")
    
    # Check for API key
    if not API_KEY:
        st.error("""
        No API key found in environment variables. 
        Please set the ANTHROPIC_API_KEY in your .env file:
        
        1. Create a file named '.env' in your project directory
        2. Add the line: ANTHROPIC_API_KEY=your-api-key-here
        3. Restart the application
        """)
        st.stop()
    
    st.write("Upload your Google Ads campaign data for analysis")
    
    uploaded_file = st.file_uploader("Choose your campaign data CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Analysis", "Visualizations", "Raw Data"])
            
            analyzer = PPCAnalyzer()
            analysis, figures = analyzer.analyze_data(data)
            
            with tab1:
                st.subheader("Campaign Analysis")
                st.write(analysis)
                
                st.download_button(
                    label="Download Analysis Report",
                    data=analysis,
                    file_name=f"ppc_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.subheader("Performance Visualizations")
                for name, fig in figures.items():
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Raw Campaign Data")
                st.dataframe(data)
        
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.write("Please ensure your CSV file matches the expected format.")

if __name__ == "__main__":
    main()