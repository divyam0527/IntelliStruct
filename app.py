import streamlit as st
import pandas as pd
import jsonlines
import os
import tempfile
import speech_recognition as sr
import pytesseract
from PIL import Image
from textblob import TextBlob
import plotly.express as px
from datetime import datetime
import base64
from fpdf import FPDF, XPos, YPos
import platform

# Set Tesseract path for Windows
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Parse Text File
def parse_text(file):
    return file.read().decode("utf-8")

# Parse CSV File
def parse_csv(file, text_column=None):
    df = pd.read_csv(file)
    
    # If no column specified, try to auto-detect text column
    if text_column is None:
        possible_columns = ['text', 'feedback', 'comment', 'review', 'content']
        for col in possible_columns:
            if col in df.columns:
                text_column = col
                break
    
    if text_column is None:
        st.error("Could not identify text column in CSV. Please specify which column contains the feedback text.")
        st.dataframe(df.head())
        text_column = st.selectbox("Select text column:", df.columns)
        
    return "\n".join(df[text_column].astype(str).tolist())

# Parse JSONL File
def parse_jsonl(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    with jsonlines.open(tmp_path) as reader:
        lines = [obj['text'] for obj in reader if 'text' in obj]
    os.remove(tmp_path)
    return "\n".join(lines)

# Parse Audio File (Windows compatible)
def parse_audio(file):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    try:
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                text = "Could not understand audio"
            except sr.RequestError as e:
                text = f"Error with speech recognition: {e}"
    except Exception as e:
        text = f"Error processing audio file: {str(e)}"
    finally:
        os.remove(tmp_path)
    return text

# Parse Image File
def parse_image(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    try:
        image = Image.open(tmp_path)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        text = f"Error processing image: {str(e)}"
    finally:
        os.remove(tmp_path)
    return text

# Generate PDF Report
class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", 'B', 16)
        self.cell(0, 10, 'Feedback Sentiment Analysis Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("helvetica", '', 12)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(10)

def generate_pdf_report(df, sentiment_counts, filename="sentiment_report.pdf"):
    pdf = PDF()
    pdf.add_page()
    
    # Set margins to ensure enough space
    pdf.set_margins(20, 20, 20)
    
    # Summary Statistics
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "Summary Statistics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=12)
    
    total_feedback = len(df)
    positive = len(df[df['Sentiment'] == 'Positive'])
    negative = len(df[df['Sentiment'] == 'Negative'])
    neutral = len(df[df['Sentiment'] == 'Neutral'])
    
    pdf.cell(0, 10, f"Total Feedback Items Analyzed: {total_feedback}", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, f"Positive Feedback: {positive} ({positive/total_feedback:.1%})", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, f"Negative Feedback: {negative} ({negative/total_feedback:.1%})", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, f"Neutral Feedback: {neutral} ({neutral/total_feedback:.1%})", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # Sample Feedback
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "Sample Feedback", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        samples = df[df['Sentiment'] == sentiment].head(3)
        if not samples.empty:
            pdf.set_font("helvetica", 'B', 12)
            pdf.cell(0, 10, f"{sentiment} Feedback Examples:", 
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("helvetica", size=10)
            for _, row in samples.iterrows():
                text = f"- {str(row['Original Text'])[:150]}..."
                pdf.multi_cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
    
    pdf.output(filename)
    return filename

# Streamlit App Setup
st.set_page_config(page_title="Feedback Sentiment Analyzer", layout="wide")
st.title("🧠 Structured Insights: Feedback Sentiment Analyzer")
st.write("Upload text, CSV, JSONL, audio, or image files to analyze sentiment and generate reports.")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "jsonl", "wav", "mp3", "png", "jpg", "jpeg"])

if uploaded_file:
    # File Parsing
    if uploaded_file.type == "text/plain":
        raw_text = parse_text(uploaded_file)
    elif uploaded_file.name.endswith(".csv"):
        raw_text = parse_csv(uploaded_file)
    elif uploaded_file.name.endswith(".jsonl"):
        raw_text = parse_jsonl(uploaded_file)
    elif "audio" in uploaded_file.type:
        raw_text = parse_audio(uploaded_file)
    elif "image" in uploaded_file.type:
        raw_text = parse_image(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # Prepare DataFrame
    if isinstance(raw_text, pd.DataFrame):
        df = raw_text.copy()
        if 'Original Text' not in df.columns:
            df['Original Text'] = df.iloc[:, 0]
    else:
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        df = pd.DataFrame(lines, columns=["Original Text"])
    
    df["Original Text"] = df["Original Text"].apply(lambda x: str(x)[:512])
    df["Sentiment"] = df["Original Text"].apply(analyze_sentiment)

    # Sentiment Filter
    sentiment_options = ["Positive", "Negative", "Neutral"]
    selected_sentiments = st.multiselect("Filter by Sentiment:", sentiment_options, default=sentiment_options)
    filtered_df = df[df["Sentiment"].isin(selected_sentiments)]

    # Chart Type Selector
    chart_type = st.selectbox("Choose chart type:", ["Bar Chart", "Pie Chart", "Both"])

    # Display Filtered Table
    st.subheader("🔍 Analysis Result")
    st.dataframe(filtered_df, use_container_width=True)

    # Sentiment Distribution
    sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    if chart_type in ["Bar Chart", "Both"]:
        st.subheader("📊 Sentiment Distribution - Bar Chart")
        st.bar_chart(sentiment_counts.set_index('Sentiment'))

    if chart_type in ["Pie Chart", "Both"]:
        fig = px.pie(sentiment_counts,
                     names='Sentiment',
                     values='Count',
                     title='📈 Sentiment Distribution - Pie Chart',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    # Auto-generated Report Section
    st.subheader("📑 Auto-generated Report")
    
    # Report Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Feedback Items", len(df))
    with col2:
        positive_count = len(df[df['Sentiment'] == 'Positive'])
        st.metric("Positive Feedback", f"{positive_count} ({positive_count/len(df):.1%})")
    with col3:
        negative_count = len(df[df['Sentiment'] == 'Negative'])
        st.metric("Negative Feedback", f"{negative_count} ({negative_count/len(df):.1%})")
    
    # Sample Feedback by Sentiment
    st.write("### Sample Feedback by Sentiment")
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        samples = df[df['Sentiment'] == sentiment].head(2)
        if not samples.empty:
            st.write(f"**{sentiment} Feedback Examples:**")
            for _, row in samples.iterrows():
                st.write(f"- {row['Original Text'][:150]}...")
    
    # Generate and Download Report
    st.write("### Download Full Report")
    if st.button("Generate Comprehensive PDF Report"):
        with st.spinner("Generating report..."):
            try:
                report_path = generate_pdf_report(df, sentiment_counts)
                
                # Show success message
                st.success("Report generated successfully!")
                
                # PDF Download Button
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Full PDF Report",
                        data=f,
                        file_name="sentiment_analysis_report.pdf",
                        mime="application/pdf"
                    )
                
                # Alternative Preview Option
                st.write("**Preview:** The full report is available for download above.")
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
            finally:
                if os.path.exists(report_path):
                    os.remove(report_path)

    # Download CSV
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Results as CSV", csv, "filtered_results.csv", "text/csv")