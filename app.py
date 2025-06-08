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
from fpdf import FPDF

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

# Parse JSONL File
def parse_jsonl(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    with jsonlines.open(tmp_path) as reader:
        lines = [obj['text'] for obj in reader if 'text' in obj]
    os.remove(tmp_path)
    return "\n".join(lines)

# Parse Audio File
def parse_audio(file):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    with sr.AudioFile(tmp_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as e:
            text = f"Error with speech recognition: {e}"
    os.remove(tmp_path)
    return text

# Parse Image File
def parse_image(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    image = Image.open(tmp_path)
    text = pytesseract.image_to_string(image)
    os.remove(tmp_path)
    return text

# Generate PDF Report
def generate_pdf_report(df, sentiment_counts, filename="sentiment_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Feedback Sentiment Analysis Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
    pdf.ln(10)
    
    # Summary Statistics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Summary Statistics", ln=1)
    pdf.set_font("Arial", size=12)
    
    total_feedback = len(df)
    positive = len(df[df['Sentiment'] == 'Positive'])
    negative = len(df[df['Sentiment'] == 'Negative'])
    neutral = len(df[df['Sentiment'] == 'Neutral'])
    
    pdf.cell(200, 10, txt=f"Total Items Analyzed: {total_feedback}", ln=1)
    pdf.cell(200, 10, txt=f"Positive Feedback: {positive} ({positive/total_feedback:.1%})", ln=1)
    pdf.cell(200, 10, txt=f"Negative Feedback: {negative} ({negative/total_feedback:.1%})", ln=1)
    pdf.cell(200, 10, txt=f"Neutral Feedback: {neutral} ({neutral/total_feedback:.1%})", ln=1)
    pdf.ln(10)
    
    # Sample Feedback
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Sample Feedback", ln=1)
    pdf.set_font("Arial", size=10)
    
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        samples = df[df['Sentiment'] == sentiment].head(3)
        if not samples.empty:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt=f"{sentiment} Feedback Examples:", ln=1)
            pdf.set_font("Arial", size=10)
            for _, row in samples.iterrows():
                pdf.multi_cell(0, 10, txt=f"- {row['Original Text'][:150]}...")
            pdf.ln(5)
    
    # Save the PDF
    pdf.output(filename)
    return filename

# Create download link for PDF
def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download Report</a>'

# Streamlit App Setup
st.set_page_config(page_title="Structuring", layout="wide")
st.title("🧠 IntelliStruct: An Automated Tool for Data Structuring")
st.write("Upload text, audio, image, or JSONL file to analyze and generate reports.")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "jsonl", "wav", "png", "jpg", "jpeg"])

if uploaded_file:
    # File Parsing
    if uploaded_file.type == "text/plain":
        raw_text = parse_text(uploaded_file)
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
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    df = pd.DataFrame(lines, columns=["Original Text"])
    df["Original Text"] = df["Original Text"].apply(lambda x: x[:512])
    df["Sentiment"] = df["Original Text"].apply(analyze_sentiment)

    # Sentiment Filter
    sentiment_options = ["Positive", "Negative", "Neutral"]
    selected_sentiments = st.multiselect("Filters :", sentiment_options, default=sentiment_options)
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
            report_path = generate_pdf_report(df, sentiment_counts)
            with open(report_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.markdown(create_download_link(f.read(), "sentiment_analysis_report.pdf"), unsafe_allow_html=True)
            os.remove(report_path)

    # Download CSV
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Results as CSV", csv, "filtered_results.csv", "text/csv")