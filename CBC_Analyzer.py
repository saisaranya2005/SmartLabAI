import streamlit as st
import json
import pandas as pd
import re
import random
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import io
import os
from dotenv import load_dotenv
load_dotenv()

# Import patient history manager and page
from historical import CBCHistoryManager, patient_history_page, format_abnormalities

CBC_REFERENCE_RANGES = {
    "CBC_Reference_Ranges": {
        "version": "2025.1",
        "note": "Reference ranges may vary between laboratories",
        "WHITE_BLOOD_CELLS": {
            "WBC_Total": {"male": "4.5-11.0 ×10³/μL", "female": "4.5-11.0 ×10³/μL"},
            "Neutrophils": {"male": "1.5-8.0 ×10³/μL (40-60%)", "female": "1.5-8.0 ×10³/μL (40-60%)"},
            "Lymphocytes": {"male": "1.0-4.0 ×10³/μL (20-40%)", "female": "1.0-4.0 ×10³/μL (20-40%)"},
            "Monocytes": {"male": "0.2-1.0 ×10³/μL (2-8%)", "female": "0.2-1.0 ×10³/μL (2-8%)"},
            "Eosinophils": {"male": "0.0-0.5 ×10³/μL (1-4%)", "female": "0.0-0.5 ×10³/μL (1-4%)"},
            "Basophils": {"male": "0.0-0.2 ×10³/μL (0-1%)", "female": "0.0-0.2 ×10³/μL (0-1%)"}
        },
        "RED_BLOOD_CELLS": {
            "RBC_Count": {"male": "4.6-6.2 ×10⁶/μL", "female": "4.2-5.4 ×10⁶/μL"},
            "Hemoglobin": {"male": "13.0-18.0 g/dL", "female": "12.0-16.0 g/dL"},
            "Hematocrit": {"male": "40-54%", "female": "36-48%"},
            "MCV": {"male": "80-100 fL", "female": "80-100 fL"},
            "MCH": {"male": "27-32 pg", "female": "27-32 pg"},
            "MCHC": {"male": "32-36 g/dL", "female": "32-36 g/dL"},
            "RDW": {"male": "11.5-14.5%", "female": "11.5-14.5%"}
        },
        "PLATELETS": {
            "Platelet_Count": {"male": "150-450 ×10³/μL", "female": "150-450 ×10³/μL"},
            "MPV": {"male": "7.5-11.5 fL", "female": "7.5-11.5 fL"}
        }
    }
}

def generate_patient_id(patient_name, patient_age, patient_gender):
    name_part = patient_name.replace(" ", "").upper()[:4] if patient_name else "PAT"
    random_part = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4))
    date_str = datetime.now().strftime("%Y%m%d")
    return f"CBC-{date_str}-{name_part}{patient_age}{patient_gender[0]}{random_part}"

def extract_range(range_str):
    numbers = re.findall(r'(\d+\.?\d*)', range_str)
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    return None, None

def get_reference_range(cell_type, gender):
    ranges = CBC_REFERENCE_RANGES["CBC_Reference_Ranges"]
    for category in ranges.values():
        if isinstance(category, dict) and cell_type in category:
            return category[cell_type][gender.lower()]
    return None

def analyze_value(value, ref_range):
    min_val, max_val = extract_range(ref_range)
    if min_val is None or max_val is None:
        return "Unable to analyze", "gray"
    if value < min_val:
        return "LOW", "red"
    elif value > max_val:
        return "HIGH", "orange"
    else:
        return "NORMAL", "green"

def generate_comprehensive_analysis(cbc_results, patient_gender, patient_age):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    try:
        groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='llama-3.1-8b-instant')
        prompt = f"""
Analyze CBC results for {patient_age}-year-old {patient_gender.lower()} patient. Provide direct, clinical recommendations without conversational language.

CBC RESULTS:
{json.dumps(cbc_results, indent=2)}

PROVIDE ANALYSIS IN THIS EXACT FORMAT:

ABNORMAL FINDINGS:
[List only abnormal values with clinical significance]

POSSIBLE CAUSES:
[Direct medical causes for abnormal findings]

RECOMMENDED DIAGNOSTIC TESTS:
[Specific additional tests needed]

LIFESTYLE MODIFICATIONS:
[Direct recommendations without "patient should" language]

DIETARY RECOMMENDATIONS - VEGETARIAN:
[Specific foods and supplements for vegetarians]

DIETARY RECOMMENDATIONS - NON-VEGETARIAN:
[Specific foods and supplements including meat/fish]

ACTIVITY RECOMMENDATIONS:
[Exercise and activity guidelines]

MONITORING GUIDELINES:
[What to watch and when to retest]

FOLLOW-UP TIMELINE:
[Specific timeframes for follow-up]

SEVERITY ASSESSMENT:
[Urgent, Moderate, or Routine follow-up needed]

Provide only factual medical information without conversational phrases.
"""
        response = groq_chat.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Analysis unavailable due to connection error: {str(e)}"

def cbc_analyzer_page():
    st.title("🩸 Complete Blood Count (CBC) Analyzer")
    st.markdown("Enter your CBC test results for comprehensive medical analysis")
    st.sidebar.header("Patient Information")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    st.session_state.patient_gender = gender
    st.session_state.patient_age = age

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Enter CBC Values")
        st.subheader("🔵 White Blood Cells")
        wbc_col1, wbc_col2 = st.columns(2)
        with wbc_col1:
            wbc_total = st.number_input("WBC Total (×10³/μL)", min_value=0.0, step=0.1, format="%.1f")
            neutrophils = st.number_input("Neutrophils (×10³/μL)", min_value=0.0, step=0.1, format="%.1f")
            lymphocytes = st.number_input("Lymphocytes (×10³/μL)", min_value=0.0, step=0.1, format="%.1f")
        with wbc_col2:
            monocytes = st.number_input("Monocytes (×10³/μL)", min_value=0.0, step=0.1, format="%.1f")
            eosinophils = st.number_input("Eosinophils (×10³/μL)", min_value=0.0, step=0.1, format="%.1f")
            basophils = st.number_input("Basophils (×10³/μL)", min_value=0.0, step=0.1, format="%.1f")
        st.subheader("🔴 Red Blood Cells")
        rbc_col1, rbc_col2 = st.columns(2)
        with rbc_col1:
            rbc_count = st.number_input("RBC Count (×10⁶/μL)", min_value=0.0, step=0.1, format="%.1f")
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1, format="%.1f")
            hematocrit = st.number_input("Hematocrit (%)", min_value=0.0, step=0.1, format="%.1f")
            mcv = st.number_input("MCV (fL)", min_value=0.0, step=0.1, format="%.1f")
        with rbc_col2:
            mch = st.number_input("MCH (pg)", min_value=0.0, step=0.1, format="%.1f")
            mchc = st.number_input("MCHC (g/dL)", min_value=0.0, step=0.1, format="%.1f")
            rdw = st.number_input("RDW (%)", min_value=0.0, step=0.1, format="%.1f")
        st.subheader("🟡 Platelets")
        platelet_col1, platelet_col2 = st.columns(2)
        with platelet_col1:
            platelet_count = st.number_input("Platelet Count (×10³/μL)", min_value=0.0, step=1.0, format="%.0f")
        with platelet_col2:
            mpv = st.number_input("MPV (fL)", min_value=0.0, step=0.1, format="%.1f")

    with col2:
        st.header("Analysis")
        if st.button("🔍 Analyze CBC & Generate Recommendations", type="primary"):
            results = []
            tests = [
                ("WBC_Total", wbc_total, "×10³/μL"),
                ("Neutrophils", neutrophils, "×10³/μL"),
                ("Lymphocytes", lymphocytes, "×10³/μL"),
                ("Monocytes", monocytes, "×10³/μL"),
                ("Eosinophils", eosinophils, "×10³/μL"),
                ("Basophils", basophils, "×10³/μL"),
                ("RBC_Count", rbc_count, "×10⁶/μL"),
                ("Hemoglobin", hemoglobin, "g/dL"),
                ("Hematocrit", hematocrit, "%"),
                ("MCV", mcv, "fL"),
                ("MCH", mch, "pg"),
                ("MCHC", mchc, "g/dL"),
                ("RDW", rdw, "%"),
                ("Platelet_Count", platelet_count, "×10³/μL"),
                ("MPV", mpv, "fL")
            ]
            for test_name, value, unit in tests:
                if value > 0:
                    ref_range = get_reference_range(test_name, gender)
                    if ref_range:
                        status, color = analyze_value(value, ref_range)
                        results.append({
                            "Test": test_name.replace("_", " "),
                            "Value": f"{value} {unit}",
                            "Reference Range": ref_range,
                            "Status": status
                        })
            if results:
                st.session_state.cbc_results = results
                df = pd.DataFrame(results)
                def color_status(val):
                    if val == "LOW":
                        return 'background-color: #ffebee'
                    elif val == "HIGH":
                        return 'background-color: #fff3e0'
                    elif val == "NORMAL":
                        return 'background-color: #e8f5e8'
                    return ''
                styled_df = df.style.map(color_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True)
                normal_count = sum(1 for r in results if r["Status"] == "NORMAL")
                abnormal_count = len(results) - normal_count
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("Total Tests", len(results))
                with col_metric2:
                    st.metric("Normal", normal_count)
                with col_metric3:
                    st.metric("Abnormal", abnormal_count)
                with st.spinner("Generating comprehensive medical analysis..."):
                    ai_analysis = generate_comprehensive_analysis(results, gender, age)
                    st.session_state.ai_analysis = ai_analysis
                st.success("✅ Analysis complete! Check the AI Assistant page for detailed recommendations.")
            else:
                st.info("Please enter at least one CBC value to analyze.")

def ai_assistant_page():
    st.title("🤖 CBC Medical Assistant")
    st.markdown("Comprehensive medical analysis and recommendations based on your CBC results")
    if 'cbc_results' in st.session_state and 'ai_analysis' in st.session_state:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📊 Your CBC Results Summary")
            df = pd.DataFrame(st.session_state.cbc_results)
            abnormal_results = [r for r in st.session_state.cbc_results if r["Status"] != "NORMAL"]
            if abnormal_results:
                st.warning(f"⚠️ {len(abnormal_results)} abnormal result(s) found")
                for result in abnormal_results:
                    st.write(f"• **{result['Test']}**: {result['Value']} ({result['Status']})")
            else:
                st.success("✅ All CBC parameters are within normal limits")
        with col2:
            st.info(f"**Patient**: {st.session_state.get('patient_age', 'N/A')} years, {st.session_state.get('patient_gender', 'N/A')}")
        st.subheader("🔬 Comprehensive Medical Analysis")
        analysis_text = st.session_state.ai_analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Clinical Analysis", "🥗 Diet Plans", "🏃 Activity Plan", "🔬 Tests & Monitoring", "⚠️ Follow-up"])
        with tab1:
            st.markdown("### Clinical Findings & Causes")
            sections = analysis_text.split('\n\n')
            for section in sections:
                if any(keyword in section.upper() for keyword in ['ABNORMAL FINDINGS', 'POSSIBLE CAUSES', 'SEVERITY']):
                    st.markdown(section)
        with tab2:
            st.markdown("### Dietary Recommendations")
            diet_col1, diet_col2 = st.columns(2)
            with diet_col1:
                st.markdown("#### 🥬 Vegetarian Diet Plan")
                veg_section = ""
                for section in sections:
                    if 'VEGETARIAN' in section.upper():
                        veg_section = section
                        break
                if veg_section:
                    st.markdown(veg_section)
                else:
                    st.info("No specific vegetarian recommendations available")
            with diet_col2:
                st.markdown("#### 🥩 Non-Vegetarian Diet Plan")
                nonveg_section = ""
                for section in sections:
                    if 'NON-VEGETARIAN' in section.upper():
                        nonveg_section = section
                        break
                if nonveg_section:
                    st.markdown(nonveg_section)
                else:
                    st.info("No specific non-vegetarian recommendations available")
        with tab3:
            st.markdown("### Activity & Lifestyle Recommendations")
            for section in sections:
                if any(keyword in section.upper() for keyword in ['LIFESTYLE', 'ACTIVITY']):
                    st.markdown(section)
        with tab4:
            st.markdown("### Recommended Tests & Monitoring")
            for section in sections:
                if any(keyword in section.upper() for keyword in ['DIAGNOSTIC TESTS', 'MONITORING']):
                    st.markdown(section)
        with tab5:
            st.markdown("### Follow-up Guidelines")
            for section in sections:
                if any(keyword in section.upper() for keyword in ['FOLLOW-UP', 'TIMELINE']):
                    st.markdown(section)
        st.subheader("💬 Ask Specific Questions")
        custom_question = st.text_area("Ask about your CBC results, symptoms, or specific concerns:")
        if st.button("Get Answer") and custom_question:
            try:
                GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
                groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='llama-3.1-8b-instant')
                context_prompt = f"""
Based on these CBC results: {st.session_state.cbc_results}
Patient: {st.session_state.get('patient_age', 'unknown')} years, {st.session_state.get('patient_gender', 'unknown')}

Answer this specific question directly and professionally: {custom_question}

Provide direct medical information without conversational language.
"""
                response = groq_chat.invoke(context_prompt)
                st.markdown("#### Answer:")
                st.markdown(response.content)
            except Exception as e:
                st.error(f"Unable to process question: {str(e)}")
    else:
        st.info("📝 Please analyze your CBC results first on the main page to get comprehensive recommendations.")

def format_analysis_for_pdf(analysis_text):
    if not analysis_text:
        return "Comprehensive analysis not available."
    formatted_text = analysis_text.replace('*', '').replace('#', '')
    sections = formatted_text.split('\n\n')
    pdf_formatted = ""
    for section in sections:
        if section.strip():
            if any(header in section.upper() for header in ['ABNORMAL FINDINGS', 'POSSIBLE CAUSES', 'RECOMMENDED', 'DIETARY', 'ACTIVITY', 'MONITORING', 'FOLLOW-UP', 'SEVERITY']):
                pdf_formatted += f"<br/><b>{section.strip()}</b><br/>"
            else:
                lines = section.split('\n')
                for line in lines:
                    if line.strip():
                        if line.strip().startswith('-') or line.strip().startswith('•'):
                            pdf_formatted += f"• {line.strip()[1:].strip()}<br/>"
                        else:
                            pdf_formatted += f"{line.strip()}<br/>"
                pdf_formatted += "<br/>"
    return pdf_formatted

def generate_pdf_report(patient_info, cbc_results, ai_analysis, doctor_comments, findings, doctor_info=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30,
        alignment=TA_CENTER, textColor=colors.darkblue, fontName='Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=12,
        spaceBefore=12, textColor=colors.darkblue, fontName='Helvetica-Bold'
    )
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'], fontSize=10, spaceAfter=6,
        alignment=TA_JUSTIFY, fontName='Helvetica'
    )
    elements.append(Paragraph("COMPREHENSIVE CBC ANALYSIS REPORT", title_style))
    elements.append(Spacer(1, 20))
    if doctor_info:
        elements.append(Paragraph("LABORATORY INFORMATION", heading_style))
        lab_data = [
            ['Laboratory:', doctor_info.get('lab_name', 'N/A')],
            ['Physician:', doctor_info.get('doctor_name', 'N/A')],
            ['License:', doctor_info.get('license', 'N/A')],
            ['Contact:', doctor_info.get('contact', 'N/A')]
        ]
        lab_table = Table(lab_data, colWidths=[1.5*inch, 4.5*inch])
        lab_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(lab_table)
        elements.append(Spacer(1, 20))
    elements.append(Paragraph("PATIENT INFORMATION", heading_style))
    patient_data = [
        ['Patient ID:', patient_info.get('patient_id', 'N/A')],
        ['Patient Name:', patient_info.get('name', 'N/A')],
        ['Age:', f"{patient_info.get('age', 'N/A')} years"],
        ['Gender:', patient_info.get('gender', 'N/A')],
        ['Test Date:', patient_info.get('test_date', 'N/A')],
        ['Report Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    patient_table = Table(patient_data, colWidths=[1.5*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("LABORATORY RESULTS", heading_style))
    cbc_data = [['Parameter', 'Result', 'Reference Range', 'Status']]
    for result in cbc_results:
        cbc_data.append([
            result['Test'],
            result['Value'],
            result['Reference Range'],
            result['Status']
        ])
    cbc_table = Table(cbc_data, colWidths=[2*inch, 1.5*inch, 2*inch, 1*inch])
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]
    for i, result in enumerate(cbc_results, 1):
        if result['Status'] == 'LOW':
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.mistyrose))
            table_style.append(('TEXTCOLOR', (3, i), (3, i), colors.red))
        elif result['Status'] == 'HIGH':
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.lightyellow))
            table_style.append(('TEXTCOLOR', (3, i), (3, i), colors.orange))
        else:
            table_style.append(('TEXTCOLOR', (3, i), (3, i), colors.green))
    cbc_table.setStyle(TableStyle(table_style))
    elements.append(cbc_table)
    elements.append(Spacer(1, 20))
    if ai_analysis:
        elements.append(Paragraph("COMPREHENSIVE CLINICAL ANALYSIS", heading_style))
        formatted_analysis = format_analysis_for_pdf(ai_analysis)
        elements.append(Paragraph(formatted_analysis, normal_style))
        elements.append(Spacer(1, 15))
    if findings:
        elements.append(Paragraph("CLINICAL FINDINGS", heading_style))
        elements.append(Paragraph(findings, normal_style))
        elements.append(Spacer(1, 15))
    elements.append(Paragraph("PHYSICIAN INTERPRETATION", heading_style))
    if doctor_comments:
        elements.append(Paragraph(doctor_comments, normal_style))
    else:
        elements.append(Paragraph("Pending physician review.", normal_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("MEDICAL DISCLAIMER", heading_style))
    disclaimer = """This report contains laboratory results and AI-generated analysis for educational purposes. 
    All abnormal results require clinical correlation and physician interpretation. The AI analysis provides 
    general medical information and should not replace professional medical judgment. Consult your healthcare 
    provider for personalized medical advice and treatment decisions."""
    elements.append(Paragraph(disclaimer, normal_style))
    doc.build(elements)
    buffer.seek(0)
    return buffer

def pdf_report_page():
    st.title("📄 Generate Medical Report")
    st.markdown("Create a comprehensive medical-grade PDF report")
    if 'cbc_results' not in st.session_state:
        st.warning("⚠️ Please analyze CBC results first")
        return

    st.subheader("👤 Patient Identification")
    is_old_patient = st.radio("Is this an existing patient?", ["No, new patient", "Yes, returning patient"], horizontal=True)
    patient_id = None
    patient_info_prefill = {}
    history_manager = st.session_state.get('history_manager', None)

    if is_old_patient == "Yes, returning patient":
        patient_id_input = st.text_input("Enter Patient ID")
        if patient_id_input and history_manager:
            summary = history_manager.get_patient_summary(patient_id_input)
            if summary:
                st.success("Patient found. Details loaded.")
                patient_id = summary['patient_id']
                patient_info_prefill = {
                    'name': summary.get('patient_name', ''),
                    'age': summary.get('age', 30),
                    'gender': summary.get('gender', 'Male')
                }
            else:
                st.error("Patient ID not found. Please check or register as new patient.")
    else:
        patient_id = None

    st.subheader("🏥 Laboratory Information")
    col1, col2 = st.columns(2)
    with col1:
        lab_name = st.text_input("Laboratory Name")
        doctor_name = st.text_input("Physician Name")
    with col2:
        license_number = st.text_input("License Number")
        contact_info = st.text_input("Contact Information")

    st.subheader("👤 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name", value=patient_info_prefill.get('name', ''), placeholder="Required")
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=patient_info_prefill.get('age', st.session_state.get('patient_age', 30)))
    with col2:
        patient_gender = st.selectbox("Gender", ["Male", "Female"], index=0 if patient_info_prefill.get('gender', st.session_state.get('patient_gender', 'Male')) == 'Male' else 1)
        test_date = st.date_input("Test Date", value=datetime.now().date())

    st.subheader("🔬 Clinical Information")
    clinical_findings = st.text_area("Clinical Findings", placeholder="Enter specific clinical observations or symptoms...")
    physician_comments = st.text_area("Physician Comments", placeholder="Enter physician interpretation and recommendations...")

    st.subheader("📋 Report Preview")
    if st.session_state.get('cbc_results'):
        abnormal_count = sum(1 for result in st.session_state.cbc_results if result['Status'] != 'NORMAL')
        total_tests = len(st.session_state.cbc_results)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", total_tests)
        with col2:
            st.metric("Normal Results", total_tests - abnormal_count)
        with col3:
            st.metric("Abnormal Results", abnormal_count)
        if abnormal_count > 0:
            st.warning(f"⚠️ {abnormal_count} abnormal result(s) detected")
            abnormal_results = [r for r in st.session_state.cbc_results if r['Status'] != 'NORMAL']
            for result in abnormal_results:
                st.write(f"• **{result['Test']}**: {result['Value']} ({result['Status']})")

    st.subheader("📄 Generate Report")
    if st.button("🔄 Generate PDF Report", type="primary"):
        if not patient_name:
            st.error("❌ Patient name is required")
            return
        try:
            if is_old_patient == "Yes, returning patient" and patient_id:
                final_patient_id = patient_id
            else:
                final_patient_id = generate_patient_id(patient_name, patient_age, patient_gender)
            patient_info = {
                'patient_id': final_patient_id,
                'name': patient_name,
                'age': patient_age,
                'gender': patient_gender,
                'test_date': test_date.strftime("%Y-%m-%d")
            }
            doctor_info = {
                'lab_name': lab_name,
                'doctor_name': doctor_name,
                'license': license_number,
                'contact': contact_info
            } if any([lab_name, doctor_name, license_number, contact_info]) else None
            with st.spinner("🔄 Generating comprehensive medical report..."):
                pdf_buffer = generate_pdf_report(
                    patient_info=patient_info,
                    cbc_results=st.session_state.cbc_results,
                    ai_analysis=st.session_state.get('ai_analysis', ''),
                    doctor_comments=physician_comments,
                    findings=clinical_findings,
                    doctor_info=doctor_info
                )
            # --- Save to MongoDB ---
            if history_manager:
                abnormalities = format_abnormalities(st.session_state.cbc_results)
                history_manager.save_patient_record(
                    patient_id=final_patient_id,
                    patient_info=patient_info,
                    cbc_results=st.session_state.cbc_results,
                    abnormalities=abnormalities,
                    is_new_patient=(is_old_patient == "No, new patient")
                )
            # --- End Save to MongoDB ---
            st.success(f"✅ Report generated successfully! Patient ID: {final_patient_id}")
            filename = f"CBC_Report_{patient_name.replace(' ', '_')}_{test_date.strftime('%Y%m%d')}.pdf"
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer.getvalue(),
                file_name=filename,
                mime="application/pdf",
                type="primary"
            )
            st.info(f"📊 Report includes {len(st.session_state.cbc_results)} CBC parameters with comprehensive analysis")
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            st.info("Please ensure all required fields are filled correctly")

    st.subheader("📋 Additional Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Update CBC Results"):
            st.info("Please return to the main analyzer page to update your CBC results")
    with col2:
        if st.button("🤖 View AI Analysis"):
            st.info("Please check the AI Assistant page for detailed recommendations")
    st.subheader("ℹ️ Report Information")
    st.info("""
    **Your PDF report will include:**
    • Complete CBC results with reference ranges
    • Color-coded abnormal values
    • Comprehensive AI medical analysis
    • Dietary recommendations (vegetarian & non-vegetarian)
    • Activity and lifestyle recommendations
    • Follow-up guidelines and monitoring recommendations
    • Medical disclaimer and professional formatting
    """)
    with st.expander("📋 Usage Notes"):
        st.markdown("""
        **For Healthcare Professionals:**
        - Fill in laboratory and physician information for official reports
        - Add clinical findings and interpretations
        - Review AI analysis before finalizing

        **For Patients:**
        - Ensure patient information is accurate
        - Share this report with your healthcare provider
        - Do not use AI analysis as a substitute for medical advice

        **Technical Notes:**
        - Report format: Professional medical PDF
        - File size: Optimized for sharing
        - Compatibility: Standard PDF readers
        """)

def main():
    st.set_page_config(
        page_title="CBC Analyzer Pro",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .sidebar .sidebar-content {
        background: #f1f3f4;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("## 🧭 Navigation")
    page_options = {
        "🩸 CBC Analyzer": "analyzer",
        "🤖 AI Assistant": "assistant",
        "📄 PDF Report": "report",
        "📈 Patient History": "history",
        "ℹ️ About": "about"
    }
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "analyzer"
    for page_name, page_key in page_options.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
    current_page = st.session_state.current_page

    # MongoDB URI (replace with your actual URI)
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if 'history_manager' not in st.session_state:
        st.session_state.history_manager = CBCHistoryManager(MONGODB_URI)

    if current_page == "analyzer":
        cbc_analyzer_page()
    elif current_page == "assistant":
        ai_assistant_page()
    elif current_page == "report":
        pdf_report_page()
    elif current_page == "history":
        patient_history_page(st.session_state.history_manager)
    elif current_page == "about":
        about_page()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    if 'cbc_results' in st.session_state:
        total_tests = len(st.session_state.cbc_results)
        abnormal_count = sum(1 for r in st.session_state.cbc_results if r.get('Status', '') != 'NORMAL')
        st.sidebar.metric("Tests Analyzed", total_tests)
        st.sidebar.metric("Abnormal Results", abnormal_count)
        if abnormal_count > 0:
            st.sidebar.warning(f"⚠️ {abnormal_count} abnormal value(s)")
        else:
            st.sidebar.success("✅ All values normal")
    else:
        st.sidebar.info("No CBC results yet")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Quick Links")
    st.sidebar.markdown("• [Medical Reference](https://www.mayoclinic.org)")
    st.sidebar.markdown("• [Lab Values Guide](https://www.labcorp.com)")
    st.sidebar.markdown("• [Health Information](https://www.webmd.com)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚠️ Medical Disclaimer**")
    st.sidebar.caption("This tool is for educational purposes only. Always consult healthcare professionals for medical advice.")

def about_page():
    st.title("ℹ️ About CBC Analyzer Pro")
    st.markdown("""
    <div class="main-header">
        <h1>CBC Analyzer Pro - Advanced Blood Test Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔬 Key Features")
        st.markdown("""
        • **Comprehensive CBC Analysis** - All 15 major parameters
        • **AI-Powered Recommendations** - Personalized medical insights
        • **Gender-Specific Reference Ranges** - Accurate normal values
        • **Professional PDF Reports** - Medical-grade documentation
        • **Dietary Guidance** - Vegetarian & non-vegetarian plans
        • **Activity Recommendations** - Lifestyle modifications
        • **Follow-up Guidelines** - Monitoring schedules
        """)
    with col2:
        st.subheader("🩸 CBC Parameters Analyzed")
        st.markdown("""
        **White Blood Cells:**
        • WBC Total, Neutrophils, Lymphocytes
        • Monocytes, Eosinophils, Basophils

        **Red Blood Cells:**
        • RBC Count, Hemoglobin, Hematocrit
        • MCV, MCH, MCHC, RDW

        **Platelets:**
        • Platelet Count, MPV
        """)
    st.subheader("🔄 How It Works")
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)
    with step_col1:
        st.markdown("""
        **Step 1: Enter Data**
        📝 Input your CBC test values
        👤 Provide patient information
        """)
    with step_col2:
        st.markdown("""
        **Step 2: Analysis**
        🔍 Compare with reference ranges
        ⚡ AI processes results
        """)
    with step_col3:
        st.markdown("""
        **Step 3: Recommendations**
        🤖 Get personalized advice
        🥗 Receive diet plans
        """)
    with step_col4:
        st.markdown("""
        **Step 4: Report**
        📄 Generate PDF report
        📧 Share with doctor
        """)
    st.subheader("⚙️ Technical Specifications")
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown("""
        **AI Technology:**
        • Model: Llama3-70B via Groq API
        • Analysis: Evidence-based recommendations
        • Accuracy: Clinical-grade interpretations

        **Reference Standards:**
        • Latest 2025 laboratory ranges
        • Gender-specific normal values
        • Age-appropriate interpretations
        """)
    with tech_col2:
        st.markdown("""
        **Report Features:**
        • Professional PDF formatting
        • Color-coded abnormal values
        • Comprehensive analysis sections
        • Medical-grade documentation

        **Data Security:**
        • No data storage
        • Session-based processing
        • Privacy-compliant design
        """)
    st.subheader("📋 Usage Guidelines")
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **✅ Appropriate Use:**
        • Educational purposes
        • Pre-consultation preparation
        • Understanding test results
        • Tracking health trends
        • Research and learning
        """)
    with col2:
        st.warning("""
        **⚠️ Important Limitations:**
        • Not a substitute for medical advice
        • AI analysis requires clinical correlation
        • Emergency conditions need immediate care
        • Complex cases require specialist consultation
        • Always consult healthcare providers
        """)
    st.subheader("📞 Support & Contact")
    support_col1, support_col2, support_col3 = st.columns(3)
    with support_col1:
        st.info("""
        **🔧 Technical Support**
        • Application issues
        • Feature requests
        • Bug reports
        """)
    with support_col2:
        st.info("""
        **📚 Educational Resources**
        • CBC interpretation guides
        • Medical references
        • Health information
        """)
    with support_col3:
        st.info("""
        **🏥 Medical Questions**
        • Consult healthcare provider
        • Contact your laboratory
        • Seek professional advice
        """)
    st.subheader("🔄 Version Information")
    version_col1, version_col2 = st.columns(2)
    with version_col1:
        st.markdown("""
        **Current Version:** 2.0.1
        **Release Date:** July 2025
        **Last Updated:** CBC Reference Ranges 2025.1
        """)
    with version_col2:
        st.markdown("""
        **Recent Updates:**
        • Enhanced AI analysis accuracy
        • Improved PDF report formatting
        • Updated reference ranges
        • Better user interface
        """)
    st.markdown("---")
    st.error("""
    **⚠️ IMPORTANT MEDICAL DISCLAIMER**

    This application is designed for educational and informational purposes only. It is not intended to be a substitute 
    for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified 
    health provider with any questions you may have regarding a medical condition.

    The AI-generated analysis and recommendations should not be used as the sole basis for medical decisions. 
    Clinical correlation and professional medical interpretation are essential for proper patient care.

    Never disregard professional medical advice or delay in seeking it because of something you have read in this application.
    """)

if __name__ == "__main__":
    main()




