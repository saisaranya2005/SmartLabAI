import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import string
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import certifi

class LFTHistoryManager:
    def __init__(self, mongodb_uri, db_name="lft_analyzer", collection_name="patient_records"):
        """Initialize MongoDB connection"""
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connect()
    
    def connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(
                self.mongodb_uri,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000
            )
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.collection.create_index("patient_id", unique=False)
            self.collection.create_index("test_date")
            return True
        except ConnectionFailure as e:
            st.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            return False
        except Exception as e:
            st.error(f"Database connection error: {e}")
            self.client = None
            return False

    def generate_patient_id(self, patient_name=None, patient_age=None, patient_gender=None):
        """Generate a unique patient ID. If info is provided, use app.py logic, else fallback to random."""
        if patient_name and patient_age and patient_gender:
            name_part = patient_name.replace(" ", "").upper()[:4]
            random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            date_str = datetime.now().strftime("%Y%m%d")
            patient_id = f"LFT-{date_str}-{name_part}{patient_age}{patient_gender[0]}{random_part}"
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            patient_id = f"LFT-{date_str}-{random_part}"
        # Ensure uniqueness
        while self.check_patient_id_exists(patient_id):
            random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            patient_id = f"LFT-{date_str}-{random_part}"
        return patient_id

    def check_patient_id_exists(self, patient_id):
        """Check if patient ID already exists"""
        if not self.client:
            return False
        try:
            return self.collection.find_one({"patient_id": patient_id}) is not None
        except Exception as e:
            st.error(f"Error checking patient ID: {e}")
            return False

    def save_patient_record(self, patient_id, patient_info, lft_results, abnormalities, is_new_patient=False):
        """Save patient record with unique ID"""
        if not self.client:
            st.error("Database connection not available")
            return False
        try:
            # Ensure test_date is a datetime object
            test_date = patient_info.get('test_date')
            if isinstance(test_date, str):
                test_date = datetime.strptime(test_date, "%Y-%m-%d")
            elif isinstance(test_date, datetime):
                pass
            else:
                test_date = datetime.now()
            record = {
                "patient_id": patient_id,
                "patient_name": patient_info.get('name', ''),
                "age": patient_info.get('age', 0),
                "gender": patient_info.get('gender', ''),
                "test_date": test_date,
                "lft_results": lft_results,
                "abnormalities": abnormalities,
                "created_at": datetime.now(),
                "is_new_patient": is_new_patient
            }
            result = self.collection.insert_one(record)
            return result.inserted_id is not None
        except Exception as e:
            st.error(f"Error saving patient record: {e}")
            return False

    def get_patient_history(self, patient_id):
        """Get all records for a patient ID"""
        if not self.client:
            return []
        try:
            records = list(self.collection.find(
                {"patient_id": patient_id}
            ).sort("test_date", 1))
            for record in records:
                record['_id'] = str(record['_id'])
                # Ensure test_date is datetime
                if isinstance(record['test_date'], str):
                    record['test_date'] = datetime.strptime(record['test_date'], "%Y-%m-%d")
            return records
        except Exception as e:
            st.error(f"Error retrieving patient history: {e}")
            return []

    def get_patient_summary(self, patient_id):
        """Get patient summary statistics"""
        if not self.client:
            return {}
        try:
            records = self.get_patient_history(patient_id)
            if not records:
                return {}
            total_visits = len(records)
            total_abnormalities = sum(len(record.get('abnormalities', [])) for record in records)
            first_visit = min(record['test_date'] for record in records)
            last_visit = max(record['test_date'] for record in records)
            latest_record = max(records, key=lambda x: x['test_date'])
            return {
                'patient_id': patient_id,
                'patient_name': latest_record.get('patient_name', ''),
                'age': latest_record.get('age', 0),
                'gender': latest_record.get('gender', ''),
                'total_visits': total_visits,
                'total_abnormalities': total_abnormalities,
                'first_visit': first_visit,
                'last_visit': last_visit
            }
        except Exception as e:
            st.error(f"Error getting patient summary: {e}")
            return {}

    def search_patients(self, query=""):
        """Search patients by ID or name"""
        if not self.client:
            return []
        try:
            if query:
                search_filter = {
                    "$or": [
                        {"patient_id": {"$regex": query, "$options": "i"}},
                        {"patient_name": {"$regex": query, "$options": "i"}}
                    ]
                }
            else:
                search_filter = {}
            pipeline = [
                {"$match": search_filter},
                {"$sort": {"test_date": -1}},
                {"$group": {
                    "_id": "$patient_id",
                    "latest_record": {"$first": "$$ROOT"}
                }},
                {"$replaceRoot": {"newRoot": "$latest_record"}},
                {"$sort": {"test_date": -1}}
            ]
            results = list(self.collection.aggregate(pipeline))
            for record in results:
                record['_id'] = str(record['_id'])
                # Ensure test_date is datetime
                if isinstance(record['test_date'], str):
                    record['test_date'] = datetime.strptime(record['test_date'], "%Y-%m-%d")
            return results
        except Exception as e:
            st.error(f"Error searching patients: {e}")
            return []

def format_abnormalities(lft_results):
    """Format abnormal results for storage"""
    abnormalities = []
    for result in lft_results:
        if result.get('Status') != 'NORMAL':
            abnormalities.append({
                'parameter': result['Test'],
                'value': result['Value'],
                'status': result['Status'],
                'reference_range': result['Reference Range']
            })
    return abnormalities

def create_trend_chart(history_data, parameter):
    """Create trend chart for specific parameter"""
    dates = []
    values = []
    statuses = []
    for record in history_data:
        test_date = record['test_date']
        for result in record.get('lft_results', []):
            if result['Test'].replace(' ', '_').upper() == parameter.replace(' ', '_').upper():
                dates.append(test_date)
                value_str = result['Value'].split()[0]
                try:
                    values.append(float(value_str))
                    statuses.append(result.get('Status', 'NORMAL'))
                except ValueError:
                    continue
                break
    if not values:
        return None
    colors_map = ['red' if s == 'LOW' else 'orange' if s == 'HIGH' else 'green' for s in statuses]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        marker=dict(color=colors_map, size=8),
        line=dict(color='blue', width=2),
        name=parameter
    ))
    fig.update_layout(
        title=f'{parameter} Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Value',
        height=400,
        showlegend=False
    )
    return fig

def patient_history_page(history_manager):
    """Patient History Management Page"""
    st.title("📈 Patient History & Trends")
    st.markdown("Track LFT results over time and analyze patient trends")
    if not history_manager or not history_manager.client:
        st.error("❌ Database connection not available")
        st.info("Patient history features require database connectivity")
        return
    st.subheader("🔍 Search Patients")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Search by Patient ID or Name",
            placeholder="Enter Patient ID (e.g., LFT-20250702-A1B2) or Patient Name",
            value=st.session_state.get('search_patient', '')
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    if search_query or search_button:
        with st.spinner("Searching patients..."):
            patients = history_manager.search_patients(search_query)
        if patients:
            st.success(f"Found {len(patients)} patient(s)")
            patient_data = []
            for patient in patients:
                summary = history_manager.get_patient_summary(patient['patient_id'])
                patient_data.append({
                    'Patient ID': patient['patient_id'],
                    'Name': patient.get('patient_name', 'N/A'),
                    'Age': patient.get('age', 'N/A'),
                    'Gender': patient.get('gender', 'N/A'),
                    'Total Visits': summary.get('total_visits', 0),
                    'Last Visit': patient['test_date'].strftime('%Y-%m-%d'),
                    'Abnormalities': summary.get('total_abnormalities', 0)
                })
            df = pd.DataFrame(patient_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.subheader("📋 Select Patient for Detailed Analysis")
            patient_options = {f"{p['patient_id']} - {p.get('patient_name', 'Unknown')}": p['patient_id'] 
                             for p in patients}
            if patient_options:
                selected_patient_display = st.selectbox(
                    "Choose Patient:",
                    options=list(patient_options.keys())
                )
                selected_patient_id = patient_options[selected_patient_display]
                display_patient_history(history_manager, selected_patient_id)
        else:
            st.info("No patients found matching your search criteria")
    st.subheader("🕒 Recent Patients")
    recent_patients = history_manager.search_patients()[:5]
    if recent_patients:
        for patient in recent_patients:
            with st.expander(f"🏥 {patient['patient_id']} - {patient.get('patient_name', 'Unknown')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Age:** {patient.get('age', 'N/A')}")
                    st.write(f"**Gender:** {patient.get('gender', 'N/A')}")
                with col2:
                    st.write(f"**Last Visit:** {patient['test_date'].strftime('%Y-%m-%d')}")
                    summary = history_manager.get_patient_summary(patient['patient_id'])
                    st.write(f"**Total Visits:** {summary.get('total_visits', 0)}")
                with col3:
                    if st.button(f"View History", key=f"view_{patient['patient_id']}"):
                        st.session_state.selected_patient_id = patient['patient_id']
                        display_patient_history(history_manager, patient['patient_id'], use_expander_for_visits=False)

def show_visit_details(record):
    results_df = pd.DataFrame(record.get('lft_results', []))
    if not results_df.empty:
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    if record.get('abnormalities'):
        st.markdown("**Abnormal Results:**")
        for abnorm in record['abnormalities']:
            status_color = "🔴" if abnorm['status'] == 'LOW' else "🟠" if abnorm['status'] == 'HIGH' else "🟢"
            st.write(f"{status_color} **{abnorm['parameter']}**: {abnorm['value']} ({abnorm['status']})")

def display_patient_history(history_manager, patient_id, use_expander_for_visits=True):
    """Display detailed patient history"""
    st.markdown("---")
    st.subheader(f"📊 Patient History: {patient_id}")
    history_data = history_manager.get_patient_history(patient_id)
    if not history_data:
        st.warning("No history found for this patient")
        return
    summary = history_manager.get_patient_summary(patient_id)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Visits", summary.get('total_visits', 0))
    with col2:
        st.metric("Total Abnormalities", summary.get('total_abnormalities', 0))
    with col3:
        if summary.get('first_visit'):
            days_tracked = (summary['last_visit'] - summary['first_visit']).days
            st.metric("Days Tracked", days_tracked)
    with col4:
        if summary.get('last_visit'):
            st.metric("Last Visit", summary['last_visit'].strftime('%Y-%m-%d'))
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📋 Visit History", "⚠️ Abnormalities", "📊 Comparison"])
    with tab1:
        st.subheader("Parameter Trends")
        all_parameters = set()
        for record in history_data:
            for result in record.get('lft_results', []):
                all_parameters.add(result['Test'])
        selected_params = st.multiselect(
            "Select parameters to analyze:",
            sorted(list(all_parameters)),
            default=list(sorted(all_parameters))[:3] if all_parameters else [],
            key=f"multiselect_{patient_id}"
        )
        for param in selected_params:
            chart = create_trend_chart(history_data, param)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
    with tab2:
        st.subheader("Visit History")
        for i, record in enumerate(reversed(history_data)):
            visit_num = len(history_data) - i
            test_date = record['test_date'].strftime('%Y-%m-%d')
            abnormal_count = len(record.get('abnormalities', []))
            if use_expander_for_visits:
                with st.expander(f"Visit #{visit_num} - {test_date} ({abnormal_count} abnormalities)"):
                    show_visit_details(record)
            else:
                st.markdown(f"**Visit #{visit_num} - {test_date} ({abnormal_count} abnormalities)**")
                show_visit_details(record)
    with tab3:
        st.subheader("Abnormality Analysis")
        all_abnormalities = []
        for record in history_data:
            for abnorm in record.get('abnormalities', []):
                all_abnormalities.append({
                    'Date': record['test_date'].strftime('%Y-%m-%d'),
                    'Parameter': abnorm['parameter'],
                    'Value': abnorm['value'],
                    'Status': abnorm['status']
                })
        if all_abnormalities:
            abnorm_df = pd.DataFrame(all_abnormalities)
            param_counts = abnorm_df['Parameter'].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Most Frequent Abnormalities:**")
                fig_freq = px.bar(
                    x=param_counts.values,
                    y=param_counts.index,
                    orientation='h',
                    title="Abnormality Frequency"
                )
                st.plotly_chart(fig_freq, use_container_width=True)
            with col2:
                st.markdown("**Abnormality Timeline:**")
                st.dataframe(abnorm_df, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 No abnormalities found in patient history!")
    with tab4:
        st.subheader("Visit Comparison")
        if len(history_data) >= 2:
            visit_options = {f"Visit {i+1} ({record['test_date'].strftime('%Y-%m-%d')})": i 
                           for i, record in enumerate(history_data)}
            col1, col2 = st.columns(2)
            with col1:
                visit1_idx = st.selectbox("Select first visit:", list(visit_options.keys()), key=f"visit1_{patient_id}")
                visit1_data = history_data[visit_options[visit1_idx]]
            with col2:
                visit2_idx = st.selectbox("Select second visit:", list(visit_options.keys()), key=f"visit2_{patient_id}")
                visit2_data = history_data[visit_options[visit2_idx]]
            st.markdown("**Comparison Results:**")
            comparison_data = []
            params1 = {r['Test']: r for r in visit1_data.get('lft_results', [])}
            params2 = {r['Test']: r for r in visit2_data.get('lft_results', [])}
            all_params = set(params1.keys()) | set(params2.keys())
            for param in sorted(all_params):
                row = {'Parameter': param}
                if param in params1:
                    row['Visit 1 Value'] = params1[param]['Value']
                    row['Visit 1 Status'] = params1[param]['Status']
                else:
                    row['Visit 1 Value'] = 'N/A'
                    row['Visit 1 Status'] = 'N/A'
                if param in params2:
                    row['Visit 2 Value'] = params2[param]['Value']
                    row['Visit 2 Status'] = params2[param]['Status']
                else:
                    row['Visit 2 Value'] = 'N/A'
                    row['Visit 2 Status'] = 'N/A'
                if param in params1 and param in params2:
                    try:
                        val1 = float(params1[param]['Value'].split()[0])
                        val2 = float(params2[param]['Value'].split()[0])
                        change = ((val2 - val1) / val1) * 100
                        row['Change %'] = f"{change:.1f}%"
                        if abs(change) > 10:
                            row['Trend'] = "📈" if change > 0 else "📉"
                        else:
                            row['Trend'] = "➡️"
                    except:
                        row['Change %'] = 'N/A'
                        row['Trend'] = 'N/A'
                else:
                    row['Change %'] = 'N/A'
                    row['Trend'] = 'N/A'
                comparison_data.append(row)
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("Need at least 2 visits for comparison analysis")