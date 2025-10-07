import streamlit as st

# Initialize session state for storing test steps
if 'test_steps' not in st.session_state:
    st.session_state.test_steps = [{'test_step': '', 'expected_result': '', 'note': ''}]

# Custom CSS for gray+blue+red color scheme
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Headers */
    h1 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
    }
    
    h2, h3 {
        color: #1e3a8a !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        background-color: #dbeafe !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    /* Specifically target step headers */
    .stMarkdown h2, .stMarkdown h3 {
        color: #1e3a8a !important;
    }
    
    /* Text area labels */
    label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Text areas - light styling that doesn't interfere */
    div[data-baseweb="textarea"] {
        background-color: white;
        border-radius: 8px;
    }
    
    /* Force text area background to be white */
    textarea {
        background-color: white !important;
        color: #1f2937 !important;
    }
    
    /* Divider */
    hr {
        border-color: #cbd5e1 !important;
        margin: 2.5rem 0 !important;
        opacity: 0.5;
    }
    
    /* Button container spacing */
    .stButton {
        margin-top: 1rem;
    }
    
    /* All buttons base style */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
        font-size: 1rem !important;
    }
    
    /* Add Another Step Button - Blue */
    .stButton button:not([kind="primary"]) {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    .stButton button:not([kind="primary"]):hover {
        background-color: #2563eb !important;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Run Test Button - Red */
    .stButton button[kind="primary"] {
        background-color: #dc2626 !important;
        color: white !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #b91c1c !important;
        box-shadow: 0 4px 6px rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #dbeafe !important;
        border-left: 4px solid #3b82f6 !important;
        color: #1e3a8a !important;
        border-radius: 6px !important;
        padding: 1rem !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #e2e8f0 !important;
        border-radius: 6px !important;
        color: #1e40af !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 0 0 6px 6px !important;
    }
    
    /* Column spacing */
    .row-widget.stHorizontal {
        gap: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AgenTest")

# Display all test steps
for i, step in enumerate(st.session_state.test_steps):
    st.subheader(f"Step {i + 1}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_step = st.text_area(
            "Test Step:",
            value=step['test_step'],
            key=f"test_step_{i}",
            height=100,
            placeholder="Enter test step description..."
        )
        st.session_state.test_steps[i]['test_step'] = test_step
    
    with col2:
        expected_result = st.text_area(
            "Expected Result:",
            value=step['expected_result'],
            key=f"expected_result_{i}",
            height=100,
            placeholder="Enter expected result..."
        )
        st.session_state.test_steps[i]['expected_result'] = expected_result
    
    note = st.text_area(
        "Note to LLM (optional):",
        value=step['note'],
        key=f"note_{i}",
        height=80,
        placeholder="Add any special notes for the LLM..."
    )
    st.session_state.test_steps[i]['note'] = note
    
    st.divider()

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Add Another Step", use_container_width=True):
        st.session_state.test_steps.append({'test_step': '', 'expected_result': '', 'note': ''})
        st.rerun()

with col2:
    if st.button("▶️ Run Test", type="primary", use_container_width=True):
        st.success("Test execution started!")
        
        # Display collected test steps
        st.subheader("Test Steps Summary:")
        for i, step in enumerate(st.session_state.test_steps):
            with st.expander(f"Step {i + 1}"):
                st.write(f"**Test Step:** {step['test_step']}")
                st.write(f"**Expected Result:** {step['expected_result']}")
                if step['note']:
                    st.write(f"**Note to LLM:** {step['note']}")