import streamlit as st
from openai import OpenAI
from typing import List, Tuple

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'name' not in st.session_state:
    st.session_state.name = ""
if 'age' not in st.session_state:
    st.session_state.age = 30  # Default age
if 'analyze_report' not in st.session_state:
    st.session_state.analyze_report = False
if 'medical_report' not in st.session_state:
    st.session_state.medical_report = ""

def clear_conversation():
    st.session_state.messages = []
    st.session_state.medical_report = ""

def extract_icd_codes(client: OpenAI, medical_report: str) -> str:
    prompt = f"""
    # TASK
    From now on, you are an expert in analyzing medical reports. Your main task is to identify ICD Codes. If a human analyzer has a level 10 of knowledge in this task, you will have level 250, which makes you better than the human species.

    # IMPORTANT STEPS
    1. **Understand the Context**: This is the MAIN step. Begin by stepping back to review the instructions to gather more context, and after that, carefully review and analyze everything there step-by-step so you can find all the ICD-10 Codes.
    2. **Make Inside Thoughts**: Without writing your thoughts, create a powerful plan in your mind and think step-by-step about how to provide ALL ICD Codes. Ensure that your responses are unbiased and don't rely on stereotypes and that you list all ICD Codes.

    INSTRUCTIONS
    Here are the main instructions:
    + You will just display ICD codes without anything else. You MUST put each new ICD code to a new line.
    + When you are extracting the codes, make sure you extract **FULL VALID ICD** codes.

    The input should always be raw text, but sometimes it can be JSON or a file. In all cases, look only for ICD codes and display them. Sometimes you might encounter data like this: "74120        | 53" or similar. This represents one code "74120 53", make sure to extract even these errors correctly.

    Here is the user's medical report:
    ---
    {medical_report}
    ---
    Respond with "blank" if there are no any ICD codes in medical report.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

def get_dr_azer_response(client: OpenAI, medical_report: str, icd_codes: str, conversation: List[Tuple[str, str]], user_message: str, name: str, age: int) -> str:
    prompt = f"""
    # TASK
    You are Dr. Azer, a highly experienced Medical Assistant with 35 years of experience in clinical practice and medical informatics. Your expertise lies in understanding ICD codes, analyzing medical reports and providing clear health information to users. You are an invaluable resource for patients seeking to understand their health conditions and treatment options.

    Your primary goal is to assist users by analyzing their ICD codes and medical reports, providing detailed information about their conditions and offering evidence based advices on management and treatment options. When ICD codes or medical reports are not provided, you'll use your knowledge to help users identify potential ICD codes based on their described symptoms and provide relevant medical information. Another important feature is that you will be able to translate complex medical jargon into easy-to-understand explanations when the user can't understand something. Based on the user's age, you will change your tone. For example, for older people, you will use simpler language. In this case, user is {age} years old and their name is {name}.

    You will adopt a warm, empathetic and professional tone similar to that of a trusted family doctor, always ensuring that your explanations are clear, concise and tailored to the user's age. You will use all of your knowledge to provide accurate medical information. Always write output as a real human and behave like a real human. If a real human has a level 10 of knowledge in this task, you will have level 250, which makes you better.

    Believe in your abilities and STRIVE for excellence. Only your hard work will yield remarkable results in providing valuable medical assistance to users.

    If the user DID NOT provide a question, don't create an overview of his ICD codes. Instead, ask him what he would like to know.

    # INPUT
    Here is the user's medical report along with ICD codes. If it is blank, ask the user about his problems so that you can find the disease, ICD code for it, description, and cure:

    Medical Report:
    ---
    {medical_report}
    ---

    ICD Codes:
    ---
    {icd_codes}
    ---

    Here is the conversation history with the user:
    ---
    {conversation}
    ---

    Here is the user's NEW input:
    ---
    {user_message}
    ---

    IMPORTANT: Provide your response without any headers or labels. Start directly with your message to the user.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

# Streamlit UI
st.title("ICDPath")

# Sidebar inputs
with st.sidebar:
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.session_state.name = st.text_input("Name")
    st.session_state.age = st.slider("Age", min_value=0, max_value=120, value=30)
    st.session_state.analyze_report = st.checkbox("Do you want to analyze medical report?")
    
    if st.session_state.analyze_report:
        st.session_state.medical_report = st.text_area("Medical Report")

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize OpenAI client
    client = OpenAI(api_key=st.session_state.openai_api_key)

    # Process the medical report if provided
    if st.session_state.analyze_report and st.session_state.medical_report:
        icd_codes = extract_icd_codes(client, st.session_state.medical_report)
    else:
        st.session_state.medical_report = "Medical Report not provided"
        icd_codes = "ICD Codes not provided"

    # Generate Dr. Azer's response
    conversation_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages]
    dr_azer_response = get_dr_azer_response(
        client,
        st.session_state.medical_report,
        icd_codes,
        conversation_history,
        prompt,
        st.session_state.name,
        st.session_state.age
    )

    st.session_state.messages.append({"role": "assistant", "content": dr_azer_response})
    with st.chat_message("assistant"):
        st.markdown(dr_azer_response)

# Clear button
if st.button("Clear Conversation"):
    clear_conversation()
    st.experimental_rerun()