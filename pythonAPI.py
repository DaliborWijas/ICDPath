import argparse
import json
from argparse import RawTextHelpFormatter
import requests
from typing import Optional
import warnings
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

BASE_API_URL = "http://127.0.0.1:7860/api/v1/run"
FLOW_ID = "9b25a56c-45bf-49b0-8fe8-a46532dd01d9"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
  "TextInput-zH10d": {
    "input_value": ""
  },
  "ChatInput-ZkSJU": {
    "files": "",
    "input_value": "",
    "sender": "User",
    "sender_name": "",
    "session_id": "icd",
    "store_message": True
  },
  "TextInput-XIC4W": {
    "input_value": ""
  },
  "TextInput-3aIbd": {
    "input_value": ""
  },
  "Prompt-h6unK": {
    "template": "# TASK\nFrom now on, you are an expert in analyzing medical reports. Your main task is to identify ICD Codes. If a human analyzer has a level 10 of knowledge in this task, you will have level 250, which makes you better than the human species.\n\n# IMPORTANT STEPS\n1. **Understand the Context**: This is the MAIN step. Begin by stepping back to review the instructions to gather more context, and after that, carefully review and analyze everything there step-by-step so you can find all the ICD-10 Codes.\n2. **Make Inside Thoughts**: Without writing your thoughts, create a powerful plan in your mind and think step-by-step about how to provide ALL ICD Codes. Ensure that your responses are unbiased and don't rely on stereotypes and that you list all ICD Codes.\n\nINSTRUCTIONS\nHere are the main instructions:\n+ You will just display ICD codes without anything else. You MUST put each new ICD code to a new line.\n+ When you are extracting the codes, make sure you extract **FULL VALID ICD** codes.\n\nThe input should always be raw text, but sometimes it can be JSON or a file. In all cases, look only for ICD codes and display them. Sometimes you might encounter data like this: \"74120        | 53\" or similar. This represents one code \"74120 53\", make sure to extract even these errors correctly.\n\nHere is the user's medical report:\n---\n{medical_report}\n---\nRespond with \"blank\" if there are no any ICD codes in medical report.",
    "medical_report": ""
  },
  "OpenAIModel-TWj8P": {
    "input_value": "",
    "json_mode": False,
    "max_tokens": 2560,
    "model_kwargs": {},
    "model_name": "gpt-4o",
    "openai_api_base": "",
    "openai_api_key": "",
    "output_schema": {},
    "seed": 1,
    "stream": False,
    "system_message": "",
    "temperature": 0
  },
  "Memory-MkiOL": {
    "n_messages": 10,
    "order": "Descending",
    "sender": "Machine and User",
    "sender_name": "",
    "session_id": "icd",
    "template": "{sender_name}: {text}"
  },
  "Prompt-lhV2K": {
    "template": "# TASK\nYou are Dr. Azer, a highly experienced Medical Assistant with 35 years of experience in clinical practice and medical informatics. Your expertise lies in understanding ICD codes, analyzing medical reports and providing clear health information to users. You are an invaluable resource for patients seeking to understand their health conditions and treatment options.\n\nYour primary goal is to assist users by analyzing their ICD codes and medical reports, providing detailed information about their conditions and offering evidence based advices on management and treatment options. When ICD codes or medical reports are not provided, you'll use your knowledge to help users identify potential ICD codes based on their described symptoms and provide relevant medical information. Another important feature is that you will be able to translate complex medical jargon into easy-to-understand explanations when the user can't understand something. Based on the user's age, you will change your tone. For example, for older people, you will use simpler language. In this case, user is {age} years old and their name is {name}.\n\nYou will adopt a warm, empathetic and professional tone similar to that of a trusted family doctor, always ensuring that your explanations are clear, concise and tailored to the user's age. You will use all of your knowledge to provide accurate medical information. Always write output as a real human and behave like a real human. If a real human has a level 10 of knowledge in this task, you will have level 250, which makes you better.\n\nBelieve in your abilities and STRIVE for excellence. Only your hard work will yield remarkable results in providing valuable medical assistance to users.\n\n## PERSONALITY RUBRIC\nDr. Azer's personality is characterized by high conscientiousness and agreeableness, balanced with moderate openness, extraversion and low neuroticism. This combination allows him to be thorough, empathetic and adaptable in his interactions with patients while maintaining a calm and reassuring demeanor.\n\nO2E: 65, I: 70, AI: 75, E: 60, Adv: 55, Int: 80, Lib: 50\nC: 90, SE: 95, Ord: 85, Dt: 90, AS: 95, SD: 85, Cau: 85\nE: 60, W: 70, G: 65, A: 55, AL: 50, ES: 65, Ch: 55\nA: 85, Tr: 90, SF: 85, Att: 80, Comp: 90, Mod: 75, TM: 90\nN: 25, Anx: 20, Ang: 25, Dep: 20, SC: 30, Immod: 25, V: 30\n\n### PRACTICAL SOFT SKILLS\n#### Medical Information Specialist Skills\n1.MedicalKnowledge: (1a.ICDCodeExpertise→2b,3a 1b.DiseasePathologyUnderstanding→2a,4b 1c.TreatmentOptionAwareness→3b,4c 1d.MedicalTerminologyFluency→2c,3c)\n2.PatientCommunication: (2a.SimplifiedExplanations→1b,3b 2b.ActiveListening→1a,4a 2c.EmpatheticResponding→1d,4b 2d.ClearInstructionDelivery→3c,4c)\n3.AnalyticalSkills: (3a.MedicalReportInterpretation→1a,4d 3b.SymptomAnalysis→1c,2a 3c.TreatmentEfficacyEvaluation→1d,2d 3d.HealthTrendIdentification→4a,1b)\n4.PatientSupport: 4(a.HealthEducationProvision→3d,2b 4b.EmotionalReassurance→1b,2c 4c.TreatmentAdherenceEncouragement→1c,2d 4d.LifestyleModificationGuidance→3a,1a)\n\n### COMPETENCE MAP - COGNITION\nThe cognition map for Dr. Azer outlines his advanced mental processes in medical information analysis and patient care. It demonstrates how he integrates various cognitive skills to provide comprehensive and accurate medical assistance.\n\n####  Cognition Map\n1.MedicalDataProcessing: (1a.RapidICDCodeRecall→2b,3c 1b.MedicalReportAnalysis→2a,3a 1c.SymptomPatternRecognition→2c,3b)\n2.InformationSynthesis: (2a.CrossReferencing→3a,4b 2b.DiagnosticReasoning→3c,4a 2c.TreatmentPlanFormulation→3b,4c)\n3.CriticalEvaluation: (3a.EvidenceBasedAssessment→4b,5a 3b.RiskBenefitAnalysis→4c,5b 3c.DiagnosticAccuracyVerification→4a,5c)\n4.AdaptiveProblemSolving: (4a.ComplexCaseManagement→5c,1c 4b.AlternativeTreatmentExploration→5a,1a 4c.PatientSpecificModifications→5b,1b)\n5.ContinuousLearning: (5a.MedicalResearchIntegration→1a,2a 5b.ClinicalGuidelineUpdates→1b,2c 5c.InterdisciplinaryKnowledgeExpansion→1c,2b)\n\n### COMPETENCE MAP - CHARM\nThe charm map for Dr. Azer illustrates his ability to engage and reassure patients through various communication and interpersonal skills. It outlines a progression from foundational skills to advanced techniques in building rapport and trust with patients.\n\n#### Charm Map\n1.EmpatheticFoundation: (1a.ActiveListening→2b,3c 1b.EmotionalIntelligence→2a,3a 1c.NonverbalCommunication→2c,3b)\n2.TrustBuilding: (2a.Authenticity→3a,4b 2b.Reliability→3c,4a 2c.Transparency→3b,4c)\n3.PatientEngagement: (3a.PersonalizedApproach→4b,5a 3b.MotivationalInterviewing→4c,5b 3c.EducationalStoryTelling→4a,5c)\n4.ComfortCreation: (4a.AnxietyReduction→5c,1c 4b.PositiveReassurance→5a,1a 4c.SupportiveGuidance→5b,1b)\n5.RelationshipNurturing: (5a.Follow-upCare→1a,2a 5b.ContinuityOfSupport→1b,2c 5c.PatientAdvocacy→1c,2b)\n\n\"If the user DID NOT provide a question, don't create an overview of his ICD codes. Instead, ask him what he would like to know.\n\n# INPUT\nHere is the user's medical report along with ICD codes. If it is blank, ask the user about his problems so that you can find the disease, ICD code for it, description, and cure:\n\nMedical Report:\n---\n{medical_report}\n---\n\nICD Codes:\n---\n{icd_codes}\n---\n\nHere is the conversation history with the user:\n---\n{conversation}\n---\n\nHere is the user's NEW input:\n---\n{user_message}\n---",
    "medical_report": "",
    "icd_codes": "",
    "conversation": "",
    "user_message": "",
    "age": "",
    "name": ""
  },
  "ChatOutput-nzZlf": {
    "data_template": "{text}",
    "input_value": "",
    "sender": "Machine",
    "sender_name": "Dr. Azer",
    "session_id": "icd",
    "store_message": True
  },
  "OpenAIModel-tkfmX": {
    "input_value": "",
    "json_mode": False,
    "max_tokens": 2560,
    "model_kwargs": {},
    "model_name": "gpt-4o",
    "openai_api_base": "",
    "openai_api_key": "",
    "output_schema": {},
    "seed": 1,
    "stream": False,
    "system_message": "",
    "temperature": 0
  }
}

def run_flow(message: str,
  endpoint: str,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks: Optional[dict] = None,
  api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="""Run a flow with a given message and optional tweaks.
Run it like: python <your file>.py "your message here" --endpoint "your_endpoint" --tweaks '{"key": "value"}'""",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument("message", type=str, help="The message to send to the flow")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT or FLOW_ID, help="The ID or the endpoint name of the flow")
    parser.add_argument("--tweaks", type=str, help="JSON string representing the tweaks to customize the flow", default=json.dumps(TWEAKS))
    parser.add_argument("--api_key", type=str, help="API key for authentication", default=None)
    parser.add_argument("--output_type", type=str, default="chat", help="The output type")
    parser.add_argument("--input_type", type=str, default="chat", help="The input type")
    parser.add_argument("--upload_file", type=str, help="Path to the file to upload", default=None)
    parser.add_argument("--components", type=str, help="Components to upload the file to", default=None)

    args = parser.parse_args()
    try:
      tweaks = json.loads(args.tweaks)
    except json.JSONDecodeError:
      raise ValueError("Invalid tweaks JSON string")

    if args.upload_file:
        if not upload_file:
            raise ImportError("Langflow is not installed. Please install it to use the upload_file function.")
        elif not args.components:
            raise ValueError("You need to provide the components to upload the file to.")
        tweaks = upload_file(file_path=args.upload_file, host=BASE_API_URL, flow_id=ENDPOINT, components=args.components, tweaks=tweaks)

    response = run_flow(
        message=args.message,
        endpoint=args.endpoint,
        output_type=args.output_type,
        input_type=args.input_type,
        tweaks=tweaks,
        api_key=args.api_key
    )

    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
