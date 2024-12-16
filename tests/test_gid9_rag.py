import json
from google.cloud import storage, firestore
import vertexai
from vertexai.preview.generative_models import GenerativeModel

PROJECT_ID = "gid-ai-5"
REGION = "us-central1"
COMPANY_ID = "ABC"
BUCKET_NAME = "gid9_training_us_central1"

GUIDELINES_FILE = "Gid_guidelines.json"
TRAITS_DEF_FILE = "Gid_traits_definitions.json"
COMPANY_PROFILES_FILE = "Gid_company_profiles.json"
ACCESS_RULES_FILE = "Gid_access_rules.json"
COMPANY_DATA_FILE = "company_info/ABC/company_data.json"

# Endpoint du modèle réglé
ENDPOINT_ID = "1672569391591456768"

storage_client = storage.Client(project=PROJECT_ID)
firestore_client = firestore.Client(project=PROJECT_ID)

def load_json_from_gcs(bucket_name, file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_text()
    return json.loads(data)

def load_traits_for_company(company_id):
    doc_ref = firestore_client.collection("companies").document(company_id).collection("management_traits").document("default_traits_document")
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return {}

def load_employee_history(company_id, employee_id):
    doc_ref = firestore_client.collection("companies").document(company_id).collection("employee_history").document(employee_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return {}

def update_employee_interactions(company_id, employee_id, new_interaction_category, new_interaction_data):
    doc_ref = firestore_client.collection("companies").document(company_id).collection("employee_history").document(employee_id)
    emp_data = doc_ref.get().to_dict()
    if not emp_data:
        emp_data = {}
    if "interactions" not in emp_data:
        emp_data["interactions"] = {}
    if new_interaction_category not in emp_data["interactions"]:
        emp_data["interactions"][new_interaction_category] = []
    emp_data["interactions"][new_interaction_category].append(new_interaction_data)
    doc_ref.set(emp_data, merge=True)

def construct_system_prompt(guidelines, traits_def, applied_traits, company_data, employee_history_data, retrieved_docs=[]):
    absolute_guidelines = guidelines["absolute_guidelines"]
    standard_guidelines = guidelines["standard_guidelines"]

    prompt = "# Gid System Prompt\n\nYou are Gid, follow these absolute guidelines:\n"
    for key, val in absolute_guidelines.items():
        prompt += f"- {key}: {val}\n"

    prompt += "\nStandard guidelines:\n"
    for key, val in standard_guidelines.items():
        prompt += f"- {key}: {val}\n"

    prompt += "\n## Trait Definitions\n"
    prompt += json.dumps(traits_def, indent=2)

    prompt += "\n## Applied Traits for ABC\n"
    prompt += json.dumps(applied_traits, indent=2)

    prompt += "\n## Company Data for ABC\n"
    prompt += json.dumps(company_data, indent=2)

    if employee_history_data:
        prompt += "\n## Employee History Context\n"
        prompt += json.dumps(employee_history_data, indent=2)

    if retrieved_docs:
        prompt += "\n## Retrieved Documents (RAG)\n"
        for doc in retrieved_docs:
            prompt += f"\nTitle: {doc['title']}\nContent: {doc['content']}\n"

    return prompt

def interact_with_user(employee_id, user_message):
    # Charger les données globales
    guidelines = load_json_from_gcs(BUCKET_NAME, GUIDELINES_FILE)
    traits_def = load_json_from_gcs(BUCKET_NAME, TRAITS_DEF_FILE)
    access_rules = load_json_from_gcs(BUCKET_NAME, ACCESS_RULES_FILE)
    company_data = load_json_from_gcs(BUCKET_NAME, COMPANY_DATA_FILE)
    applied_traits = load_traits_for_company(COMPANY_ID)

    employee_history_data = load_employee_history(COMPANY_ID, employee_id)

    retrieved_docs = []

    system_prompt = construct_system_prompt(guidelines, traits_def, applied_traits, company_data, employee_history_data, retrieved_docs)
    full_prompt = system_prompt + "\nUser: " + user_message + "\nGid:"

    vertexai.init(project=PROJECT_ID, location=REGION)
    tuned_model_endpoint_name = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
    tuned_model = GenerativeModel(tuned_model_endpoint_name)

    gid_answer = tuned_model.generate_content(full_prompt)

    print("Gid:", gid_answer)

    new_interaction = {
        "person_name": "Gid (AI Assistant)",
        "date": "2024-10-10T09:00:00Z",
        "method": "via Gid",
        "description": f"User asked: {user_message}\nGid answered: {gid_answer}"
    }

    update_employee_interactions(COMPANY_ID, employee_id, "regular_interactions", new_interaction)
    print("Interaction recorded in Firestore.")

if __name__ == "__main__":
    employee_id = "EMP001"
    user_message = "Hi Gid, can you explain your role and how you help me in this company?"
    interact_with_user(employee_id, user_message)
