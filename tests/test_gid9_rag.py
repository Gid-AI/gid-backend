import json
from google.cloud import storage, firestore
from google.cloud import aiplatform
# On suppose que vous avez une fonction pour le vector store
# from vector_store import retrieve_relevant_docs

PROJECT_ID = "gid-ai-5"
REGION = "us-central1"
COMPANY_ID = "ABC"
BUCKET_NAME = "gid9_training_us_central1"

ENDPOINT_NAME = "projects/gid-ai-5/locations/us-central1/endpoints/1672569391591456768"

storage_client = storage.Client(project=PROJECT_ID)
firestore_client = firestore.Client(project=PROJECT_ID)

GUIDELINES_FILE = "Gid_guidelines.json"
TRAITS_DEF_FILE = "Gid_traits_definitions.json"
COMPANY_PROFILES_FILE = "Gid_company_profiles.json"
ACCESS_RULES_FILE = "Gid_access_rules.json"
COMPANY_DATA_FILE = "company_info/ABC/company_data.json"

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

def construct_system_prompt(guidelines, traits_def, applied_traits, company_data, employee_history_data, retrieved_docs):
    # guidelines, traits_def, applied_traits, company_data sont les mêmes qu'avant
    # employee_history_data: l'historique complet de l'employé
    # retrieved_docs: documents pertinents du vector store (RAG), si vous les avez

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

    if retrieved_docs and len(retrieved_docs) > 0:
        prompt += "\n## Retrieved Documents (RAG)\n"
        for doc in retrieved_docs:
            prompt += f"\nTitle: {doc['title']}\nContent: {doc['content']}\n"

    return prompt

def call_vertex_ai(endpoint_name, prompt):
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
    instances = [{"content": prompt}]
    parameters = {
        "temperature": 0.2,
        "maxOutputTokens": 1024,
        "topP": 0.95,
        "topK": 40
    }
    response = endpoint.predict(instances=instances, parameters=parameters)
    return response.predictions

def interact_with_user(employee_id, user_message):
    # 1. Charger données globales (on peut les charger une fois pour toutes au début du script)
    guidelines = load_json_from_gcs(BUCKET_NAME, GUIDELINES_FILE)
    traits_def = load_json_from_gcs(BUCKET_NAME, TRAITS_DEF_FILE)
    # company_profiles = load_json_from_gcs(BUCKET_NAME, COMPANY_PROFILES_FILE) # si nécessaire
    access_rules = load_json_from_gcs(BUCKET_NAME, ACCESS_RULES_FILE)
    company_data = load_json_from_gcs(BUCKET_NAME, COMPANY_DATA_FILE)
    applied_traits = load_traits_for_company(COMPANY_ID)

    # 2. Charger l'historique employé actuel
    employee_history_data = load_employee_history(COMPANY_ID, employee_id)

    # 3. Récupérer documents pertinents via RAG (si vous avez un vector store)
    # retrieved_docs = retrieve_relevant_docs(user_message) # si non dispo, mettez retrieved_docs = []
    retrieved_docs = []  # si pas de vector store, on laisse vide

    # 4. Construire le prompt
    system_prompt = construct_system_prompt(guidelines, traits_def, applied_traits, company_data, employee_history_data, retrieved_docs)
    full_prompt = system_prompt + "\nUser: " + user_message + "\nGid:"

    # 5. Appeler Vertex AI
    response = call_vertex_ai(ENDPOINT_NAME, full_prompt)
    gid_answer = response[0]["content"] if response and len(response) > 0 else "No answer"

    # 6. Afficher la réponse
    print("Gid:", gid_answer)

    # 7. Mettre à jour l'historique de l'employé avec cette nouvelle interaction
    new_interaction = {
        "person_name": "Gid (AI Assistant)",
        "date": "2024-10-10T09:00:00Z",  # idéalement dynamique
        "method": "via Gid",
        "description": f"User asked: {user_message}\nGid answered: {gid_answer}"
    }
    # Catégorie d'interaction : on peut choisir en fonction du contexte, ici on met "regular_interactions"
    update_employee_interactions(COMPANY_ID, employee_id, "regular_interactions", new_interaction)

    # Lors du prochain message, la fonction sera rappelée, rechargera l'historique,
    # et Gid aura l'historique mis à jour.

if __name__ == "__main__":
    employee_id = "EMP001"
    # Simuler une conversation multi-tours
    user_message = "Hi Gid, can you explain your role?"
    interact_with_user(employee_id, user_message)

    # Plus tard, un second message:
    user_message = "Thanks, can you help me set a new career goal?"
    interact_with_user(employee_id, user_message)
