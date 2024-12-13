# Gid Backend

This repository hosts Gid's backend logic, leveraging Retrieval-Augmented Generation (RAG) with the Gemini 1.5 Pro model. The backend orchestrates:

- Interactions with Vertex AI (embeddings, Matching Engine).
- Twilio integration for inbound employee/manager messages and outbound responses.
- Retrieval of contextual data (company policies, employee history, management traits).
- End-to-end RAG pipeline execution, ensuring context-aware, personalized answers.

Structure:
- src/: Backend source code (API endpoints, business logic, AI integration).
- tests/: Unit and integration tests to maintain code quality and reliability.

