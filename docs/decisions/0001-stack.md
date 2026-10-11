# 0001: Application stack and repository layout

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

The system needs a web UI where a user asks a question and gets a grounded answer. The repo must be easy for another engineer to clone and run, and it must keep the AI/ML parts (retrieval, generation, evaluation) cleanly separate from ordinary application concerns (HTTP, validation, UI).

## Decision

- **Backend: Python + FastAPI.** Python has the strongest ecosystem for RAG work: document parsing, embeddings, vector stores and evaluation libraries. FastAPI gives typed request/response models (Pydantic), automatic OpenAPI docs, and dependency injection, which is what lets the RAG pipeline be swapped behind an interface.
- **Frontend: React + Vite + TypeScript**, a separate single-page app. It keeps the UI decoupled from the API contract, and TypeScript types mirror the backend schemas.
- **Same-origin `/api` proxy** through the Vite dev server instead of cross-origin calls. There is no CORS configuration, and secrets never reach the browser.
- **uv for Python dependencies** (`uv.lock`) and **npm** (`package-lock.json`) for the frontend. Both give reproducible installs. npm ships with Node, so reviewers need no extra tool.
- **No containers or deployment setup.** This is a proof of concept that runs locally with two commands. Docker and a production web server are deferred until there is something to deploy.
- **Walking skeleton first.** The end-to-end path, API contract, tests and packaging exist before any RAG logic. This removes integration risk early and lets each RAG stage be added and measured on its own.

## Alternatives considered

| Option                          | Why not (for now)                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------- |
| Streamlit / Gradio              | Fastest to build, but mixes UI and pipeline code and is hard to evolve into a real service |
| FastAPI + server-rendered HTMX  | Simpler (one language), but less flexible for richer citation UX such as highlighting and source previews |
| Next.js full-stack (TypeScript) | Good UI story, but a weaker RAG/eval ecosystem than Python                        |
| pip + requirements.txt          | No lockfile with hashes for the whole tree by default; uv is faster and reproducible |

## Consequences

- Two languages and two toolchains: running the PoC needs Python, uv and Node installed locally.
- Running it needs two terminals, one for the backend and one for the frontend.
- Backend schemas and frontend types are kept in sync by hand. If the API grows, generating TypeScript types from the OpenAPI schema is the natural next step.
