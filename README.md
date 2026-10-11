# Policy Assistant

An AI chatbot that answers questions about five fictional company policies: annual leave, travel and expenses, remote work, information security, and the code of conduct. Answers are grounded in the policy PDFs and cite the sections they came from.

It uses OpenRouter for both embeddings (`openai/text-embedding-3-small`) and answers (`openai/gpt-oss-20b`), so you need an OpenRouter API key.

## 1. Download

On this repository's GitHub page, click **Code**, then either:
- copy the URL, run `git clone <URL>`, and `cd` into the new folder; or
- choose **Download ZIP**, unzip it, and open a terminal in the unzipped folder.

You need [Python 3.12+](https://www.python.org/downloads/), [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Node.js](https://nodejs.org/) 22 (22.12 or later) or 24. If `uv` isn't on your PATH after installing it, use `python -m uv` in its place.

## 2. Add your API key

Copy `.env.example` to `.env` in the repository root:

~~~powershell
Copy-Item .env.example .env   # macOS/Linux: cp .env.example .env
~~~

Open `.env` and set `OPENROUTER_API_KEY` to your key from https://openrouter.ai/keys. `.env` is gitignored, so the key is never committed.

## 3. Run

Use two terminals, both starting from the repository root.

**Terminal 1: backend**

~~~powershell
cd backend
uv sync
uv run uvicorn app.main:app --reload --reload-dir app
~~~

**Terminal 2: frontend**

~~~powershell
cd frontend
npm install
npm run dev
~~~

Open http://localhost:5173 and ask a question, for example "How many unused leave days can I carry over?". Each answer costs well under $0.001.

The first question takes a few seconds longer, because the backend embeds the policy documents then. The API docs are at http://localhost:8000/docs.

## Run the tests (optional)

The tests use mocked API responses, so they need no key and cost nothing.

~~~powershell
cd backend
uv run pytest

cd ../frontend
npm test
~~~

## More information

- [Overview](docs/overview.md): how the assistant behaves, the API, configuration, evaluation results, and known issues.
- [Architecture](docs/architecture.md): components, request flow, failure modes, and what production would need.
- [Decision records](docs/decisions/): why the stack, retrieval and hosted models were chosen.
