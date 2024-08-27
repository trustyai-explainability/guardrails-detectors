## Running app locally

1. Ensure you have a Python 3.11 virtual environment set up. If you don't, you can create one by running e.g in your terminal:

```bash
pyenv install 3.11.0
pyenv local 3.11.0
python3 -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install the __minmal__ dependencies for the regex match detector:

```bash
pip install uvicorn "fastapi[standard]" PyYAML
```

4. Navigate to `detectors/regex_match` and run the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

5. Once the app is running, you can send a POST request to `http://localhost:8000/regex_match` with a JSON payload containing the `text` and `regex` keys. For example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/text/contents" \
-H "Content-Type: application/json" \
-H "detector-id: has_regex_match" \
-d '{
    "contents": ["My email address is xx@domain.com and zzz@hotdomain.co.uk"],
    "regex_pattern": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
}'
```

## Running app inside a container

1. Navigate to the `detectors/` directory and build the Docker image from the `Dockerfile.regex` file:

```bash
podman build -t regex-match -f Dockerfile.regex .  
```

2. Run the Docker container:

```bash
podman run -d -p 8000:8000 regex-match:latest      
```

3. Once the container is running, you can send a POST request to `http://localhost:8000/regex_match` with a JSON payload containing the `text` and `regex` keys. For example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/text/contents" \
-H "Content-Type: application/json" \
-H "detector-id: has_regex_match" \
-d '{
    "contents": ["My email address is xx@domain.com and zzz@hotdomain.co.uk"],
    "regex_pattern": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
}'
```