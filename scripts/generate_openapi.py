"""Generate the Hugging Face detector's text-contents API contract.

The application lifespan is deliberately not entered: schema generation imports
the Hugging Face dependencies but does not load model weights.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = "/api/v1/text/contents"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on a missing or stale contract")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from fastapi.openapi.utils import get_openapi
    from detectors.huggingface.app import app

    # Use the actual route and Pydantic models without duplicating definitions.
    document = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        routes=[route for route in app.routes if getattr(route, "path", None) == CONTRACT_PATH],
    )
    if CONTRACT_PATH not in document["paths"]:
        raise RuntimeError(f"Hugging Face detector no longer serves {CONTRACT_PATH}")
    generated = json.dumps(document, indent=2, sort_keys=True) + "\n"
    artifact = ROOT / "api" / "openapi" / "huggingface.json"
    if args.check:
        if not artifact.is_file() or artifact.read_text() != generated:
            print(f"Missing or stale OpenAPI document: {artifact.relative_to(ROOT)}", file=sys.stderr)
            print("Run python scripts/generate_openapi.py and commit the result.", file=sys.stderr)
            return 1
        print("Generated OpenAPI document is current.")
    else:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(generated)
        print(f"Generated {artifact.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
