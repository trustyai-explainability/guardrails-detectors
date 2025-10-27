# Standard
import os
import ssl
import sys

import uvicorn
import yaml
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter

import logging

from fastapi import FastAPI, status
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)
uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_error_logger.name = "uvicorn"

app = FastAPI(
    title="WxPE Detectors API",
    version="0.0.1",
    contact={
        "name": "Alan Braz",
        "url": "http://alanbraz.com.br/en/",
    },
    dependencies=[],
)


class DetectorBaseAPI(FastAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state.detectors = {}
        self.state.instruments = {
            "detections": Counter(
                "trustyai_guardrails_detections",
                "Number of detections per detector function",
                ["detector_kind", "detector_name"]
            ),
            "requests": Counter(
                "trustyai_guardrails_requests",
                "Number of requests per detector function",
                ["detector_kind", "detector_name"]
            ),
            "errors": Counter(
                "trustyai_guardrails_errors",
                "Number of errors per detector function",
                ["detector_kind", "detector_name"]
            ),
            "runtime": Counter(
                "trustyai_guardrails_runtime",
                "Total runtime of a detector function- this is the induced latency of this guardrail",
                ["detector_kind", "detector_name"]
            )
        }
        #self.state.instruments["detection_rate"].set_function(lambda: self.state.detectors["detections"])
        self.add_exception_handler(
            RequestValidationError, self.validation_exception_handler
        )
        self.add_exception_handler(StarletteHTTPException, self.http_exception_handler)
        self.add_api_route("/health", health, description="Check if server is alive")


    async def validation_exception_handler(self, request, exc):
        errors = exc.errors()
        if len(errors) > 0 and errors[0]["type"] == "missing":
            return await self.parse_missing_required_parameter_response(request, exc)
        elif len(errors) > 0 and errors[0]["type"].endswith("type"):
            return await self.parse_invalid_type_parameter_response(request, exc)
        else:
            # return await request_validation_exception_handler(request, exc)
            return await self.parse_generic_validation_response(request, exc)

    async def parse_missing_required_parameter_response(self, request, exc):
        errors = [
            error["loc"][-1] for error in exc.errors() if error["type"] == "missing"
        ]
        message = f"Missing required parameters: {errors}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "message": message,
            },
        )

    async def parse_invalid_type_parameter_response(self, request, exc):
        errors = [
            error["loc"][-1] for error in exc.errors() if error["type"].endswith("type")
        ]
        message = f"Parameters with invalid type: {errors}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "message": message,
            },
        )

    async def parse_generic_validation_response(self, request, exc):
        errors = [error["loc"][-1] for error in exc.errors()]
        message = f"Invalid parameters: {errors}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "message": message,
            },
        )

    async def http_exception_handler(self, request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.status_code, "message": exc.detail},
        )

    def set_detector(self, detector, detector_name="default") -> None:
        """Store detector in app.state"""
        self.state.detectors[detector_name] = detector

    def get_detector(self, detector_name="default"):
        """Retrieve detector from app.state"""
        return self.state.detectors.get(detector_name)

    def get_all_detectors(self) -> dict:
        """Retrieve all detectors from app.state"""
        return self.state.detectors
    
    def cleanup_detector(self) -> None:
        """Clean up detector resources"""
        self.state.detectors.clear()

async def health():
    return "ok"

def main(app):
    # "loop": "uvloop", (thats default in our setting)
    # "backlog": 10000
    # "timeout_keep_alive": 30
    # limit_concurrency: Maximum number of concurrent connections or tasks to allow, before issuing HTTP 503 responses.
    # timeout_keep_alive: Close Keep-Alive connections if no new data is received within this timeout.
    config = {
        "server": {
            "host": "0.0.0.0",
            "port": "8080",
            "workers": 1,
            "limit_concurrency": 1000,
            "timeout_keep_alive": 30,
        }
    }

    logger.info("config:", os.getenv("CONFIG_FILE_PATH"))

    try:
        with open(os.getenv("CONFIG_FILE_PATH", "config.yaml")) as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError as fnf:
        print(fnf)
    except yaml.YAMLError as exc:
        print(exc)

    for e in os.environ:
        if e.startswith("SERVER_"):
            print(e)
            name = e[len("SERVER_") :].lower()
            config["server"][name] = os.getenv(e)

    if os.getenv("HOST"):
        config["server"]["host"] = os.getenv("HOST")
    config["server"]["port"] = int(
        os.getenv("PORT") if os.getenv("PORT") else config["server"]["port"]
    )
    config["server"]["workers"] = (
        int(config["server"]["workers"])
        if str(config["server"]["workers"])
        else config["server"]["workers"]
    )

    if "ssl_ca_certs" in config["server"]:
        config["server"]["ssl_cert_reqs"] = ssl.CERT_REQUIRED

    logger.info("server configuration: {0}".format(config["server"]))

    try:
        uvicorn.run(app, **config["server"])
    except Exception as e:
        print(e)
        sys.exit(1)
