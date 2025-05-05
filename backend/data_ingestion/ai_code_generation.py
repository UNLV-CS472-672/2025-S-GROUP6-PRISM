"""
Created by Jesse Ortega, Spring 2025.

================================================================================
 AI CODE GENERATION PIPELINE
================================================================================

Overview:
---------
This script automates the generation of sample programming solutions by prompting
multiple Large Language Models (LLMs) — including Gemini, ChatGPT, Claude, Grok,
and Ollama (Phi-4, DeepSeek) — using a structured programming assignment as input.

The script parses and refines the assignment PDF and injects detailed test cases
to guide each LLM into generating a functionally accurate and well-commented solution.

--------------------------------------------------------------------------------
Required Folder Structure per Assignment:
-----------------------------------------
The following structure is required for proper functioning of the script.
The assignment directory must be named as `assignment_<id>`.

Example:
assignment_7/
├── ai_submissions/
├── bulk_submission/
├── moss_reports/
├── test_cases/
│   ├── 1/
│   │   ├── input_files/
│   │   │   └── input10.txt
│   │   ├── output_files/
│   │   │   └── grade_report.txt
│   │   ├── console_command.txt
│   │   ├── console_input.txt
│   │   └── expected_console_output.txt
│   ├── 2/
│   │   └── ...
│   └── 3/
│       └── ...
├── assignment_6.pdf

Note:
- Without all files above (including assignment PDF and test case files), the system cannot accurately construct prompts for any LLM.
- The assignment PDF is parsed to extract core requirements and stripped of noise.
- The test cases guide program logic through structured inputs/outputs.

--------------------------------------------------------------------------------
.env Configuration (MANDATORY):
-------------------------------
All credentials and runtime settings are loaded from the `.env` file.
The following keys must be set:

[Google Gemini]
- GEMINI_API_KEY
- GEMINI_MODEL
- GEMINI_REQUESTS_PER_MINUTE_LIMIT
- GEMINI_REQUESTS_PER_DAY_LIMIT
- GEMINI_TOKENS_PER_MINUTE
- GEMINI_LOG_PATH

[OpenAI ChatGPT]
- OPENAI_API_KEY
- OPENAI_MODEL
- OPENAI_REQUESTS_PER_MINUTE_LIMIT
- OPENAI_TOKENS_PER_MINUTE_LIMIT
- OPENAI_LOG_PATH
- OPENAI_MODEL_COST_PER_INPUT_TOKEN
- OPENAI_MODEL_COST_PER_CACHE_INPUT_TOKEN
- OPENAI_MODEL_COST_PER_OUTPUT_TOKEN
- OPENAI_MINIMUM_BALANCE

[Anthropic Claude]
- ANTHROPIC_API_KEY
- ANTHROPIC_MODEL
- ANTHROPIC_REQUESTS_PER_MINUTE_LIMIT
- ANTHROPIC_INPUT_TOKENS_PER_MINUTE_LIMIT
- ANTHROPIC_OUTPUT_TOKENS_PER_MINUTE_LIMIT
- ANTHROPIC_LOG_PATH
- ANTHROPIC_MODEL_COST_PER_INPUT_TOKEN
- ANTHROPIC_MODEL_COST_PER_CACHE_WRITE_TOKEN
- ANTHROPIC_MODEL_COST_PER_CACHE_READ_TOKEN
- ANTHROPIC_MODEL_COST_PER_OUTPUT_TOKEN
- ANTHROPIC_MINUMUM_BALANCE

[Ollama (Local Models)]
- OLLAMA_API_ADDRESS
- OLLAMA_LOG_PATH
- OLLAMA_PHI_MODEL=phi4:latest
- OLLAMA_DEEPSEEK_MODEL=deepseek-r1:32b-qwen-distill-q8_0

[Grok (OpenAI-Compatible API)]
- GROK_API_KEY
- GROK_API_URL
- GROK_LOG_PATH
- GROK_MODEL
- GROK_MODEL_CONTEXT_LIMIT
- GROK_MODEL_COST_PER_INPUT_TOKEN
- GROK_MODEL_COST_PER_OUTPUT_TOKEN
- GROK_MINIMUM_BALANCE

--------------------------------------------------------------------------------
STRICT USAGE POLICY:
---------------------
All API keys connect to the developer’s personal paid accounts.
**Do NOT abuse or re-use these credentials outside of this script.**
This project has budget-sensitive LLM usage — prompt costs are logged and enforced.

--------------------------------------------------------------------------------
Model Quality Notes:
--------------------

[OLLAMA_PHI_MODEL = phi4:latest]
- Performs well on small assignments with simple context windows.
- Breaks down with complex/multi-page assignments (often generates gibberish).
- Prompt compression may help but doesn't fully fix context overflows.

[OLLAMA_DEEPSEEK_MODEL = deepseek-r1:32b-qwen-distill-q8_0]
- Very slow to respond; sometimes stalls indefinitely.
- Highly sensitive to prompt structure and can fail unpredictably.
- Performance may improve with tuning, but results remain unreliable.

--------------------------------------------------------------------------------
Areas for Improvement (TODO):
-----------------------------
- Parallelize LLM requests for faster generation.
- Refactor duplicated logic (e.g., test case parsing).
- Improve prompt quality and model-specific tuning.
- Track and compare processed PDFs across models to assess quality.
- Enhance error handling (support retries or resume-on-crash logging).
- Integrate with Django as a callable function or REST API endpoint.

--------------------------------------------------------------------------------
"""

# LLM API clients
import google.generativeai as genai  # Google Gemini API client
from openai import OpenAI  # OpenAI (ChatGPT, Grok) API client
import anthropic  # Anthropic (Claude) API client
import ollama  # Ollama API client (local LLM inference)

# Core Python libraries
import json  # JSON encoding and decoding
import sys  # Access to command-line arguments and system functions
import logging  # Structured logging for API calls and application events
import inspect  # Runtime inspection for method metadata (e.g., getting current method names)
import time  # Sleep and timing operations
from datetime import (
    datetime,
    timedelta,
)  # Date and time handling for API rate limits and metadata
from decimal import (
    Decimal,
)  # Precise decimal arithmetic for API billing/budget tracking

# Third-party libraries
import pandas as pd  # Data manipulation and database querying (assignment metadata)
from sqlalchemy import create_engine  # Database connection engine (PostgreSQL)
from decouple import config  # Environment variable and configuration management
from pathlib import Path  # Cross-platform filesystem path handling
import pymupdf4llm  # PDF-to-Markdown conversion optimized for LLM processing
from lxml import etree  # XML/HTML element building for structured prompt construction


# Basic configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create or get the logger
api_logger = logging.getLogger("api_logger")
api_logger.setLevel(logging.INFO)

# Define formatter
formatter = logging.Formatter(
    "{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M:%S"
)

# Main log file handler (logs everything)
file_handler_all = logging.FileHandler("logs/api_calls_all.log")
file_handler_all.setLevel(logging.INFO)
file_handler_all.setFormatter(formatter)

# Gemini-specific log file handler
file_handler_gemini = logging.FileHandler(config("GEMINI_LOG_PATH"))
file_handler_gemini.setLevel(logging.INFO)
file_handler_gemini.setFormatter(formatter)


# Custom filter to allow only Gemini logs
class GeminiFilter(logging.Filter):
    """
    GeminiFilter logging stuff.

    logging stuff.
    """

    def filter(self, record):
        """
        filter: Filters log records to include only those related to the Gemini model.

        This method is typically used in a custom logging filter to control which log
        entries are processed or displayed. It checks whether the message content contains
        the string "Gemini".

        Parameters
        ----------
        record : logging.LogRecord
            A log record object containing information about the event being logged.

        Returns
        -------
        bool
            True if the log message includes "Gemini", otherwise False.
        """
        return "Gemini" in record.getMessage()


file_handler_gemini.addFilter(GeminiFilter())

# ChatGPT-specific log file handler
file_handler_openai = logging.FileHandler(config("OPENAI_LOG_PATH"))
file_handler_openai.setLevel(logging.INFO)
file_handler_openai.setFormatter(formatter)


# Custom filter to allow only ChatGPT logs
class OpenAIFilter(logging.Filter):
    """
    OpenAIFilter logging stuff.

    logging stuff.
    """

    def filter(self, record):
        """
        filter: Filters log records to include only those related to the ChatGPT model.

        This method is typically used in a custom logging filter to control which log
        entries are processed or displayed. It checks whether the message content contains
        the string "ChatGPT".

        Parameters
        ----------
        record : logging.LogRecord
            A log record object containing information about the event being logged.

        Returns
        -------
        bool
            True if the log message includes "ChatGPT", otherwise False.
        """
        return "ChatGPT" in record.getMessage()


file_handler_openai.addFilter(OpenAIFilter())

# Claude-specific log file handler
file_handler_anthropic = logging.FileHandler(config("ANTHROPIC_LOG_PATH"))
file_handler_anthropic.setLevel(logging.INFO)
file_handler_anthropic.setFormatter(formatter)


# Custom filter to allow only Claude logs
class AnthropicFilter(logging.Filter):
    """
    AnthropicFilter logging stuff.

    logging stuff.
    """

    def filter(self, record):
        """
        filter: Filters log records to include only those related to the Claude model.

        This method is typically used in a custom logging filter to control which log
        entries are processed or displayed. It checks whether the message content contains
        the string "Claude".

        Parameters
        ----------
        record : logging.LogRecord
            A log record object containing information about the event being logged.

        Returns
        -------
        bool
            True if the log message includes "Claude", otherwise False.
        """
        return "Claude" in record.getMessage()


file_handler_anthropic.addFilter(AnthropicFilter())

# Ollama-specific log file handler
file_handler_ollama = logging.FileHandler(config("OLLAMA_LOG_PATH"))
file_handler_ollama.setLevel(logging.INFO)
file_handler_ollama.setFormatter(formatter)


# Custom filter to allow only ollama logs
class OllamaFilter(logging.Filter):
    """
    OllamaFilter logging stuff.

    logging stuff.
    """

    def filter(self, record):
        """
        filter: Filters log records to include only those related to the Ollama model.

        This method is typically used in a custom logging filter to control which log
        entries are processed or displayed. It checks whether the message content contains
        the string "Ollama".

        Parameters
        ----------
        record : logging.LogRecord
            A log record object containing information about the event being logged.

        Returns
        -------
        bool
            True if the log message includes "Ollama", otherwise False.
        """
        return "Ollama" in record.getMessage()


file_handler_ollama.addFilter(OllamaFilter())

# Grok-specific log file handler
file_handler_grok = logging.FileHandler(config("GROK_LOG_PATH"))
file_handler_grok.setLevel(logging.INFO)
file_handler_grok.setFormatter(formatter)


# Custom filter to allow only ollama logs
class GrokFilter(logging.Filter):
    """
    GrokFilter logging stuff.

    logging stuff.
    """

    def filter(self, record):
        """
        filter: Filters log records to include only those related to the Grok model.

        This method is typically used in a custom logging filter to control which log
        entries are processed or displayed. It checks whether the message content contains
        the string "Grok".

        Parameters
        ----------
        record : logging.LogRecord
            A log record object containing information about the event being logged.

        Returns
        -------
        bool
            True if the log message includes "Grok", otherwise False.
        """
        return "Grok" in record.getMessage()


file_handler_grok.addFilter(GrokFilter())

# Clear existing handlers to prevent duplicate logs
if api_logger.hasHandlers():
    api_logger.handlers.clear()

# Add fresh file handlers
api_logger.addHandler(file_handler_all)
api_logger.addHandler(file_handler_gemini)
api_logger.addHandler(file_handler_openai)
api_logger.addHandler(file_handler_anthropic)
api_logger.addHandler(file_handler_ollama)
api_logger.addHandler(file_handler_grok)

# Establish database connection
ENGINE = create_engine(
    f"postgresql+psycopg2://{config("DB_USER")}:{config("DB_PASSWORD")}@{config("DB_HOST")}:{config("DB_PORT")}/{config("DB_NAME")}"
)


def main() -> None:
    """
    Entry point of the AI code generation script.

    This function initializes the `genAI` class using command-line arguments
    and orchestrates the end-to-end process of querying the database, processing
    the assignment PDF, prompting multiple LLMs (Gemini, ChatGPT, Claude, Ollama, Grok),
    and saving the generated code outputs.

    It expects exactly one command-line argument specifying the assignment ID.
    """
    genAI(sys.argv)


class promptAI:
    """
     PromptAI: Prompt AI to generate code.

    Prompt AI to generate code.
    """

    GEMINI_FINISH_REASON = {
        0: "FINISH_REASON_UNSPECIFIED",
        1: "STOP",
        2: "MAX_TOKENS",
        3: "SAFETY",
        4: "RECITATION",
        5: "LANGUAGE",
        6: "OTHER",
        7: "BLOCKLIST",
        8: "PROHIBITED_CONTENT",
        9: "SPII",
        10: "MALFORMED_FUNCTION_CALL",
        11: "IMAGE_SAFETY",
    }

    def __init__(self) -> None:
        """
        __init__: Initializes the `promptAI` class by configuring API clients, environment parameters, logging directories, token usage tracking, and budget tracking for multiple LLM providers.

        This constructor loads credentials and configuration values for:
        - Google Gemini
        - OpenAI (ChatGPT and Grok)
        - Anthropic Claude
        - Ollama (local models)

        It also prepares log files for structured API call metadata, creates necessary directories
        if missing, initializes per-minute and per-day usage histories, and verifies that each
        model is ready for interaction.

        Raises
        ------
        Exception
            If any API client configuration, environment loading, or filesystem operation fails during initialization.
        """
        try:
            logging.info(
                "Loading environment variables and initializing API parameters..."
            )

            log_dir = Path(config("LOG_DIR"))
            if not log_dir.exists():
                log_dir.mkdir(mode=551)

            # Google (Gemini)
            genai.configure(api_key=config("GEMINI_API_KEY"))
            self.GEMINI_REQUESTS_PER_MINUTE_LIMIT = int(
                config("GEMINI_REQUESTS_PER_MINUTE_LIMIT")
            )
            self.GEMINI_REQUESTS_PER_DAY_LIMIT = int(
                config("GEMINI_REQUESTS_PER_DAY_LIMIT")
            )
            self.GEMINI_TOKENS_PER_MINUTE = int(config("GEMINI_TOKENS_PER_MINUTE"))
            self.GEMINI_MODEL = genai.GenerativeModel(config("GEMINI_MODEL"))
            self.GEMINI_MODEL_NAME = config("GEMINI_MODEL")
            self.GEMINI_LOG_PATH = Path(config("GEMINI_LOG_PATH"))
            self.gemini_minute_history = []
            self.gemini_day_history = []
            self.gemini_tokens_minute_history = 0

            self.GEMINI_LOG_PATH.touch()
            logging.info(
                f"Gemini model '{self.GEMINI_MODEL_NAME}' selected for prompting..."
            )

            # OpenAI (ChatGPT)
            self.OPENAI_API_KEY = config("OPENAI_API_KEY")
            self.OPENAI_CLIENT = OpenAI(api_key=config("OPENAI_API_KEY"))
            self.OPENAI_REQUESTS_PER_MINUTE_LIMIT = int(
                config("OPENAI_REQUESTS_PER_MINUTE_LIMIT")
            )
            self.OPENAI_TOKENS_PER_MINUTE_LIMIT = int(
                config("OPENAI_TOKENS_PER_MINUTE_LIMIT")
            )
            self.OPENAI_MODEL = config("OPENAI_MODEL")
            self.OPENAI_LOG_PATH = Path(config("OPENAI_LOG_PATH"))
            self.OPENAI_MODEL_COST_PER_INPUT_TOKEN = Decimal(
                config("OPENAI_MODEL_COST_PER_INPUT_TOKEN")
            )
            self.OPENAI_MODEL_COST_PER_CACHE_INPUT_TOKEN = Decimal(
                config("OPENAI_MODEL_COST_PER_CACHE_INPUT_TOKEN")
            )
            self.OPENAI_MODEL_COST_PER_OUTPUT_TOKEN = Decimal(
                config("OPENAI_MODEL_COST_PER_OUTPUT_TOKEN")
            )
            self.OPENAI_MINIMUM_BALANCE = Decimal(config("OPENAI_MINIMUM_BALANCE"))
            self.openai_budget = None
            self.openai_minute_history = []
            self.openai_tokens_minute_history = 0

            self.OPENAI_LOG_PATH.touch()
            logging.info(
                f"ChatGPT model '{self.OPENAI_MODEL}' selected for prompting..."
            )

            # Anthropic (Claude)
            self.ANTHROPIC_CLIENT = anthropic.Anthropic(
                api_key=config("ANTHROPIC_API_KEY")
            )
            self.ANTHROPIC_MODEL = config("ANTHROPIC_MODEL")
            self.ANTHROPIC_LOG_PATH = Path(config("ANTHROPIC_LOG_PATH"))
            self.ANTHROPIC_REQUESTS_PER_MINUTE_LIMIT = int(
                config("ANTHROPIC_REQUESTS_PER_MINUTE_LIMIT")
            )
            self.ANTHROPIC_INPUT_TOKENS_PER_MINUTE_LIMIT = int(
                config("ANTHROPIC_INPUT_TOKENS_PER_MINUTE_LIMIT")
            )
            self.ANTHROPIC_OUTPUT_TOKENS_PER_MINUTE_LIMIT = int(
                config("ANTHROPIC_OUTPUT_TOKENS_PER_MINUTE_LIMIT")
            )
            self.ANTHROPIC_MODEL_COST_PER_INPUT_TOKEN = Decimal(
                config("ANTHROPIC_MODEL_COST_PER_INPUT_TOKEN")
            )
            self.ANTHROPIC_MODEL_COST_PER_CACHE_WRITE_TOKEN = Decimal(
                config("ANTHROPIC_MODEL_COST_PER_CACHE_WRITE_TOKEN")
            )
            self.ANTHROPIC_MODEL_COST_PER_CACHE_READ_TOKEN = Decimal(
                config("ANTHROPIC_MODEL_COST_PER_CACHE_READ_TOKEN")
            )
            self.ANTHROPIC_MODEL_COST_PER_OUTPUT_TOKEN = Decimal(
                config("ANTHROPIC_MODEL_COST_PER_OUTPUT_TOKEN")
            )
            self.ANTHROPIC_MINUMUM_BALANCE = Decimal(
                config("ANTHROPIC_MINUMUM_BALANCE")
            )
            self.anthropic_budget = None
            self.anthropic_minute_history = []
            self.anthropic_input_tokens_minute_history = 0
            self.anthropic_output_tokens_minute_history = 0

            self.ANTHROPIC_LOG_PATH.touch()
            logging.info(
                f"Claude model '{self.ANTHROPIC_MODEL}' selected for prompting..."
            )

            # Ollama (Microsoft's Phi, Deepseek R1)
            self.OLLAMA_CLIENT = ollama.Client(host=config("OLLAMA_API_ADDRESS"))
            self.OLLAMA_LOG_PATH = config("OLLAMA_LOG_PATH")
            self.OLLAMA_MODELS = [
                config("OLLAMA_PHI_MODEL"),
                config("OLLAMA_DEEPSEEK_MODEL"),
            ]

            self.OLLAMA_LOG_PATH.touch()
            logging.info(
                f"Ollama models {str(self.OLLAMA_MODELS)[1:-1]} selected for prompting..."
            )

            # Grok API configs
            self.GROK_CLIENT = OpenAI(
                api_key=config("GROK_API_KEY"), base_url=config("GROK_API_URL")
            )
            self.GROK_LOG_PATH = Path(config("GROK_LOG_PATH"))
            # self.GROK_PROMPTS_PER_SECOND_LIMIT = config("GROK_PROMPTS_PER_SECOND_LIMIT")
            self.GROK_MODEL = config("GROK_MODEL")
            self.GROK_MODEL_CONTEXT_LIMIT = int(config("GROK_MODEL_CONTEXT_LIMIT"))
            self.GROK_MODEL_COST_PER_INPUT_TOKEN = Decimal(
                config("GROK_MODEL_COST_PER_INPUT_TOKEN")
            )
            self.GROK_MODEL_COST_PER_OUTPUT_TOKEN = Decimal(
                config("GROK_MODEL_COST_PER_OUTPUT_TOKEN")
            )
            self.GROK_MINIMUM_BALANCE = Decimal(config("GROK_MINIMUM_BALANCE"))
            self.grok_budget = None

            self.GROK_LOG_PATH.touch()
            logging.info(
                f"Dumbfuck Elon's '{self.GROK_MODEL}' model selected for prompting..."
            )

            self.init_api_history()

            logging.info("Initialization done.")
        except Exception:
            logging.exception("Fatal error occurred in: promptAI.__init__")
            raise

    def init_api_history(self) -> None:
        """
        init_api_history: Initializes API call histories for all supported LLM providers.

        This method parses previous API call logs for Gemini, OpenAI, Anthropic, and Grok,
        rebuilding internal state for tracking request counts, token usage, and remaining budget.
        It ensures that current usage and limits are accurately maintained between script runs.

        Notes
        -----
        - Ollama does not enforce rate limits or budgeting, so its history is not initialized.

        Raises
        ------
        Exception
            If an error occurs during the extraction or initialization of any provider's API history.
        """
        try:
            self.init_api_history_gemini()
            self.init_api_history_openai()
            self.init_api_history_anthropic()
            # No restriction on Ollama; therefore, the historical context the logs store are not necessary
            self.init_api_history_grok()
        except Exception:
            logging.exception("Fatal error occurred in: promptAI.init_api_history")
            raise

    def init_api_history_gemini(self):
        """
        init_api_history_gemini: Initializes Gemini API call history by parsing previous log entries.

        This method reads the Gemini log file in reverse chronological order and reconstructs:
        - Daily prompt usage (`gemini_day_history`)
        - Per-minute prompt usage (`gemini_minute_history`)
        - Tokens used in the past minute (`gemini_tokens_minute_history`)

        It ensures that the current state of Gemini API usage (requests and tokens) is consistent with
        historical usage and checks whether the daily API prompt limit has already been reached.

        Raises
        ------
        ValueError
            If the daily Gemini API request limit has already been reached based on parsed logs.
        Exception
            If any unexpected error occurs during log parsing or data processing.
        """
        try:
            logging.info("Gemini - Extracting API call history from logs...")

            with open(self.GEMINI_LOG_PATH, mode="r", encoding="UTF-8") as log:
                for line in reversed(log.readlines()):
                    record = line.split(" - ")
                    if record[1] == "INFO" and record[2] == "Gemini":
                        metadata = json.loads(record[3].strip())
                        metadata["timestamp"] = datetime.fromisoformat(
                            metadata["timestamp"]
                        )

                        if datetime.now() < metadata["timestamp"] + timedelta(days=1):
                            self.gemini_day_history.append(metadata)
                        else:
                            break

                        if datetime.now() < metadata["timestamp"] + timedelta(
                            minutes=1
                        ):
                            self.gemini_minute_history.append(metadata)
                            self.gemini_tokens_minute_history += metadata[
                                "total_token_count"
                            ]

            if len(self.gemini_day_history) < self.GEMINI_REQUESTS_PER_DAY_LIMIT:
                api_logger.warning(
                    f"Gemini - Starting daily prompt usage: {len(self.gemini_day_history)} / {self.GEMINI_REQUESTS_PER_DAY_LIMIT} requests used."
                )
            else:
                raise ValueError(
                    f"Gemini - Daily API prompt limit reached: {len(self.gemini_day_history)} / {self.GEMINI_REQUESTS_PER_DAY_LIMIT}."
                )

            api_logger.warning(
                f"Gemini - Starting minute history: {len(self.gemini_minute_history)} / {self.GEMINI_REQUESTS_PER_MINUTE_LIMIT} prompts used in the past minute."
            )
            api_logger.warning(
                f"Gemini - Starting minute history: {self.gemini_tokens_minute_history} / {self.GEMINI_TOKENS_PER_MINUTE} tokens used in the past minute."
            )
            logging.info("Gemini - Finished extracting API call history from logs.")

        except ValueError as ve:
            api_logger.error(f"{ve}")
            raise

        except Exception as e:
            logging.exception(
                f"Fatal error occurred in: promptAI.init_api_history_gemini on line {e.__traceback__.tb_lineno}"
            )
            raise

    def prompt_gemini(self, prompt: str) -> str:
        """
        prompt_gemini: Sends a text prompt to the Gemini model and returns the generated response.

        This method manages Gemini API rate limits (per-minute and per-day), token usage restrictions,
        and automatic waiting if necessary. It logs prompt metadata, updates historical usage records,
        and enforces budget/rate restrictions before sending the prompt.

        Parameters
        ----------
        prompt : str
            The textual content to send to the Gemini model for code generation or instruction following.

        Returns
        -------
        str
            The textual response generated by the Gemini model.

        Raises
        ------
        ValueError
            If the daily API prompt limit has been exceeded and no further requests are allowed.
        Exception
            If an unexpected error occurs during the prompting process or response handling.
        """
        try:
            logging.info(f"Prompting Gemini model '{self.GEMINI_MODEL_NAME}'...")

            while len(
                self.gemini_day_history
            ) > 0 and datetime.now() >= self.gemini_day_history[0][
                "timestamp"
            ] + timedelta(
                days=1
            ):
                self.gemini_day_history.pop(0)

            api_logger.warning(
                f"Gemini - Current day history: {len(self.gemini_day_history)} / {self.GEMINI_REQUESTS_PER_DAY_LIMIT} prompts used in the past day."
            )
            api_logger.warning(
                f"Gemini - Current minute history: {len(self.gemini_minute_history)} / {self.GEMINI_REQUESTS_PER_MINUTE_LIMIT} prompts used in the past minute."
            )
            api_logger.warning(
                f"Gemini - Current tokens/minute history: {self.gemini_tokens_minute_history} / {self.GEMINI_TOKENS_PER_MINUTE} tokens used in the past minute."
            )

            if len(self.gemini_day_history) >= self.GEMINI_REQUESTS_PER_DAY_LIMIT:
                raise ValueError(
                    f"Gemini - Daily API prompt limit reached: {len(self.gemini_day_history)} / {self.GEMINI_REQUESTS_PER_DAY_LIMIT}."
                )

            while len(
                self.gemini_minute_history
            ) > 0 and datetime.now() >= self.gemini_minute_history[0][
                "timestamp"
            ] + timedelta(
                minutes=1
            ):
                self.gemini_tokens_minute_history -= self.gemini_minute_history[0][
                    "total_token_count"
                ]
                self.gemini_minute_history.pop(0)

            if len(self.gemini_minute_history) >= self.GEMINI_REQUESTS_PER_MINUTE_LIMIT:
                wait_time = max(
                    0,
                    (
                        self.gemini_minute_history[0]["timestamp"]
                        + timedelta(minutes=1)
                        - datetime.now()
                    ).total_seconds(),
                )

                if wait_time > 0:
                    logging.warning(
                        f"Gemini prompts/minute limit reached: {len(self.gemini_minute_history)}. Waiting {wait_time} seconds before next prompt..."
                    )
                    time.sleep(wait_time)

                while len(self.gemini_minute_history) > 0 and datetime.now() >= (
                    self.gemini_minute_history[0]["timestamp"] + timedelta(minutes=1)
                ):
                    self.gemini_tokens_minute_history -= self.gemini_minute_history[0][
                        "total_token_count"
                    ]
                    self.gemini_minute_history.pop(0)

            if self.gemini_tokens_minute_history >= self.GEMINI_TOKENS_PER_MINUTE:
                logging.warning(
                    "Gemini tokens/minute limit reached. Waiting 60 seconds before next prompt..."
                )
                time.sleep(60)
                self.gemini_tokens_minute_history = 0
                self.gemini_minute_history.clear()

            response = self.GEMINI_MODEL.generate_content(prompt)

            metadata = {
                "finish_reason": promptAI.GEMINI_FINISH_REASON[
                    response.candidates[0].finish_reason
                ],
                "timestamp": datetime.now(),
                "avg_logprobs": response.candidates[0].avg_logprobs,
                "total_token_count": response.usage_metadata.total_token_count,
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
                "cached_content_token_count": response.usage_metadata.cached_content_token_count,
                "model_version": response.model_version,
            }

            self.gemini_minute_history.append(metadata)
            self.gemini_day_history.append(metadata)
            self.gemini_tokens_minute_history += (
                response.usage_metadata.total_token_count
            )

            log_metadata = metadata.copy()
            log_metadata["timestamp"] = log_metadata["timestamp"].isoformat()

            api_logger.info(f"Gemini - {json.dumps(log_metadata)}")

            return str(response.text)

        except ValueError as ve:
            api_logger.error(f"{ve}")
            raise

        except Exception as e:
            api_logger.error(
                f"Gemini - Error encountered during API call on line {e.__traceback__.tb_lineno}."
            )
            raise

    def init_api_history_openai(self):
        """
        init_api_history_openai: Initializes OpenAI (ChatGPT) API call history by parsing previous log entries.

        This method reads the OpenAI log file in reverse chronological order to reconstruct:
        - Per-minute prompt usage (`openai_minute_history`)
        - Tokens used in the past minute (`openai_tokens_minute_history`)
        - Remaining API budget (`openai_budget`)

        It ensures that API rate limits, token usage, and available funds are correctly reflected
        based on prior API activity before new prompts are sent.

        Raises
        ------
        ValueError
            If the available OpenAI budget is below the configured minimum threshold.
        Exception
            If an unexpected error occurs during log parsing or budget initialization.
        """
        try:
            logging.info("ChatGPT - Extracting API call history from logs...")

            with open(self.OPENAI_LOG_PATH, mode="r", encoding="UTF-8") as log:
                for line in reversed(log.readlines()):
                    record = line.split(" - ")
                    if record[1] == "INFO" and record[2] == "ChatGPT":
                        metadata = json.loads(record[3].strip())
                        metadata["timestamp"] = datetime.fromisoformat(
                            metadata["timestamp"]
                        )
                        metadata["remaining_budget"] = Decimal(
                            metadata["remaining_budget"]
                        )

                        if self.openai_budget is None:
                            self.openai_budget = metadata["remaining_budget"]

                            if self.openai_budget <= self.OPENAI_MINIMUM_BALANCE:
                                raise ValueError(
                                    f"ChatGPT - balance below minimum threshold: {self.openai_budget} / {self.OPENAI_MINIMUM_BALANCE}"
                                )

                        if datetime.now() < metadata["timestamp"] + timedelta(
                            minutes=1
                        ):
                            self.openai_minute_history.append(metadata)
                            self.openai_tokens_minute_history += metadata[
                                "total_token_count"
                            ]
                        else:
                            break

            api_logger.warning(
                f"ChatGPT - Starting minute history: {len(self.openai_minute_history)} / {self.OPENAI_REQUESTS_PER_MINUTE_LIMIT} prompts used in the past minute."
            )
            api_logger.warning(
                f"ChatGPT - Starting minute history: {self.openai_tokens_minute_history} / {self.OPENAI_TOKENS_PER_MINUTE_LIMIT} tokens used in the past minute."
            )
            api_logger.warning(
                f"ChatGPT - Starting remaining balance: $ {self.openai_budget}"
            )
            logging.info("ChatGPT - Finished extracting API call history from logs.")

        except ValueError as ve:
            api_logger.error(f"{ve}")
            raise

        except Exception as e:
            logging.exception(
                f"Fatal error occurred in: promptAI.init_api_history_openai on line {e.__traceback__.tb_lineno}"
            )
            raise

    def prompt_chatgpt(self, prompt: list[dict[str, str]]) -> str:
        """
        prompt_chatgpt: Sends a structured conversation prompt to the ChatGPT model and returns the generated response.

        This method manages OpenAI API rate limits (per-minute), token usage restrictions,
        and budget constraints before dispatching the prompt. It logs prompt metadata,
        updates internal usage tracking, and ensures that remaining credits are within safe thresholds.

        Parameters
        ----------
        prompt : list[dict[str, str]]
            A list of message dictionaries representing a conversation history,
            where each dictionary must have 'role' (e.g., "user", "assistant") and 'content' fields.

        Returns
        -------
        str
            The textual content of the response generated by the ChatGPT model.

        Raises
        ------
        ValueError
            If the available OpenAI budget is below the configured minimum threshold.
        Exception
            If any unexpected error occurs during the API call or response processing.
        """
        try:
            logging.info(
                f"Prompting OpenAI model '{self.OPENAI_MODEL}' with {len(prompt)} message(s)..."
            )

            while len(
                self.openai_minute_history
            ) > 0 and datetime.now() >= self.openai_minute_history[0][
                "timestamp"
            ] + timedelta(
                minutes=1
            ):
                self.openai_tokens_minute_history -= self.openai_minute_history[0][
                    "total_token_count"
                ]
                self.openai_minute_history.pop(0)

            if self.openai_budget <= self.OPENAI_MINIMUM_BALANCE:
                raise ValueError(
                    f"ChatGPT - balance below minimum threshold: {self.openai_budget} / {self.OPENAI_MINIMUM_BALANCE}"
                )

            api_logger.warning(
                f"ChatGPT - Current minute history: {len(self.openai_minute_history)} / {self.OPENAI_REQUESTS_PER_MINUTE_LIMIT} prompts used in the past minute."
            )
            api_logger.warning(
                f"ChatGPT - Current minute history: {self.openai_tokens_minute_history} / {self.OPENAI_TOKENS_PER_MINUTE_LIMIT} tokens used in the past minute."
            )
            api_logger.warning(
                f"ChatGPT - Current remaining balance: $ {self.openai_budget}"
            )

            if len(self.openai_minute_history) == self.OPENAI_REQUESTS_PER_MINUTE_LIMIT:
                wait_time = (
                    self.openai_minute_history[0]["timestamp"]
                    + timedelta(minutes=1)
                    - datetime.now()
                )
                logging.warning(
                    f"ChatGPT prompts/minute limit reached. Waiting {wait_time} seconds before next prompt..."
                )
                time.sleep(wait_time)
                self.openai_tokens_minute_history -= self.openai_minute_history[0][
                    "total_token_count"
                ]
                self.openai_minute_history.pop(0)

            if (
                self.openai_tokens_minute_history
                >= self.OPENAI_TOKENS_PER_MINUTE_LIMIT * 0.9
            ):
                logging.warning(
                    "ChatGPT tokens/minute limit reached. Waiting 60 seconds before next prompt..."
                )
                time.sleep(60)
                self.openai_tokens_minute_history = 0
                self.openai_minute_history.clear()

            response = self.OPENAI_CLIENT.chat.completions.create(
                model=self.OPENAI_MODEL, messages=prompt
            ).model_dump()

            self.openai_budget -= (
                self.OPENAI_MODEL_COST_PER_INPUT_TOKEN
                * response["usage"]["prompt_tokens"]
                + self.OPENAI_MODEL_COST_PER_CACHE_INPUT_TOKEN
                * response["usage"]["prompt_tokens_details"]["cached_tokens"]
                + self.OPENAI_MODEL_COST_PER_OUTPUT_TOKEN
                * response["usage"]["completion_tokens"]
            )

            metadata = {
                "finish_reason": response["choices"][0]["finish_reason"],
                "timestamp": datetime.fromtimestamp(response["created"]),
                "total_token_count": response["usage"]["total_tokens"],
                "prompt_token_count": response["usage"]["prompt_tokens"],
                "candidates_token_count": response["usage"]["completion_tokens"],
                "cached_content_token_count": response["usage"][
                    "prompt_tokens_details"
                ]["cached_tokens"],
                "model_version": response["model"],
                "remaining_budget": self.openai_budget,
            }

            self.openai_minute_history.append(metadata)
            self.openai_tokens_minute_history += metadata["total_token_count"]

            log_metadata = metadata.copy()
            log_metadata["timestamp"] = log_metadata["timestamp"].isoformat()
            log_metadata["remaining_budget"] = str(log_metadata["remaining_budget"])

            api_logger.info(f"ChatGPT - {json.dumps(log_metadata)}")

            return response["choices"][0]["message"]["content"]

        except ValueError as ve:
            api_logger.error(f"{ve}")
            raise

        except Exception as e:
            api_logger.error(
                f"ChatGPT - Error encountered during API call on line {e.__traceback__.tb_lineno}."
            )
            raise

    def init_api_history_anthropic(self) -> None:
        """
        init_api_history_anthropic: Initializes Anthropic (Claude) API call history by parsing previous log entries.

        This method reads the Claude-specific log file in reverse chronological order to rebuild:
        - Per-minute prompt usage (`anthropic_minute_history`)
        - Input token usage history (`anthropic_input_tokens_minute_history`)
        - Output token usage history (`anthropic_output_tokens_minute_history`)
        - Remaining API budget (`anthropic_budget`)

        It ensures that API rate limits, token usage, and available budget are accurately reflected
        based on previous activity before allowing new prompts to be sent.

        Raises
        ------
        ValueError
            If the available Anthropic budget is below the configured minimum threshold.
        Exception
            If an unexpected error occurs during log parsing or data initialization.
        """
        try:
            logging.info("Claude - Extracting API call history from logs...")

            with open(self.ANTHROPIC_LOG_PATH, mode="r", encoding="UTF-8") as log:
                for line in reversed(log.readlines()):
                    record = line.split(" - ")
                    if record[1] == "INFO" and record[2] == "Claude":
                        metadata = json.loads(record[3].strip())
                        metadata["timestamp"] = datetime.fromisoformat(
                            metadata["timestamp"]
                        )
                        metadata["remaining_budget"] = Decimal(
                            metadata["remaining_budget"]
                        )

                        if self.anthropic_budget is None:
                            self.anthropic_budget = metadata["remaining_budget"]

                            if self.anthropic_budget <= self.ANTHROPIC_MINUMUM_BALANCE:
                                raise ValueError(
                                    f"Anthropic balance below minimum threshold: {self.anthropic_budget} / {self.ANTHROPIC_MINUMUM_BALANCE}"
                                )

                        if datetime.now() < metadata["timestamp"] + timedelta(
                            minutes=1
                        ):
                            self.anthropic_minute_history.append(metadata)
                            self.anthropic_input_tokens_minute_history += metadata[
                                "prompt_token_count"
                            ]
                            self.anthropic_output_tokens_minute_history += metadata[
                                "candidates_token_count"
                            ]
                        else:
                            break

            api_logger.warning(
                f"Claude - Starting minute history: {len(self.anthropic_minute_history)} / {self.ANTHROPIC_REQUESTS_PER_MINUTE_LIMIT} prompts used in the past minute."
            )
            api_logger.warning(
                f"Claude - Starting minute history: {self.anthropic_input_tokens_minute_history} / {self.ANTHROPIC_INPUT_TOKENS_PER_MINUTE_LIMIT} input tokens used in the past minute."
            )
            api_logger.warning(
                f"Claude - Starting minute history: {self.anthropic_output_tokens_minute_history} / {self.ANTHROPIC_OUTPUT_TOKENS_PER_MINUTE_LIMIT} output tokens used in the past minute."
            )
            api_logger.warning(
                f"Claude - Starting remaining balance: $ {self.anthropic_budget}"
            )

            logging.info("Claude - Finished extracting API call history from logs.")

        except ValueError as ve:
            api_logger.error(f"{ve}")
            raise

        except Exception as e:
            logging.exception(
                f"Fatal error occurred in: promptAI.init_api_history_anthropic on line {e.__traceback__.tb_lineno}"
            )
            raise

    def prompt_anthropic(
        self, prompt: list[dict[str, str]], system_instruction: str = ""
    ) -> str:
        """
        prompt_anthropic: Sends a structured conversation prompt to the Claude model and returns the generated response.

        This method manages Anthropic API rate limits (per-minute), input/output token usage restrictions,
        and budget constraints. It enforces waiting periods if limits are reached, logs prompt metadata,
        and updates internal usage tracking based on the model's response.

        Parameters
        ----------
        prompt : list[dict[str, str]]
            A list of message dictionaries representing the conversation history, where each dictionary
            contains 'role' (e.g., "user", "assistant") and 'content' fields.
        system_instruction : str, optional
            An optional system-level instruction to guide the Claude model's behavior during the conversation,
            by default "" (no special instruction).

        Returns
        -------
        str
            The textual content generated by the Claude model in response to the provided conversation.

        Raises
        ------
        ValueError
            If the available Anthropic budget is below the configured minimum threshold.
        Exception
            If an unexpected error occurs during the API call or response handling.
        """
        try:
            logging.info(
                f"Prompting Anthropic model '{self.ANTHROPIC_MODEL}' with {len(prompt)} message(s)..."
            )

            while len(
                self.anthropic_minute_history
            ) > 0 and datetime.now() >= self.anthropic_minute_history[0][
                "timestamp"
            ] + timedelta(
                minutes=1
            ):
                self.anthropic_input_tokens_minute_history -= (
                    self.anthropic_minute_history[0]["prompt_token_count"]
                )
                self.anthropic_output_tokens_minute_history -= (
                    self.anthropic_minute_history[0]["candidates_token_count"]
                )
                self.anthropic_minute_history.pop(0)

            if self.anthropic_budget <= self.OPENAI_MINIMUM_BALANCE:
                raise ValueError(
                    f"Claude - balance below minimum threshold: {self.anthropic_budget} / {self.ANTHROPIC_MINUMUM_BALANCE}"
                )

            api_logger.warning(
                f"Claude - Current minute history: {len(self.anthropic_minute_history)} / {self.ANTHROPIC_REQUESTS_PER_MINUTE_LIMIT} prompts used in the past minute."
            )
            api_logger.warning(
                f"Claude - Current minute history: {self.anthropic_input_tokens_minute_history} / {self.ANTHROPIC_INPUT_TOKENS_PER_MINUTE_LIMIT} input tokens used in the past minute."
            )
            api_logger.warning(
                f"Claude - Current minute history: {self.anthropic_output_tokens_minute_history} / {self.ANTHROPIC_OUTPUT_TOKENS_PER_MINUTE_LIMIT} output tokens used in the past minute."
            )
            api_logger.warning(
                f"Claude - Current remaining balance: $ {self.anthropic_budget}"
            )

            if (
                len(self.anthropic_minute_history)
                >= self.ANTHROPIC_REQUESTS_PER_MINUTE_LIMIT
            ):
                wait_time = (
                    self.anthropic_minute_history[0]["timestamp"]
                    + timedelta(minutes=1)
                    - datetime.now()
                )
                logging.warning(
                    f"Claude prompts/minute limit reached. Waiting {wait_time} seconds before next prompt..."
                )
                time.sleep(wait_time)
                self.anthropic_input_tokens_minute_history -= (
                    self.anthropic_minute_history[0]["prompt_token_count"]
                )
                self.anthropic_output_tokens_minute_history -= (
                    self.anthropic_minute_history[0]["candidates_token_count"]
                )
                self.anthropic_minute_history.pop(0)

            while (
                self.anthropic_input_tokens_minute_history
                >= self.ANTHROPIC_INPUT_TOKENS_PER_MINUTE_LIMIT * 0.9
                or self.anthropic_output_tokens_minute_history
                >= self.ANTHROPIC_OUTPUT_TOKENS_PER_MINUTE_LIMIT * 0.9
            ):
                wait_time = (
                    self.anthropic_minute_history[0]["timestamp"]
                    + timedelta(minutes=1)
                    - datetime.now()
                )
                logging.warning(
                    f"Claude tokens/minute limit reached. Waiting {wait_time} seconds before next prompt..."
                )
                time.sleep(wait_time)
                self.anthropic_input_tokens_minute_history -= (
                    self.anthropic_minute_history[0]["prompt_token_count"]
                )
                self.anthropic_output_tokens_minute_history -= (
                    self.anthropic_minute_history[0]["candidates_token_count"]
                )
                self.anthropic_minute_history.pop(0)

            timestamp = datetime.now().replace(microsecond=0)

            response = self.ANTHROPIC_CLIENT.messages.create(
                model=config("ANTHROPIC_MODEL"),
                system=system_instruction,
                max_tokens=int(config("ANTHROPIC_OUTPUT_TOKENS_PER_MINUTE_LIMIT")),
                messages=prompt,
            ).model_dump()

            self.anthropic_budget -= (
                self.ANTHROPIC_MODEL_COST_PER_INPUT_TOKEN
                * response["usage"]["input_tokens"]
                + self.ANTHROPIC_MODEL_COST_PER_CACHE_WRITE_TOKEN
                * response["usage"]["cache_creation_input_tokens"]
                + self.ANTHROPIC_MODEL_COST_PER_CACHE_READ_TOKEN
                * response["usage"]["cache_read_input_tokens"]
                + self.ANTHROPIC_MODEL_COST_PER_OUTPUT_TOKEN
                * response["usage"]["output_tokens"]
            )

            metadata = {
                "finish_reason": response["stop_reason"],
                "timestamp": timestamp,
                "total_token_count": response["usage"]["input_tokens"]
                + response["usage"]["output_tokens"],
                "prompt_token_count": response["usage"]["input_tokens"],
                "candidates_token_count": response["usage"]["output_tokens"],
                "written_cached_content_token_count": response["usage"][
                    "cache_creation_input_tokens"
                ],
                "read_cached_content_token_count": response["usage"][
                    "cache_read_input_tokens"
                ],
                "model_version": response["model"],
                "remaining_budget": self.anthropic_budget,
            }

            self.anthropic_minute_history.append(metadata)
            self.anthropic_input_tokens_minute_history += metadata["prompt_token_count"]
            self.anthropic_output_tokens_minute_history += metadata[
                "candidates_token_count"
            ]

            log_metadata = metadata.copy()
            log_metadata["timestamp"] = log_metadata["timestamp"].isoformat()
            log_metadata["remaining_budget"] = str(log_metadata["remaining_budget"])

            api_logger.info(f"Claude - {json.dumps(log_metadata)}")

            return response["content"][0]["text"]

        except ValueError as ve:
            api_logger.error(f"{ve}")
            raise

        except Exception as e:
            api_logger.error(
                f"Claude - Error encountered during API call on line {e.__traceback__.tb_lineno}."
            )
            raise

    def prompt_ollama(self, prompt: list[dict[str, str]], ollama_model: str) -> str:
        """
        prompt_ollama: Sends a structured conversation prompt to a specified Ollama model and returns the generated response.

        This method dispatches the conversation history to a local Ollama model, logs response metadata,
        and post-processes the generated text to remove any "<think>" tags if present.

        Parameters
        ----------
        prompt : list[dict[str, str]]
            A list of message dictionaries representing the conversation history, where each dictionary
            contains 'role' (e.g., "user", "assistant") and 'content' fields.
        ollama_model : str
            The name of the Ollama model to which the prompt will be sent (e.g., "phi", "deepseek").

        Returns
        -------
        str
            The textual response generated by the Ollama model.

        Raises
        ------
        Exception
            If an unexpected error occurs during the API call or response processing.
        """
        try:
            logging.info(
                f"Prompting Ollama model '{ollama_model}' with {len(prompt)} message(s)..."
            )

            timestamp = datetime.now().replace(microsecond=0)
            response = self.OLLAMA_CLIENT.chat(
                model=ollama_model, messages=prompt
            ).model_dump()

            metadata = {
                "finish_reason": response["done_reason"],
                "timestamp": timestamp,
                "total_token_count": response["prompt_eval_count"]
                + response["eval_count"],
                "prompt_token_count": response["prompt_eval_count"],
                "candidates_token_count": response["eval_count"],
                "model_version": response["model"],
            }

            log_metadata = metadata.copy()
            log_metadata["timestamp"] = log_metadata["timestamp"].isoformat()

            api_logger.info(f"Ollama - {json.dumps(log_metadata)}")

            response_text = response["message"]["content"]

            if response_text.find("<think>") != -1:
                response_text = response_text.replace(
                    response_text[
                        response_text.find("<think>") : response_text.find("</think>")
                        + 10
                    ],
                    "",
                )

            return response_text

        except Exception as e:
            api_logger.error(
                f"Ollama - Error encountered during API call on line {e.__traceback__.tb_lineno}."
            )
            raise

    def init_api_history_grok(self):
        """
        init_api_history_grok: Initializes Grok API call history by parsing previous log entries.

        This method reads the Grok-specific log file in reverse chronological order
        to reconstruct the remaining API budget (`grok_budget`).
        It ensures that the available funds are accurately tracked based on prior activity
        before allowing new prompts to be sent.

        Raises
        ------
        ValueError
            If the available Grok budget is below the configured minimum threshold.
        Exception
            If an unexpected error occurs during log parsing or budget initialization.
        """
        try:
            logging.info("Grok - Extracting API call history from logs...")
            with open(self.GROK_LOG_PATH, mode="r", encoding="UTF-8") as log:
                for line in reversed(log.readlines()):
                    record = line.split(" - ")
                    if record[1] == "INFO" and record[2] == "Grok":
                        metadata = json.loads(record[3].strip())
                        metadata["timestamp"] = datetime.fromisoformat(
                            metadata["timestamp"]
                        )
                        metadata["remaining_budget"] = Decimal(
                            metadata["remaining_budget"]
                        )

                        if self.grok_budget is None:
                            self.grok_budget = Decimal(metadata["remaining_budget"])

                            if self.grok_budget <= self.GROK_MINIMUM_BALANCE:
                                raise ValueError(
                                    f"Grok - balance below minimum threshold: {self.grok_budget} / {self.GROK_MINIMUM_BALANCE}"
                                )

                        break

            api_logger.warning(
                f"Grok - Starting remaining balance: $ {self.grok_budget}"
            )
            logging.info("Grok - Finished extracting API call history from logs.")

        except ValueError as ve:
            api_logger.error(f"{ve}")
            raise

        except Exception as e:
            logging.exception(
                f"Fatal error occurred in: promptAI.init_api_history_grok on line {e.__traceback__.tb_lineno}"
            )
            raise

    def prompt_grok(self, prompt: list[dict[str, str]]) -> str:
        """
        prompt_grok: Sends a structured conversation prompt to the Grok model and returns the generated response.

        This method manages Grok API budget constraints, tracks token usage,
        monitors for context length warnings, and logs prompt metadata.
        It enforces budget limits before dispatching the prompt to prevent unauthorized spending.

        Parameters
        ----------
        prompt : list[dict[str, str]]
            A list of message dictionaries representing the conversation history,
            where each dictionary contains 'role' (e.g., "user", "assistant") and 'content' fields.

        Returns
        -------
        str
            The textual response generated by the Grok model.

        Raises
        ------
        ValueError
            If the available Grok budget is below the configured minimum threshold.
        Exception
            If any unexpected error occurs during the API call or response handling.
        """
        try:
            logging.info(
                f"Prompting Grok model '{self.GROK_MODEL}' with {len(prompt)} message(s)..."
            )

            if self.grok_budget <= self.GROK_MINIMUM_BALANCE:
                raise ValueError(
                    f"Grok - balance below minimum threshold: {self.grok_budget} / {self.GROK_MINIMUM_BALANCE}"
                )

            api_logger.warning(
                f"Grok - Current remaining balance: $ {self.grok_budget}"
            )

            response = self.GROK_CLIENT.chat.completions.create(
                model=self.GROK_MODEL, messages=prompt
            ).model_dump()

            self.grok_budget -= (
                self.GROK_MODEL_COST_PER_INPUT_TOKEN
                * response["usage"]["prompt_tokens"]
                + self.GROK_MODEL_COST_PER_OUTPUT_TOKEN
                * response["usage"]["completion_tokens"]
            )

            metadata = {
                "finish_reason": response["choices"][0]["finish_reason"],
                "timestamp": datetime.fromtimestamp(response["created"]),
                "total_token_count": response["usage"]["total_tokens"],
                "prompt_token_count": response["usage"]["prompt_tokens"],
                "candidates_token_count": response["usage"]["completion_tokens"],
                "model_version": response["model"],
                "remaining_budget": self.grok_budget,
            }

            log_metadata = metadata.copy()
            log_metadata["timestamp"] = log_metadata["timestamp"].isoformat()
            log_metadata["remaining_budget"] = str(log_metadata["remaining_budget"])

            api_logger.info(f"Grok - {json.dumps(log_metadata)}")

            if metadata["total_token_count"] >= self.GROK_MODEL_CONTEXT_LIMIT * 0.9:
                api_logger.warning(
                    f"Grok - Approaching context limit per prompt: {metadata["total_token_count"]} / {self.GROK_MODEL_CONTEXT_LIMIT}"
                )
            elif metadata["total_token_count"] >= self.GROK_MODEL_CONTEXT_LIMIT * 0.9:
                api_logger.error(
                    f"Grok - Context limit per prompt reached: {metadata["total_token_count"]} / {self.GROK_MODEL_CONTEXT_LIMIT}"
                )

            return response["choices"][0]["message"]["content"]

        except ValueError as e:
            api_logger.error(f"{e}")
            raise

        except Exception as e:
            api_logger.error(
                f"Grok - Error encountered during API call on line {e.__traceback__.tb_lineno}."
            )
            raise


class genAI:
    """
    GenAI: Gen the AI.

    Gen the AI.
    """

    BASE_PATH = Path(config("ASSIGNMENTS_BASE_PATH"))

    def __init__(self, cmd_args: list[str]):
        """
        __init__: Initializes the `genAI` class and orchestrates the full AI code generation workflow.

        This constructor:
        - Initializes the `promptAI` manager for interacting with multiple LLM APIs.
        - Parses the assignment ID from command-line arguments.
        - Queries the database for assignment metadata.
        - Loads and processes the assignment PDF for each model (Gemini, ChatGPT, Claude, Ollama, Grok).
        - Prompts each model twice (processed and raw versions) to generate code submissions.
        - Tracks and logs final API budget usage across providers.

        Parameters
        ----------
        cmd_args : list[str]
            A list of command-line arguments where the assignment ID is expected to be included.
        """
        logging.info("Initializing AI code generation process...")

        self.promptAI = promptAI()
        self.raw_pdf = None

        logging.info("Parsing command-line arguments to extract assignment ID...")
        self.get_assignment_id_from_args(cmd_args)

        logging.info(
            f"Querying database for assignment with ID: {self.assignment_id}..."
        )
        self.query_assignments_assignments()

        logging.info("Loading assignment PDF and preparing for Gemini prompting...")
        self.gemini_process_assignment_pdf()

        logging.info("Prompting Gemini to generate code...")
        self.gemini_generate_code()

        logging.info("Loading assignment PDF and preparing for ChatGPT prompting...")
        self.openai_process_assignment_pdf()

        logging.info("Prompting ChatGPT to generate code...")
        self.openai_generate_code()

        logging.info("Loading assignment PDF and preparing for Claude prompting...")
        self.anthropic_process_assignment_pdf()

        logging.info("Prompting Claude to generate code...")
        self.anthropic_generate_code()

        for count, model in enumerate(self.promptAI.OLLAMA_MODELS):
            logging.info(
                f"Prompting Ollama model '{model}', ({count + 1} / {len(self.promptAI.OLLAMA_MODELS)})"
            )

            logging.info("Loading assignment PDF and preparing for Ollama prompting...")
            self.ollama_process_assignment_pdf(model)

            logging.info("Prompting Ollama to generate code...")
            self.ollama_generate_code(model)

        logging.info("Loading assignment PDF and preparing for Grok prompting...")
        self.grok_process_assignment_pdf()

        logging.info("Prompting Grok to generate code...")
        self.grok_generate_code()

        logging.warning(
            f"ChatGPT - Ending budget: $ {self.promptAI.openai_budget if self.promptAI.openai_budget is not None else "NULL"}"
        )
        logging.warning(
            f"Claude - Ending budget: $ {self.promptAI.anthropic_budget if self.promptAI.anthropic_budget is not None else "NULL"}"
        )
        logging.warning(
            f"Grok - Ending budget: $ {self.promptAI.grok_budget if self.promptAI.grok_budget is not None else "NULL"}"
        )

        logging.info("AI code generation process completed successfully.")

    def get_current_scope(self) -> str:
        """
        get_current_scope: Returns the current class and method name for logging and debugging purposes.

        This method uses Python's runtime inspection tools to dynamically retrieve
        the class name and the name of the calling method, formatted as 'ClassName.method_name'.

        Returns
        -------
        str
            A string representing the current scope in the format 'ClassName.method_name'.

        Raises
        ------
        Exception
            If an unexpected error occurs while retrieving the scope information.
        """
        try:
            return f"{self.__class__.__name__}.{inspect.currentframe().f_back.f_code.co_name}"

        except Exception:
            logging.exception("Fatal error occurred in: genAI.get_current_scope")
            raise

    def get_assignment_id_from_args(self, args: list[str]) -> None:
        """
        get_assignment_id_from_args: Parses and validates the assignment ID from the command-line arguments.

        This method checks that exactly two command-line arguments are provided
        and that the second argument is a numeric assignment ID. If validation passes,
        the assignment ID is stored for later use; otherwise, a runtime error is raised.

        Parameters
        ----------
        args : list[str]
            A list of command-line arguments where the assignment ID is expected as the second element.

        Raises
        ------
        RuntimeError
            If the number of provided arguments is incorrect.
        RuntimeError
            If the assignment ID argument is not a valid integer.
        Exception
            If any unexpected error occurs during argument parsing or validation.
        """
        try:
            if len(args) != 2:
                raise RuntimeError(
                    f"Invalid number of command-line arguments: received {len(args)}, expected 2.\n"
                    "Usage: python3 prompt_AI.py <assignment_id>"
                )
            if not args[1].isnumeric():
                raise RuntimeError(
                    f"Invalid assignment ID: received '{args[1]}', expected an integer.\n"
                    "Usage: python3 prompt_AI.py <assignment_id>"
                )

            logging.info(
                "Successfully extracted assignment_id from command-line arguments."
            )
            self.assignment_id = int(args[1])

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def query_assignments_assignments(self, print_df: bool = False) -> None:
        """
        query_assignments_assignments: Queries the assignments database for metadata corresponding to the provided assignment ID.

        This method retrieves assignment metadata (assignment number, PDF filepath, base code status,
        AI directory path, and programming language) from the database and stores it as instance attributes.
        It validates that exactly one matching record exists and optionally prints the retrieved metadata.

        Parameters
        ----------
        print_df : bool, optional
            If True, prints the retrieved assignment metadata DataFrame to the console, by default False.

        Raises
        ------
        RuntimeError
            If no assignment is found matching the given assignment ID.
        RuntimeError
            If multiple assignments are found when only one is expected.
        Exception
            If any unexpected error occurs during the database query or metadata extraction.
        """
        try:
            columns = "assignment_number, pdf_filepath, has_base_code,  bulk_ai_directory_path, language"
            result = pd.read_sql(
                f"SELECT {columns} FROM assignments_assignments WHERE id = {self.assignment_id}",
                ENGINE,
            )

            if not len(result):
                raise RuntimeError(
                    f"Assignment_id '{self.assignment_id}' does not match any records in the database."
                )

            if len(result) != 1:
                raise RuntimeError(f"Expected exactly one row, got {len(result)}.")

            result = result.squeeze()
            result.name = "Assignment Metadata"

            self.assignment_number = int(result["assignment_number"])
            self.pdf_path = Path(result["pdf_filepath"])
            self.has_base_code = bool(result["has_base_code"])
            self.ai_dir_path = Path(result["bulk_ai_directory_path"])
            self.language = str(result["language"])

            if print_df:
                print(f"{result.name}:\n{result}")

            logging.info("Successfully extracted assignment metadata from database.")

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def gemini_process_assignment_pdf(self) -> None:
        """
        gemini_process_assignment_pdf: Loads and preprocesses the assignment PDF to create a cleaned prompt for the Gemini model.

        This method:
        - Loads the assignment PDF and converts it into markdown format using `pymupdf4llm`.
        - Constructs a structured XML-like prompt containing cleaning instructions and the raw PDF text.
        - Sends the prompt to the Gemini model to refine the markdown by cleaning artifacts,
        normalizing formatting, and removing unnecessary content.
        - Stores the cleaned and processed assignment text for subsequent code generation.

        Raises
        ------
        Exception
            If any unexpected error occurs during PDF loading, prompt building, or Gemini API prompting.
        """
        instructions = (
            "Please clean the following text to make it easier to extract information about a programming assignment.\n"
            "Specifically:\n"
            "-   Remove any redundant 'Contents' sections.\n"
            "-   Normalize whitespace: replace multiple spaces with single spaces, remove leading/trailing spaces, and reduce multiple blank lines to a maximum of two blank lines.\n"
            "-   Remove any leftover formatting characters or artifacts that don't belong in the instructions.\n"
            "-   Remove any unnecessary information such as background information that doesn't contribute the coding specifications.\n"
        )

        try:
            if self.raw_pdf is None:
                logging.info(
                    "Loading assignment PDF and converting to markdown using pymupdf4llm..."
                )
                self.raw_pdf = pymupdf4llm.to_markdown(self.pdf_path)

            logging.info("Building the prompt's content...")
            prompt_elem = etree.Element("prompt")

            instructions_elem = etree.SubElement(prompt_elem, "instructions")
            instructions_elem.text = instructions

            pdf_text_elem = etree.SubElement(prompt_elem, "PDF_text")
            pdf_text_elem.text = self.raw_pdf

            prompt = etree.tostring(
                prompt_elem, pretty_print=True, encoding="utf-8"
            ).decode("utf-8")

            logging.info("Prompting Gemini to refine markdown text...")
            self.processed_pdf = self.promptAI.prompt_gemini(prompt)[8:]

            logging.info(
                "Successfully loaded and prepared the assignment PDF's text for prompting."
            )

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def gemini_generate_code(self) -> None:
        """
        gemini_generate_code: Generates multiple AI code submissions using the Gemini model based on the assignment PDF and test cases.

        This method:
        - Constructs a structured prompt combining assignment instructions (processed and raw PDFs) and test cases.
        - Sends the prompt to the Gemini model multiple times to generate diverse code submissions.
        - Parses the response to extract the generated code.
        - Saves each generated program into organized output directories for later analysis or evaluation.

        The method repeats prompting using both a cleaned version of the assignment (processed PDF)
        and the original version (raw PDF), producing multiple independent generations for each.

        Raises
        ------
        Exception
            If any unexpected error occurs during prompt construction, model prompting, or file saving.
        """
        code_gen_prompt = (
            "Using the provided assignment description and test cases, "
            "generate a complete program that behaves exactly as specified. "
            "Each test case includes:\n"
            "- console_command: the command used to run the program\n"
            "- console_input: input provided during program execution\n"
            "- expected_console_output: expected output to the console\n"
            "- input_files: files that must be read during execution\n"
            "- output_files: files that must be created or written during execution\n\n"
            "Analyze the provided context carefully and synthesize an appropriate, complete program. "
            "Ensure that your code correctly handles all specified inputs and produces all required outputs. "
            "The program should be fully functional according to the test cases.\n\n"
            "Include clear and concise comments in your code to explain the purpose of each function, "
            "the logic behind complex sections, and any important variables. "
            "Use comments to enhance the readability and maintainability of the code."
        )

        try:
            ai_submission_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "ai_submissions"
            )
            test_cases_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "test_cases"
            )

            logging.info("Gathering context for prompt to Gemini...")

            for cycle in range(0, 2):
                logging.info(
                    f"Gemini - Prompting model with {"processed PDF"if cycle == 0 else "raw PDF"} to generate code..."
                )

                prompt_elem = etree.Element("prompt")

                instructions = etree.SubElement(prompt_elem, "instructions")
                instructions.text = (
                    f"You are an expert {self.language} programmer. " + code_gen_prompt
                )

                PDF_text = etree.SubElement(prompt_elem, "PDF")
                PDF_text.text = self.processed_pdf if cycle == 0 else self.raw_pdf

                if test_cases_path.exists() and test_cases_path.is_dir():
                    for iter, path in enumerate(test_cases_path.iterdir()):
                        test_case = etree.SubElement(
                            prompt_elem, f"test_case_{iter + 1}"
                        )

                        with open(
                            path / "console_command.txt", mode="r", encoding="UTF-8"
                        ) as file:
                            console_command = etree.SubElement(
                                test_case, "console_command"
                            )
                            console_command.text = file.read()

                        with open(
                            path / "console_input.txt", mode="r", encoding="UTF-8"
                        ) as file:
                            console_input = etree.SubElement(test_case, "console_input")
                            console_input.text = file.read()

                        with open(
                            path / "expected_console_output.txt",
                            mode="r",
                            encoding="UTF-8",
                        ) as file:
                            console_output = etree.SubElement(
                                test_case, "expected_console_output"
                            )
                            console_output.text = file.read()

                        if any((path / "input_files").iterdir()):
                            input_files = etree.SubElement(test_case, "input_files")
                            for input_path in (path / "input_files").iterdir():
                                with open(
                                    input_path, mode="r", encoding="UTF-8"
                                ) as file:
                                    input_file = etree.SubElement(
                                        input_files, f"{input_path.name}"
                                    )
                                    input_file.text = file.read()

                        if any((path / "output_files").iterdir()):
                            output_files = etree.SubElement(test_case, "output_files")
                            for output_path in (path / "output_files").iterdir():
                                with open(
                                    output_path, mode="r", encoding="UTF-8"
                                ) as file:
                                    output_file = etree.SubElement(
                                        output_files, f"{output_path.name}"
                                    )
                                    output_file.text = file.read()

                prompt = etree.tostring(
                    prompt_elem, pretty_print=True, encoding="utf-8"
                ).decode("utf-8")

                for count in range(1, 6):
                    logging.info(
                        f"Gemini - {"processed PDF" if cycle == 0 else "raw PDF"} - prompting cycle progress: {count} / 5"
                    )
                    output_path = (
                        ai_submission_path
                        / f"{self.promptAI.GEMINI_MODEL_NAME}_{"processed" if cycle == 0 else "raw"}_{count}"
                    )
                    output_path.mkdir(parents=True, exist_ok=True)
                    with open(
                        output_path / "main.cpp", mode="w", encoding="UTF-8"
                    ) as file:
                        program = self.promptAI.prompt_gemini(prompt)
                        index_opening = program.find("```")
                        if index_opening != -1:
                            index_shift = program.find("\n", index_opening)
                            program = program[
                                index_opening + 7 : program.find("```", index_shift)
                            ]
                        file.write(program)

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def openai_process_assignment_pdf(self) -> None:
        """
        openai_process_assignment_pdf: Loads and preprocesses the assignment PDF to create a cleaned prompt for the ChatGPT model.

        This method:
        - Loads the assignment PDF and converts it into markdown format using `pymupdf4llm`.
        - Constructs a conversation-style prompt with system instructions for cleaning the text.
        - Sends the prompt to ChatGPT to refine the assignment by removing irrelevant sections,
        normalizing formatting, and focusing on coding specifications.
        - Stores the cleaned and processed assignment text for subsequent code generation.

        Raises
        ------
        Exception
            If any unexpected error occurs during PDF loading, prompt building, or ChatGPT API prompting.
        """
        instructions = (
            "You are an expert assistant specialized in preparing programming assignment instructions for analysis.\n"
            "Your task is to clean and normalize the provided text according to the following guidelines:\n"
            "- Remove any redundant 'Contents' sections.\n"
            "- Normalize whitespace: use single spaces between words, remove leading and trailing spaces, and limit consecutive blank lines to a maximum of two.\n"
            "- Eliminate any leftover formatting artifacts or stray characters that do not contribute to the clarity of the instructions.\n"
            "- Remove background information or narrative sections that do not specify coding requirements.\n"
            "Focus on producing a clean, concise version of the assignment that emphasizes the coding specifications."
        )

        try:
            prompt = list()

            if self.raw_pdf is None:
                logging.info(
                    "Loading assignment PDF and converting to markdown using pymupdf4llm..."
                )
                self.raw_pdf = pymupdf4llm.to_markdown(self.pdf_path)

            logging.info("Building the prompt's content...")
            prompt.append({"role": "system", "content": instructions})
            prompt.append({"role": "user", "content": self.raw_pdf})

            logging.info("Prompting ChatGPT to refine markdown text...")
            self.processed_pdf = self.promptAI.prompt_chatgpt(prompt)

            logging.info(
                "Successfully loaded and prepared the assignment PDF's text for prompting."
            )

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def openai_generate_code(self) -> None:
        """
        openai_generate_codeGenerates multiple AI code submissions using the ChatGPT model based on the assignment PDF and test cases.

        This method:
        - Constructs a conversation-style prompt sequence including system instructions, assignment description,
        structured test cases, and a final code generation request.
        - Sends the prompt sequence to the ChatGPT model multiple times to generate diverse code submissions.
        - Parses the model's response to extract the generated code.
        - Saves each generated program into organized output directories for later analysis or evaluation.

        The method repeats prompting using both a cleaned version of the assignment (processed PDF)
        and the original version (raw PDF), producing multiple independent generations for each.

        Raises
        ------
        Exception
            If any unexpected error occurs during prompt construction, model prompting, or file saving.
        """
        system_message = (
            "You carefully review assignment descriptions and multiple structured test cases, "
            "including input files and expected output files. "
            f"You generate complete, correct, and well-commented {self.language} programs that meet all requirements."
        )

        code_gen_instructions = (
            f"You are an expert {self.language} programming assistant.\n"
            "Using the provided assignment description and test cases, "
            "generate a complete program that behaves exactly as specified.\n\n"
            "Each test case includes:\n"
            "- console_command: the command used to run the program\n"
            "- console_input: input provided during program execution\n"
            "- expected_console_output: expected output to the console\n"
            "- input_files: files that must be read during execution\n"
            "- output_files: files that must be created or written during execution\n\n"
            "Analyze all provided context carefully. Synthesize an appropriate, complete program. "
            "Ensure that your code correctly handles all specified inputs and produces all required outputs. "
            "Include clear and concise comments to explain the purpose of each function, "
            "the logic behind complex sections, and important variables."
            "Write clean and professional code, ensuring consistent indentation and formatting throughout the program."
        )

        code_gen_conclusion = (
            f"Please now generate the full {self.language} program based on the provided instructions, assignment description, and test cases."
            "Only output the complete source code — do not include any additional notes, explanations, or submission instructions outside of the program itself.\n"
            "Focus on writing clear, well-commented code that satisfies all provided test cases.\n\n"
            "If minor formatting inconsistencies appear in the sample outputs (such as spaces after colons or extra blank lines), prioritize producing consistent, clean formatting within your program rather than exactly replicating minor inconsistencies."
        )

        try:
            ai_submission_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "ai_submissions"
            )
            test_cases_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "test_cases"
            )

            logging.info("Gathering context for prompt to ChatGPT...")

            test_cases = dict()

            for iter, path in enumerate(test_cases_path.iterdir()):
                case = dict()

                with open(
                    path / "console_command.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["console_command"] = file.read()

                with open(
                    path / "console_input.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["console_input"] = file.read()

                with open(
                    path / "expected_console_output.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["expected_console_output.txt"] = file.read()

                if any((path / "input_files").iterdir()):
                    input_files = dict()
                    for input_path in (path / "input_files").iterdir():
                        with open(input_path, mode="r", encoding="UTF-8") as file:
                            input_files[input_path.name] = file.read()
                    case["input_files"] = input_files

                if any((path / "output_files").iterdir()):
                    output_files = dict()
                    for output_path in (path / "output_files").iterdir():
                        with open(output_path, mode="r", encoding="UTF-8") as file:
                            output_files[output_path.name] = file.read()
                    case["output_files"] = output_files

                test_cases[f"test_case_{iter}"] = case

            for cycle in range(0, 2):
                logging.info(
                    f"ChatGPT - Prompting model with {"processed PDF"if cycle == 0 else "raw PDF"} to generate code..."
                )
                prompt = list()

                prompt.append({"role": "system", "content": system_message})
                prompt.append({"role": "user", "content": code_gen_instructions})
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Instructions received. Ready for assignment description.",
                    }
                )
                prompt.append(
                    {
                        "role": "user",
                        "content": self.processed_pdf if cycle == 0 else self.raw_pdf,
                    }
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Assignment description received. Ready for test cases.",
                    }
                )
                prompt.append(
                    {"role": "user", "content": json.dumps(test_cases, indent=4)}
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Test cases received. Ready to generate code.",
                    }
                )
                prompt.append({"role": "user", "content": code_gen_conclusion})

                for count in range(1, 6):
                    logging.info(
                        f"ChatGPT - {"processed PDF" if cycle == 0 else "raw PDF"} - prompting cycle progress: {count} / 5"
                    )
                    output_path = (
                        ai_submission_path
                        / f"{self.promptAI.OPENAI_MODEL}_{"processed" if cycle == 0 else "raw"}_{count}"
                    )
                    output_path.mkdir(parents=True, exist_ok=True)
                    with open(
                        output_path / "main.cpp", mode="w", encoding="UTF-8"
                    ) as file:
                        program = self.promptAI.prompt_chatgpt(prompt)
                        index_opening = program.find("```")
                        if index_opening != -1:
                            index_shift = program.find("\n", index_opening)
                            program = program[
                                index_opening + 7 : program.find("```", index_shift)
                            ]
                        file.write(program)

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def anthropic_process_assignment_pdf(self) -> None:
        """
        anthropic_process_assignment_pdf: Loads and preprocesses the assignment PDF to create a cleaned prompt for the Claude model.

        This method:
        - Loads the assignment PDF and converts it into markdown format using `pymupdf4llm`.
        - Constructs a structured prompt with detailed system instructions for cleaning and organizing the text.
        - Sends the prompt to Claude to refine the assignment by removing irrelevant sections, normalizing formatting,
        and preserving coding specifications and technical requirements.
        - Stores the cleaned and processed assignment text for subsequent code generation.

        Raises
        ------
        Exception
            If any unexpected error occurs during PDF loading, prompt construction, or Claude API prompting.
        """
        system_instructions = (
            "You are an expert assistant specialized in preparing programming assignment instructions for analysis.\n\n"
            "Clean and normalize the provided text according to these guidelines:\n"
            "1. Remove redundant sections like 'Contents', 'Table of Contents', or navigation elements\n"
            "2. Normalize whitespace: use single spaces between words, preserve paragraph breaks, and limit consecutive blank lines to one\n"
            "3. Remove formatting artifacts, page numbers, headers/footers, and any non-instructional elements such as background information\n"
            "4. Preserve code snippets, examples, and technical specifications exactly as provided\n"
            "5. Maintain all assignment requirements and evaluation criteria\n"
            "Think step-by-step when processing the text. First identify the core assignment components, then eliminate non-essential elements, and finally structure the output in a clean, consistent format optimized for programmatic analysis."
        )

        try:
            prompt = list()

            if self.raw_pdf is None:
                logging.info(
                    "Loading assignment PDF and converting to markdown using pymupdf4llm..."
                )
                self.raw_pdf = pymupdf4llm.to_markdown(self.pdf_path)

            logging.info("Building the prompt's content...")
            prompt.append({"role": "user", "content": self.raw_pdf})

            logging.info("Prompting Claude to refine markdown text...")
            self.processed_pdf = self.promptAI.prompt_anthropic(
                prompt, system_instructions
            )

            logging.info(
                "Successfully loaded and prepared the assignment PDF's text for prompting using Claude."
            )

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def anthropic_generate_code(self) -> None:
        """
        anthropic_generate_code: Generates multiple AI code submissions using the Claude model based on the assignment PDF and test cases.

        This method:
        - Constructs a structured conversation prompt sequence including assignment descriptions,
        structured test cases, and final code generation instructions.
        - Sends the prompt sequence to the Claude model multiple times to produce diverse code submissions.
        - Parses the Claude model's responses to extract the generated code.
        - Saves each generated program into organized output directories for later analysis or evaluation.

        The method repeats prompting using both a cleaned version of the assignment (processed PDF)
        and the original version (raw PDF), producing multiple independent generations for each.

        Raises
        ------
        Exception
            If any unexpected error occurs during prompt construction, model prompting, or file saving.
        """
        system_instructions = (
            f"You are an expert {self.language} programming assistant. You thoroughly analyze assignment descriptions and multiple structured test cases, including input files and expected output files. You create complete, correct, and well-commented {self.language} programs that satisfy all requirements precisely. Based on the provided assignment description and test cases, develop a complete program that functions exactly as specified.\n\n"
            "Each test case contains:\n"
            "- console_command: the command used to execute the program\n"
            "- console_input: input provided during program execution\n"
            "- expected_console_output: expected output to the console\n"
            "- input_files: files that must be read during execution\n"
            "- output_files: files that must be created or modified during execution\n\n"
            "Carefully examine all provided information. Develop an appropriate, complete program. Verify that your code correctly processes all specified inputs and generates all required outputs. Include clear, concise comments explaining each function's purpose, the reasoning behind complex sections, and important variables. Write elegant and professional code with consistent indentation and formatting."
        )

        code_gen_conclusion = (
            f"Please now generate the full {self.language} program based on the provided instructions, assignment description, and test cases."
            "Output only the complete source code — avoid including additional notes, explanations, or submission instructions outside the program itself.\n"
            "Focus on writing clear, well-commented code that fulfills all provided test cases.\n\n"
            "If minor formatting inconsistencies appear in the sample outputs (such as spaces after colons or extra blank lines), prioritize producing consistent, clean formatting in your program rather than exactly replicating minor inconsistencies."
        )

        try:
            ai_submission_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "ai_submissions"
            )
            test_cases_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "test_cases"
            )

            logging.info("Gathering context for prompt to Claude...")

            test_cases = dict()
            for iter, path in enumerate(test_cases_path.iterdir()):
                case = dict()

                with open(
                    path / "console_command.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["console_command"] = file.read()

                with open(
                    path / "console_input.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["console_input"] = file.read()

                with open(
                    path / "expected_console_output.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["expected_console_output.txt"] = file.read()

                if any((path / "input_files").iterdir()):
                    input_files = dict()
                    for input_path in (path / "input_files").iterdir():
                        with open(input_path, mode="r", encoding="UTF-8") as file:
                            input_files[input_path.name] = file.read()
                    case["input_files"] = input_files

                if any((path / "output_files").iterdir()):
                    output_files = dict()
                    for output_path in (path / "output_files").iterdir():
                        with open(output_path, mode="r", encoding="UTF-8") as file:
                            output_files[output_path.name] = file.read()
                    case["output_files"] = output_files

                test_cases[f"test_case_{iter}"] = case

            for cycle in range(0, 2):
                logging.info(
                    f"Claude - Prompting model with {"processed PDF"if cycle == 0 else "raw PDF"} to generate code..."
                )
                prompt = list()

                prompt.append(
                    {
                        "role": "user",
                        "content": self.processed_pdf if cycle == 0 else self.raw_pdf,
                    }
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Assignment description received. Ready for test cases.",
                    }
                )
                prompt.append(
                    {"role": "user", "content": json.dumps(test_cases, indent=4)}
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Test cases received. Ready to generate code.",
                    }
                )
                prompt.append({"role": "user", "content": code_gen_conclusion})

                for count in range(1, 6):
                    logging.info(
                        f"Claude - {"processed PDF" if cycle == 0 else "raw PDF"} - prompting cycle progress: {count} / 5"
                    )
                    output_path = (
                        ai_submission_path
                        / f"{self.promptAI.ANTHROPIC_MODEL}_{"processed" if cycle == 0 else "raw"}_{count}"
                    )
                    output_path.mkdir(parents=True, exist_ok=True)
                    with open(
                        output_path / "main.cpp", mode="w", encoding="UTF-8"
                    ) as file:
                        program = self.promptAI.prompt_anthropic(
                            prompt, system_instructions
                        )
                        index_opening = program.find("```")
                        if index_opening != -1:
                            index_shift = program.find("\n", index_opening)
                            program = program[
                                index_opening + 7 : program.find("```", index_shift)
                            ]
                        file.write(program)

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def ollama_process_assignment_pdf(self, model: str) -> None:
        """
        ollama_process_assignment_pdf: Loads and preprocesses the assignment PDF to create a cleaned prompt for a specified Ollama model.

        This method:
        - Loads the assignment PDF and converts it into markdown format using `pymupdf4llm`.
        - Constructs a structured prompt with detailed system instructions for cleaning and normalizing the text.
        - Sends the prompt to the specified Ollama model to refine the assignment by removing irrelevant content
        and preserving coding specifications.
        - Saves the cleaned and processed text to a local file for reference.

        Parameters
        ----------
        model : str
            The name of the Ollama model used to process the assignment text.

        Raises
        ------
        Exception
            If any unexpected error occurs during PDF loading, prompt construction, Ollama prompting, or file saving.
        """
        system_instructions = (
            "You are an expert assistant specialized in preparing programming assignment instructions for analysis.\n\n"
            "Clean and normalize the provided text according to these guidelines:\n"
            "1. Remove redundant sections like 'Contents', 'Table of Contents', or navigation elements\n"
            "2. Normalize whitespace: use single spaces between words, preserve paragraph breaks, and limit consecutive blank lines to one\n"
            "3. Remove formatting artifacts, page numbers, headers/footers, and any non-instructional elements such as background information\n"
            "4. Preserve code snippets, examples, and technical specifications exactly as provided\n"
            "5. Maintain all assignment requirements and evaluation criteria\n"
            "Think step-by-step when processing the text. First identify the core assignment components, then eliminate non-essential elements, and finally structure the output in a clean, consistent format optimized for programmatic analysis."
            "Output only the complete processed text — avoid including additional notes or explanations.\n"
        )

        try:
            prompt = list()

            if self.raw_pdf is None:
                logging.info(
                    "Loading assignment PDF and converting to markdown using pymupdf4llm..."
                )
                self.raw_pdf = pymupdf4llm.to_markdown(self.pdf_path)

            logging.info("Building the prompt's content...")

            prompt.append({"role": "system", "content": system_instructions})
            prompt.append({"role": "user", "content": self.raw_pdf})

            logging.info("Prompting Ollama to refine markdown text...")
            self.processed_pdf = self.promptAI.prompt_ollama(prompt, model)

            with open(
                f"processed_pdf_ollama_{model}.txt", mode="w", encoding="UTF-8"
            ) as file:
                file.write(self.processed_pdf)

            logging.info(
                "Successfully loaded and prepared the assignment PDF's text for prompting using Ollama."
            )

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def ollama_generate_code(self, model) -> None:
        """
        ollama_generate_code: Generates multiple AI code submissions using a specified Ollama model based on the assignment PDF and test cases.

        This method:
        - Constructs a conversation-style prompt sequence including assignment descriptions,
        structured test cases, and final code generation instructions.
        - Sends the prompt sequence to the specified Ollama model multiple times to generate diverse code submissions.
        - Parses the model's responses to extract the generated code.
        - Saves each generated program into organized output directories for later analysis or evaluation.

        The method repeats prompting using both a cleaned version of the assignment (processed PDF)
        and the original version (raw PDF), producing multiple independent generations for each.

        Parameters
        ----------
        model : str
            The name of the Ollama model used to generate the code (e.g., "phi", "deepseek").

        Raises
        ------
        Exception
            If any unexpected error occurs during prompt construction, model prompting, or file saving.
        """
        system_instructions = (
            f"You are an expert {self.language} programming assistant. Analyze the assignment description and structured test cases, including input files and expected output files. Write complete, correct, and well-commented {self.language} programs that satisfy all requirements precisely.\n\n"
            "Each test case contains:\n"
            "- console_command: the command used to execute the program\n"
            "- console_input: input provided during program execution\n"
            "- expected_console_output: expected output to the console\n"
            "- input_files: files that must be read during execution\n"
            "- output_files: files that must be created or modified during execution\n\n"
            "You must output ONLY the complete source code for the program.\n"
        )

        code_gen_conclusion = (
            f"Please now generate the full {self.language} program based on the provided instructions, assignment description, and test cases.\n"
            "Output only the complete source code — avoid including additional notes, explanations, or submission instructions outside the program itself.\n"
            "Focus on writing clear, well-commented code that fulfills all provided test cases.\n"
            "For minor formatting inconsistencies in sample outputs, prioritize consistent, clean formatting."
        )

        try:
            ai_submission_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "ai_submissions"
            )
            test_cases_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "test_cases"
            )

            logging.info("Gathering context for prompt to Ollama...")

            test_cases = dict()
            for iter, path in enumerate(test_cases_path.iterdir()):
                case = dict()

                with open(
                    path / "console_command.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["console_command"] = file.read()

                with open(
                    path / "console_input.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["console_input"] = file.read()

                with open(
                    path / "expected_console_output.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["expected_console_output.txt"] = file.read()

                if any((path / "input_files").iterdir()):
                    input_files = dict()
                    for input_path in (path / "input_files").iterdir():
                        with open(input_path, mode="r", encoding="UTF-8") as file:
                            input_files[input_path.name] = file.read()
                    case["input_files"] = input_files

                if any((path / "output_files").iterdir()):
                    output_files = dict()
                    for output_path in (path / "output_files").iterdir():
                        with open(output_path, mode="r", encoding="UTF-8") as file:
                            output_files[output_path.name] = file.read()
                    case["output_files"] = output_files

                test_cases[f"test_case_{iter}"] = case

            for cycle in range(0, 2):
                logging.info(
                    f"Ollama - Prompting model '{model}' with {"processed PDF"if cycle == 0 else "raw PDF"} to generate code..."
                )
                prompt = list()

                prompt.append({"role": "system", "content": system_instructions})
                prompt.append(
                    {
                        "role": "user",
                        "content": self.processed_pdf if cycle == 0 else self.raw_pdf,
                    }
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Assignment description received. Ready for test cases.",
                    }
                )
                prompt.append(
                    {"role": "user", "content": json.dumps(test_cases, indent=4)}
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Test cases received. Ready to generate code.",
                    }
                )
                prompt.append({"role": "user", "content": code_gen_conclusion})

                for count in range(1, 6):
                    logging.info(
                        f"Ollama - {"processed PDF" if cycle == 0 else "raw PDF"} - prompting cycle progress: {count} / 5"
                    )
                    output_path = (
                        ai_submission_path
                        / f"{model}_{"processed" if cycle == 0 else "raw"}_{count}"
                    )
                    output_path.mkdir(parents=True, exist_ok=True)
                    with open(
                        output_path / "main.cpp", mode="w", encoding="UTF-8"
                    ) as file:
                        program = self.promptAI.prompt_ollama(prompt, model)
                        index_opening = program.find("```")
                        if index_opening != -1:
                            index_shift = program.find("\n", index_opening)
                            program = program[
                                index_opening + 7 : program.find("```", index_shift)
                            ]
                        file.write(program)

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def grok_process_assignment_pdf(self) -> None:
        """
        grok_process_assignment_pdf: Loads and preprocesses the assignment PDF to create a cleaned prompt for the Grok model.

        This method:
        - Loads the assignment PDF and converts it into markdown format using `pymupdf4llm`.
        - Constructs a conversation-style prompt with detailed system instructions for cleaning and organizing the text.
        - Sends the prompt to the Grok model to refine the assignment by removing redundant sections, normalizing whitespace,
        and focusing on the coding specifications.
        - Stores the cleaned and processed assignment text for subsequent code generation.

        Raises
        ------
        Exception
            If any unexpected error occurs during PDF loading, prompt construction, or Grok API prompting.
        """
        instructions = (
            "You are an expert assistant specialized in preparing programming assignment instructions for analysis.\n"
            "Your task is to clean and normalize the provided text according to the following guidelines:\n"
            "- Remove any redundant 'Contents' sections.\n"
            "- Normalize whitespace: use single spaces between words, remove leading and trailing spaces, and limit consecutive blank lines to a maximum of two.\n"
            "- Eliminate any leftover formatting artifacts or stray characters that do not contribute to the clarity of the instructions.\n"
            "- Remove background information or narrative sections that do not specify coding requirements.\n"
            "Focus on producing a clean, concise version of the assignment that emphasizes the coding specifications."
        )

        try:
            prompt = list()

            if self.raw_pdf is None:
                logging.info(
                    "Loading assignment PDF and converting to markdown using pymupdf4llm..."
                )
                self.raw_pdf = pymupdf4llm.to_markdown(self.pdf_path)

            prompt.append({"role": "system", "content": instructions})
            prompt.append({"role": "user", "content": self.raw_pdf})

            self.processed_pdf = self.promptAI.prompt_grok(prompt)

            # with open("processed_pdf_grok.txt", mode = "w", encoding = "UTF-8") as file:
            #     file.write(self.processed_pdf)

            logging.info(
                "Successfully loaded and prepared the assignment PDF's text for prompting."
            )

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise

    def grok_generate_code(self) -> None:
        """
        grok_generate_code: Generates multiple AI code submissions using the Grok model based on the assignment PDF and test cases.

        This method:
        - Constructs a conversation-style prompt sequence including system instructions, assignment descriptions,
        structured test cases, and final code generation instructions.
        - Sends the prompt sequence to the Grok model multiple times to produce diverse code submissions.
        - Parses the Grok model's responses to extract the generated source code.
        - Saves each generated program into organized output directories for later analysis or evaluation.

        The method repeats prompting using both a cleaned version of the assignment (processed PDF)
        and the original version (raw PDF), producing multiple independent generations for each.

        Raises
        ------
        Exception
            If any unexpected error occurs during prompt construction, model prompting, or file saving.
        """
        system_message = (
            "You carefully review assignment descriptions and multiple structured test cases, "
            "including input files and expected output files. "
            f"You generate complete, correct, and well-commented {self.language} programs that meet all requirements."
        )

        code_gen_instructions = (
            f"You are an expert {self.language} programming assistant.\n"
            "Using the provided assignment description and test cases, "
            "generate a complete program that behaves exactly as specified.\n\n"
            "Each test case includes:\n"
            "- console_command: the command used to run the program\n"
            "- console_input: input provided during program execution\n"
            "- expected_console_output: expected output to the console\n"
            "- input_files: files that must be read during execution\n"
            "- output_files: files that must be created or written during execution\n\n"
            "Analyze all provided context carefully. Synthesize an appropriate, complete program. "
            "Ensure that your code correctly handles all specified inputs and produces all required outputs. "
            "Include clear and concise comments to explain the purpose of each function, "
            "the logic behind complex sections, and important variables."
            "Write clean and professional code, ensuring consistent indentation and formatting throughout the program."
        )

        code_gen_conclusion = (
            f"Please now generate the full {self.language} program based on the provided instructions, assignment description, and test cases."
            "Only output the complete source code — do not include any additional notes, explanations, or submission instructions outside of the program itself.\n"
            "Focus on writing clear, well-commented code that satisfies all provided test cases.\n\n"
            "If minor formatting inconsistencies appear in the sample outputs (such as spaces after colons or extra blank lines), prioritize producing consistent, clean formatting within your program rather than exactly replicating minor inconsistencies."
        )

        try:
            ai_submission_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "ai_submissions"
            )
            test_cases_path = (
                genAI.BASE_PATH / f"assignment_{self.assignment_id}" / "test_cases"
            )

            logging.info("Gathering context for prompt to Grok...")

            test_cases = dict()

            for iter, path in enumerate(test_cases_path.iterdir()):
                case = dict()

                with open(
                    path / "console_command.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["console_command"] = file.read()

                with open(
                    path / "console_input.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["console_input"] = file.read()

                with open(
                    path / "expected_console_output.txt", mode="r", encoding="UTF-8"
                ) as file:
                    case["expected_console_output.txt"] = file.read()

                if any((path / "input_files").iterdir()):
                    input_files = dict()
                    for input_path in (path / "input_files").iterdir():
                        with open(input_path, mode="r", encoding="UTF-8") as file:
                            input_files[input_path.name] = file.read()
                    case["input_files"] = input_files

                if any((path / "output_files").iterdir()):
                    output_files = dict()
                    for output_path in (path / "output_files").iterdir():
                        with open(output_path, mode="r", encoding="UTF-8") as file:
                            output_files[output_path.name] = file.read()
                    case["output_files"] = output_files

                test_cases[f"test_case_{iter}"] = case

            for cycle in range(0, 2):
                logging.info(
                    f"Grok - Prompting model with {"processed PDF"if cycle == 0 else "raw PDF"} to generate code..."
                )
                prompt = list()
                prompt.append({"role": "system", "content": system_message})
                prompt.append({"role": "user", "content": code_gen_instructions})
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Instructions received. Ready for assignment description.",
                    }
                )
                prompt.append(
                    {
                        "role": "user",
                        "content": self.processed_pdf if cycle == 0 else self.raw_pdf,
                    }
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Assignment description received. Ready for test cases.",
                    }
                )
                prompt.append(
                    {"role": "user", "content": json.dumps(test_cases, indent=4)}
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": "Test cases received. Ready to generate code.",
                    }
                )
                prompt.append({"role": "user", "content": code_gen_conclusion})

                for count in range(1, 6):
                    logging.info(
                        f"Grok - {"processed PDF" if cycle == 0 else "raw PDF"} - prompting cycle progress: {count} / 5"
                    )
                    output_path = (
                        ai_submission_path
                        / f"{self.promptAI.GROK_MODEL}_{"processed" if cycle == 0 else "raw"}_{count}"
                    )
                    output_path.mkdir(parents=True, exist_ok=True)
                    with open(
                        output_path / "main.cpp", mode="w", encoding="UTF-8"
                    ) as file:
                        program = self.promptAI.prompt_grok(prompt)
                        index_opening = program.find("```")
                        if index_opening != -1:
                            index_shift = program.find("\n", index_opening)
                            program = program[
                                index_opening + 7 : program.find("```", index_shift)
                            ]
                        file.write(program)

        except Exception:
            logging.exception(f"Fatal error occurred in: {self.get_current_scope()}")
            raise


if __name__ == "__main__":
    main()
