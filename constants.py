"""
Constants for the Ada Traceability Project.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ADA_CODE_DIR = Path("datasets/ada-awa")
REQUIREMENTS_DIR = Path("datasets/ada-awa/ada_requirements")
OUTPUT_DIR = Path("outputs")
SUMMARIES_DIR = Path("outputs/summaries")
GRAPH_DIR = Path("outputs/graphs")

# Graph Database
NEO4J_URI = "bolt://localhost:7687"  # Change this if your Neo4j is on a different host/port
NEO4J_USER = "neo4j"                 # Your Neo4j username
NEO4J_PASSWORD = "nithinrajeev"          # Your Neo4j password

# Code file extensions
ADA_EXTENSIONS = [".ads", ".adb", ".ada"]

# Model parameters
BERT_MODEL_NAME = "all-MiniLM-L6-v2"  # Local BERT model for embeddings
MAX_TOKENS = 8000  # Maximum tokens to process at once

# GitHub API
GITHUB_API_TOKEN = ""  # Add your GitHub API token here if needed
GITHUB_REPO_OWNER = ""  # Add GitHub repo owner if needed
GITHUB_REPO_NAME = ""  # Add GitHub repo name if needed

# Neo4j relationship types
IMPLEMENTS = "IMPLEMENTS"
DEPENDS_ON = "DEPENDS_ON"
REFERENCES = "REFERENCES"
SIMILAR_TO = "SIMILAR_TO"
COMMITS_TO = "COMMITS_TO"
ISSUES_RELATED = "ISSUES_RELATED" 