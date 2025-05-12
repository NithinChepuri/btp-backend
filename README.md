# Ada Code Traceability System

A comprehensive traceability system for Ada code that creates a knowledge graph connecting requirements, code entities, GitHub issues, and Git commit history.

## Overview

This system provides traceability between software artifacts for Ada codebases by:

1. Extracting nodes from requirements, Ada code, GitHub issues, and Git commits
2. Generating embeddings to represent these nodes semantically
3. Creating similarity-based links between different types of nodes
4. Building a Neo4j graph database to store and query the traceability relationships
5. Providing tools to explore and analyze traceability coverage

## Features

- Ada code parsing to extract packages, procedures, functions, and types
- Requirements parsing from text files
- GitHub issues and Git commit history integration
- Semantic linking based on OpenAI embeddings
- Neo4j graph database for storing and querying the traceability data
- Command-line tools for analyzing traceability coverage

## Prerequisites

- Python 3.8+
- Neo4j Database (local or remote)
- OpenAI API key (for generating embeddings)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the Neo4j database:
   - Install Neo4j (see [Neo4j installation guide](https://neo4j.com/docs/operations-manual/current/installation/))
   - Start the Neo4j server
   - Update the connection parameters in `constants.py` if needed

4. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your-api-key
   ```

## Project Structure

- `constants.py` - Paths, database settings, and relationship types
- `code2graph.py` - Ada code parser to extract code entities
- `req2nodes.py` - Requirements parser
- `github_integration.py` - GitHub issues and Git commits extraction
- `embeddings.py` - Embedding generation and node linking
- `graph_database.py` - Neo4j database integration
- `querying.py` - Tools for querying the traceability graph
- `run.py` - Main script to orchestrate the pipeline
- `query_tool.py` - Command-line interface for querying traceability

## Usage

### Running the Complete Pipeline

```
python run.py
```

This will:
1. Extract nodes from code, requirements, issues, and commits
2. Apply pre-generated summaries to code nodes
3. Generate embeddings for all nodes
4. Link nodes based on semantic similarity
5. Build the Neo4j graph database
6. Generate a traceability report

### Command-line Options

```
python run.py --help
```

Options include:
- `--skip-embeddings`: Skip generating embeddings
- `--skip-graph`: Skip building the graph database
- `--report-only`: Only generate the traceability report

### Querying the Traceability Graph

```
python query_tool.py [command]
```

Available commands:
- `report`: Generate a comprehensive traceability report
- `export`: Export traceability data to JSON
- `stats`: Show traceability statistics
- `uncovered-reqs`: Find requirements without code
- `uncovered-code`: Find code without requirements
- `trace-req [REQ_ID]`: Trace a requirement to code
- `trace-code [CODE_NAME]`: Trace code to requirements
- `trace-issue [ISSUE_ID]`: Trace an issue to affected requirements

Example:
```
python query_tool.py trace-req 42
```

## Configuration

Edit `constants.py` to configure:
- Paths for the Ada code, requirements, and output directories
- Neo4j database connection parameters
- GitHub API credentials
- Embedding model settings

## Dataset Structure

The system expects the following structure:
- Ada code files (.ads, .adb, .ada) in the `datasets/ada-awa` directory
- Requirements files in the `datasets/ada-awa/ada_requirements` directory (format: reqXX.txt)
- Pre-generated summaries in the `outputs/summaries/ada_summaries.json` file (optional)

## License

[MIT License](LICENSE)

## Acknowledgements

This project is inspired by the nl2codeTrace GitHub project and adapts its approach for Ada code, with additional features for GitHub issues and commit history integration. 