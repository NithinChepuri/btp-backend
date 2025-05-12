# Ada Code Traceability System

This system implements a traceability mechanism for Ada code, connecting natural language requirements to Ada code files based on the approach from the Requirements2Code paper.

## Overview

The traceability system uses a combination of semantic similarity (70% weight) and keyword matching (30% weight) to find the most relevant Ada code files for a given requirement. The system provides:

1. Analysis of requirements against Ada code files
2. Visualization of traceability links in Neo4j
3. Persistence of analysis results for future reference

## Requirements Analysis Process

The analysis process involves:

1. Parsing Ada code files to extract code elements (packages, procedures, functions)
2. Generating embeddings for both requirements and code elements using BERT
3. Calculating relevance scores based on semantic similarity and keyword matching
4. Identifying the most relevant code files for a requirement

## Project Components

- `analyze_single_req.py`: Analyzes a single requirement against Ada code files
- `analyze_req.py`: Simplified version for analyzing sample requirements
- `code2graph.py`: Parses Ada files and builds a code graph
- `graph_database.py`: Manages Neo4j graph database operations
- `load_and_persist_neo4j.py`: Loads analysis results into Neo4j and exports them
- `restore_neo4j_data.py`: Restores Neo4j data from previously saved exports

## Getting Started

### Prerequisites

- Python 3.x
- Neo4j (local installation)
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure Neo4j connection in `constants.py`

## Usage

### 1. Analyze a Single Requirement

```bash
python analyze_single_req.py --req-file sample_requirement.txt --req-id AUTH-01 --output outputs/saved_analysis/auth_requirement_analysis.json
```

Parameters:
- `--req-file`: Path to file containing the requirement text
- `--req-id`: Identifier for the requirement
- `--output`: Path to save the analysis results

### 2. Load Results into Neo4j and Save for Future

```bash
python load_and_persist_neo4j.py
```

This will:
- Load the requirement analysis into Neo4j
- Create traceability links between requirements and code
- Save metadata for future restoration

### 3. Restore Previously Saved Analysis

```bash
python restore_neo4j_data.py
```

This will:
- List available saved analyses
- Prompt to select one to restore
- Restore the selected analysis to Neo4j

## Accessing Neo4j Visualization

1. Ensure Neo4j is running
2. Open a web browser and navigate to `http://localhost:7474/`
3. Log in with your credentials (as configured in `constants.py`)
4. Run Cypher queries to explore the data:

```cypher
// View the requirement and related files
MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File) 
RETURN r, rel, f 
ORDER BY rel.rank

// View the requirement and all traced code elements
MATCH (r:Requirement)-[rel:TRACED_TO]->(n:AdaNode) 
RETURN r, rel, n 
ORDER BY rel.score DESC 
LIMIT 50

// View files and their code elements
MATCH (f:File)-[:CONTAINS]->(n:AdaNode) 
RETURN f, n 
LIMIT 100

// View code hierarchy
MATCH p=(n:AdaNode)-[:CONTAINS*1..2]->(c) 
RETURN p 
LIMIT 100
```

## Example Authentication Requirement Analysis

We've analyzed an authentication requirement:

```
The system must provide a secure authentication mechanism for users to log in with their credentials. The authentication process should validate user credentials against stored data and enforce password policies including minimum length and complexity requirements. Failed login attempts should be recorded and after a configurable number of failed attempts, the account should be temporarily locked.
```

Top matching files:
1. `awa-sysadmin-beans.adb` (Score: 0.3227)
2. `awa-users-services-tests.adb` (Score: 0.3119)
3. `awa-users-servlets.ads` (Score: 0.3083)
4. `awa-users-services.ads` (Score: 0.3075)
5. `awa-users-tests.adb` (Score: 0.2983)

## Directory Structure

```
.
├── analyze_req.py                  # Simplified requirements analysis
├── analyze_single_req.py           # Main requirement analysis script
├── code2graph.py                   # Ada code parsing and graph building
├── constants.py                    # Configuration constants
├── graph_database.py               # Neo4j database operations
├── load_and_persist_neo4j.py       # Load analysis into Neo4j and save
├── restore_neo4j_data.py           # Restore saved analysis
├── requirements.txt                # Python dependencies
├── sample_requirement.txt          # Example requirement
├── outputs/                        # Output directory
│   ├── checkpoints/                # Code graph checkpoints
│   └── saved_analysis/             # Saved analyses
│       └── neo4j_exports/          # Neo4j exports
└── datasets/                       # Code datasets
    └── ada-awa/                    # Ada AWA codebase
```

## Notes on the Authentication System

The authentication system in the AWA codebase provides:

1. **User credential validation**: Verifies user credentials against stored data
2. **Password policy enforcement**: Requires minimum length and complexity
3. **Failed login tracking**: Records login attempts and locks accounts after failures
4. **Secure password storage**: Uses HMAC-SHA1 hashing with salt

Key components:
- `awa-users-services.ads/adb`: Core authentication implementation
- `awa-users-servlets.ads/adb`: Web interface for authentication
- `awa-users-beans.adb`: UI components for authentication
- `awa-users-filters.ads`: Request filtering for authentication

## Extensions and Future Work

1. Batch analysis of multiple requirements
2. Interactive visualization dashboard
3. Improved code parsing for more languages
4. Natural language query interface for finding relevant code 