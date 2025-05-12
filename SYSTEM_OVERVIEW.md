# Ada Requirements Traceability System

This document provides an overview of the traceability system implementation, explaining each file's purpose and how the system works to connect requirements to code.

## System Overview

This traceability system implements the approach described in the "Requirements2Code" paper, creating connections between:

1. **Requirements** (from ada_requirements directory)
2. **Ada code files and structures** (.ads, .adb files)
3. **GitHub issues** (from issues.json)
4. **Git commits** (from commits.json)

The system uses both semantic similarity (via BERT embeddings) and keyword matching to identify which code files implement each requirement, with a weighting of 70% for semantic similarity and 30% for keyword matching.

## File Descriptions

### Main Analysis Scripts

1. **analyze_ada_requirements.py**
   - The main script that processes all requirements from the `ada_requirements` directory
   - Analyzes each requirement against the Ada code files
   - Creates a Neo4j database with the results
   - Attempts to connect issues and commits to files they modify or mention

2. **analyze_single_req.py**
   - Standalone script to analyze a single requirement against the codebase
   - Used by other scripts to handle individual requirements
   - Returns the top-N most relevant files for a requirement

3. **enhance_neo4j_connections.py**
   - Enhances the Neo4j database with additional relationships
   - Creates direct, visible connections between requirements, issues, and commits
   - Has a special mode to analyze a single requirement (--analyze-req)

4. **list_requirements.py**
   - Lists all requirements found in the Neo4j database
   - Shows which requirements have the most connected files
   - Helps identify requirements to analyze individually

### Supporting Modules

5. **code2graph.py**
   - Parses Ada code files into a graph of nodes
   - Extracts packages, functions, procedures, etc.
   - Creates a graph representation of the codebase structure

6. **req2nodes.py**
   - Processes requirements into nodes for analysis
   - Handles text preparation and keyword extraction

7. **graph_database.py**
   - Handles all Neo4j database operations
   - Creates nodes, relationships, and constraints
   - Contains methods for querying the database

### Configuration and Data

8. **constants.py**
   - Defines paths, weights, and configuration settings
   - Central place for adjusting system parameters

9. **issues.json**
   - Contains GitHub issues data
   - Used to trace issues to requirements via file modifications or mentions

10. **commits.json**
    - Contains Git commit history
    - Used to trace commits to requirements via file modifications

## Workflow

### Basic Traceability Workflow

1. **Extract Code Structure**:
   - Ada code files are parsed into nodes (packages, functions, etc.)
   - These nodes are stored in a checkpoint file for faster processing

2. **Analyze Requirements**:
   - Each requirement is analyzed against the code files
   - Semantic similarity and keyword matching are used
   - Results are saved as JSON files

3. **Build Graph Database**:
   - Requirements, files, and code nodes are loaded into Neo4j
   - Relationships between them are established
   - Issues and commits are connected to files they modify

4. **Enhance Connections**:
   - Additional direct relationships are created
   - This improves visualization in Neo4j

### How to Run The System

#### Full Analysis

```bash
# Run the full analysis of all requirements
python analyze_ada_requirements.py

# Enhance Neo4j connections to make them more visible
python enhance_neo4j_connections.py --enhance
```

#### Analyzing a Single Requirement

```bash
# List available requirements in Neo4j
python list_requirements.py

# Analyze a specific requirement
python enhance_neo4j_connections.py --analyze-req req1
```

#### Visualizing in Neo4j

Open Neo4j Browser and try these queries:

```cypher
// View all requirements and their related files
MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File) RETURN r, rel, f LIMIT 10

// View files that might relate to both issues and requirements
MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[:LINKED_TO]-(i:Issue) 
RETURN r, f, i LIMIT 10

// View files that might relate to both commits and requirements
MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[:LINKED_TO]-(c:Commit) 
RETURN r, f, c LIMIT 10
```

## System Features

1. **Semantic Matching**:
   - Uses BERT embeddings to understand the meaning of requirements
   - Can identify files that don't share exact words but have similar concepts

2. **Keyword Matching**:
   - Uses keywords from requirements to find relevant code files
   - Helps with precise matching of technical terms

3. **Graph-Based Structure**:
   - Represents requirements, code, issues, and commits as a knowledge graph
   - Allows for complex queries and relationship exploration

4. **Requirements Impact Analysis**:
   - Can identify which requirements are affected by specific code changes
   - Enables tracing from requirements to implementation and back

5. **Issue and Commit Integration**:
   - Connects GitHub issues and Git commits to requirements
   - Provides a full lifecycle view from requirement to implementation to issues/changes

## Limitations and Future Work

1. **Connection Quality**:
   - Some connections between requirements and code might be weak
   - Manual validation of high-importance connections is recommended

2. **Issue/Commit Matching**:
   - Connections to issues and commits rely on text analysis
   - Direct references (e.g., "fixes issue #42") work best

3. **Future Improvements**:
   - Improve file matching algorithms
   - Add more metadata to connections
   - Create a web interface for easier exploration 