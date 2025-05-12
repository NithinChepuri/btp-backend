# Ada Requirements Traceability with Neo4j

This document explains how to run a comprehensive traceability analysis on the Ada-AWA codebase, using the existing requirements in the `ada_requirements` directory and GitHub issues/commits data from `issues.json` and `commits.json`.

## Overview

The system implements a traceability approach from the "Requirements2Code" paper, connecting:
- Requirements from the ada_requirements directory
- Ada code files and their internal structures
- GitHub issues from issues.json
- Git commits from commits.json

This creates a comprehensive traceability view in Neo4j that shows how requirements, code, issues, and commits are related.

## Setup

1. Make sure Neo4j is running:
   - If using Neo4j Desktop, start your database
   - If using Docker: `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest`

2. The `graph_database.py` file contains connection settings. Update if needed.

## Running the Analysis

To analyze all requirements and load them into Neo4j:

```bash
python analyze_ada_requirements.py
```

To test with a smaller set of requirements first:

```bash
python analyze_ada_requirements.py --max-reqs 10
```

If you've already run the analysis and want to load the existing results into Neo4j:

```bash
python analyze_ada_requirements.py --skip-analysis
```

## Neo4j Visualization

After loading the data into Neo4j, you can visualize and explore the traceability graph.

1. Open Neo4j Browser:
   - Neo4j Desktop: Click "Open" on your database card
   - Browser URL: http://localhost:7474/

2. Login with your Neo4j credentials (default: neo4j/password)

3. Run these Cypher queries to explore the traceability data:

### Basic Requirement Traceability

```cypher
// View all requirements and their related files
MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File)
RETURN r, rel, f
```

### Finding Requirements Meeting Points

```cypher
// View files that satisfy multiple requirements (meeting point)
MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File)
WITH f, count(r) as req_count, collect(r) as requirements
WHERE req_count > 1
RETURN f, requirements
```

### Direct Issue-Requirement Connections

```cypher
// View issues directly related to requirements
MATCH (r:Requirement)-[rel:RELATED_ISSUE]->(i:Issue)
RETURN r, rel, i
```

### Direct Commit-Requirement Connections 

```cypher
// View commits directly related to requirements
MATCH (r:Requirement)-[rel:RELATED_COMMIT]->(c:Commit)
RETURN r, rel, c
```

### Complete Traceability Chain

```cypher
// View the complete traceability chain
MATCH p=(r:Requirement)-[:RELATED_TO]->(f:File)<-[:MENTIONS|MODIFIES|REFERENCES]-(x)
WHERE x:Issue OR x:Commit
RETURN p
```

### Issues and Their Connected Files

```cypher
// View issues and the files they mention
MATCH (i:Issue)-[:MENTIONS|MIGHT_RELATE_TO]->(f:File)
RETURN i, f
```

### Commits and Their Connected Files

```cypher
// View commits and the files they modify or reference
MATCH (c:Commit)-[:MODIFIES|REFERENCES|MIGHT_RELATE_TO]->(f:File)
RETURN c, f
```

### Issue-Commit Connections

```cypher
// View issues that are resolved by commits
MATCH (c:Commit)-[:RESOLVES]->(i:Issue)
RETURN c, i
```

### Detailed Code Structure

```cypher
// View the internal structure of Ada files implementing a requirement
MATCH (r:Requirement)-[:RELATED_TO]->(f:File)-[:CONTAINS]->(n:AdaNode)
RETURN r, f, n
```

### Finding Implementation of Specific Requirements

```cypher
// Find the implementation of a specific requirement
MATCH (r:Requirement)
WHERE r.req_id = "req8" // Replace with your requirement ID
MATCH (r)-[:RELATED_TO]->(f:File)
OPTIONAL MATCH (f)-[:CONTAINS]->(n:AdaNode)
RETURN r, f, n
```

## Understanding the Graph Structure

The Neo4j graph has the following node types:
- **Requirement**: Ada requirements from the ada_requirements directory
- **File**: Ada code files (.ads, .adb)
- **AdaNode**: Internal code structures (packages, procedures, functions, etc.)
- **Issue**: GitHub issues from issues.json
- **Commit**: Git commits from commits.json

And the following relationship types:
- **RELATED_TO**: Connects requirements to files they're implemented in
- **CONTAINS**: Connects files to their internal code nodes
- **MENTIONS**: Connects issues to files they explicitly mention
- **MIGHT_RELATE_TO**: Connects issues/commits to files based on package name matching
- **MODIFIES**: Connects commits to files they modify (from commit data)
- **REFERENCES**: Connects commits to files mentioned in commit messages
- **TRACED_TO**: Direct connection from requirements to code nodes
- **PARENT_OF**: Hierarchical relationship between code nodes
- **REFERENCES**: References between code nodes
- **RESOLVES**: Connects commits to issues they resolve
- **RELATED_ISSUE**: Direct connection from requirements to related issues
- **RELATED_COMMIT**: Direct connection from requirements to related commits

## Visualization Tips

1. In Neo4j Browser, click the graph icon at the bottom left to switch to graph view.
2. Use the node labels in the left sidebar to filter what's displayed.
3. Double-click on nodes to expand their relationships.
4. Use the styling options to customize the visualization based on node types and relationship types.
5. For large graphs, use the "Limit" parameter in your queries to restrict the number of results.

## Advanced Queries

### Finding Requirements with the Most Issues

```cypher
MATCH (r:Requirement)-[:RELATED_ISSUE]->(i:Issue)
WITH r, count(i) as issueCount
RETURN r.req_id, r.text, issueCount
ORDER BY issueCount DESC
LIMIT 10
```

### Finding Most Modified Files

```cypher
MATCH (f:File)<-[:MODIFIES]-(c:Commit)
WITH f, count(c) as commitCount
RETURN f.name, f.path, commitCount
ORDER BY commitCount DESC
LIMIT 10
```

### Finding Files That Satisfy Requirements and Have Issues

```cypher
MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[:MENTIONS|MIGHT_RELATE_TO]-(i:Issue)
WITH f, count(distinct r) as reqCount, count(distinct i) as issueCount
RETURN f.name, reqCount, issueCount
ORDER BY reqCount DESC, issueCount DESC
LIMIT 20
```

## Exporting/Importing the Database

To save your Neo4j database for future use, in Neo4j Desktop:
1. Stop your database
2. Click "..." > "Manage" > "Export database"
3. Choose a location to save the export

To import:
1. Create a new database or select an existing one
2. Click "..." > "Manage" > "Import database"
3. Select your exported file 