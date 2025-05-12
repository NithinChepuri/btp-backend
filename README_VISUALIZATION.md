# Visualizing Requirements Traceability in Neo4j

This document explains how to effectively visualize the traceability relationships between requirements, code files, issues, and commits in Neo4j.

## Overview

The traceability visualization system creates strong, visible connections between different elements in the system:

- **Requirements** - From the ada_requirements directory
- **Code Files** - Ada `.ads` and `.adb` files
- **Issues** - From GitHub issues data
- **Commits** - From Git commit history

These connections are made visible with different colored, thick arrows that make the relationships immediately apparent in the Neo4j graph browser.

## Running the Visualization

To create or refresh the visualization relationships, run:

```bash
python visualize_traceability.py
```

This script:
1. Creates strong visual `IMPACTS` relationships from issues to code files
2. Creates visible `CHANGES` relationships from commits to code files
3. Creates direct `SATISFIES` relationships from requirements to issues and commits
4. Creates `ADDRESSES` relationships between commits and the issues they fix

## Fixing Missing Commit Relationships

If you notice that commits are not showing up properly in the visualization, you can use the following command to rebuild all commit relationships:

```bash
python visualize_traceability.py --rebuild-commits
```

This will:
1. Clear all existing CHANGES relationships
2. Use multiple methods to connect commits to files:
   - Extract file mentions from commit messages
   - Use commit file data if available
   - Connect based on keywords in commit messages
   - Apply fallback connections to ensure all commits have relationships

## Analyzing All Requirements

To process all requirements from the ada_requirements directory and ensure proper commit connections:

```bash
python analyze_all_requirements.py
```

This script:
1. Processes all .txt files in the ada_requirements directory
2. Analyzes each requirement against the codebase
3. Creates nodes and relationships in Neo4j
4. Rebuilds commit relationships to ensure proper connections
5. Creates visualization relationships

## Neo4j Queries for Visualization

After running the script, use these queries in the Neo4j Browser to see the relationships:

### 1. Requirements and Issues

```cypher
// View requirements connected to issues
MATCH p=(r:Requirement)-[rel:SATISFIES]->(i:Issue) 
RETURN p LIMIT 25
```

### 2. Requirements and Commits

```cypher
// View requirements connected to commits
MATCH p=(r:Requirement)-[rel:SATISFIES]->(c:Commit) 
RETURN p LIMIT 25
```

### 3. Commits Addressing Issues

```cypher
// View commits addressing issues
MATCH p=(c:Commit)-[rel:ADDRESSES]->(i:Issue) 
RETURN p LIMIT 25
```

### 4. Complete Traceability Chain

```cypher
// View the complete traceability chain
MATCH p=(r:Requirement)-[:RELATED_TO]->(f:File)<-[:CHANGES]-(c:Commit)
RETURN p LIMIT 10
```

### 5. Issues Impacting Files

```cypher
// View issues impacting files
MATCH p=(i:Issue)-[rel:IMPACTS]->(f:File) 
RETURN p LIMIT 25
```

### 6. Commits Changing Files

```cypher
// View commits changing files (with higher limit to ensure results)
MATCH p=(c:Commit)-[rel:CHANGES]->(f:File) 
RETURN p LIMIT 100
```

## Visualization Tips

1. **Color Coding**: 
   - Red arrows: Issues impacting files
   - Blue arrows: Commits changing files
   - Green arrows: Requirements satisfying issues/commits
   - Purple arrows: Commits addressing issues

2. **Focus on the Most Connected Nodes**:
   ```cypher
   // Find requirements with the most connections
   MATCH (r:Requirement)-[rel:SATISFIES]->()
   WITH r, count(rel) as connections
   WHERE connections > 1
   RETURN r, connections
   ORDER BY connections DESC
   LIMIT 10
   ```

3. **Find Connection Points**:
   ```cypher
   // Find files that connect requirements and issues
   MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[:IMPACTS]-(i:Issue)
   RETURN r, f, i
   LIMIT 20
   ```

## Analyzing a Single Requirement

To analyze a single requirement and update the visualization:

```bash
# From text with automatic commit indexing
python visualize_traceability.py --req "The system must provide authentication for users" --req-id SEC-01 --index-commits

# From file with automatic commit indexing
python visualize_traceability.py --req-file sample_requirement.txt --req-id AUTH-01 --index-commits
```

The `--index-commits` flag ensures direct relationships to commits are created, making them immediately visible in the visualization.

After analyzing, you'll be asked if you want to add the requirement to Neo4j. If you choose yes, the script will:
1. Create a new requirement node
2. Connect it to the relevant files
3. If using `--index-commits`, directly create relationships to relevant commits

## Querying Requirements and Their Files

To get a list of all files for a specific requirement:

```bash
python -c "from graph_database import Neo4jConnector; db = Neo4jConnector(); db.connect(); result = db._execute_query('MATCH (r:Requirement {req_id: \"REQ-01\"})-[:RELATED_TO]->(f:File) RETURN f.name, f.path'); print('\n'.join([f'{r[\"f.name\"]} - {r[\"f.path\"]}' for r in result]))"
```

To get a list of all requirements with their file count:

```bash
python -c "from graph_database import Neo4jConnector; db = Neo4jConnector(); db.connect(); result = db._execute_query('MATCH (r:Requirement)-[:RELATED_TO]->(f:File) WITH r, count(f) as file_count RETURN r.req_id, file_count ORDER BY file_count DESC'); print('\n'.join([f'{r[\"r.req_id\"]} - {r[\"file_count\"]} files' for r in result]))"
```

## Troubleshooting Visualization

If the relationships are not visible in Neo4j Browser:

1. **Check Styling**: In Neo4j Browser, click on the database icon in the left sidebar, then the "Style" tab

2. **Set Relationship Thickness**:
   - Set relationship width based on property: `width`
   - Set relationship color based on property: `color`

3. **Clear Browser Cache**: Sometimes you need to clear the Neo4j Browser cache to see new relationship styles

4. **Check Node Count**: 
   ```cypher
   MATCH (n) RETURN labels(n) as Type, count(*) as Count
   ```

5. **Check Relationship Count**: 
   ```cypher
   MATCH ()-[r]->() RETURN type(r) as Type, count(*) as Count
   ```

6. **Fix Missing Commits**: If commits are not showing up in the visualization:
   ```bash
   python visualize_traceability.py --rebuild-commits
   ```

## Going Beyond Visualization

The visual relationships help you see the big picture, but you can also extract specific information:

```cypher
// Find requirements affected by a specific issue
MATCH (r:Requirement)-[:SATISFIES]->(i:Issue {number: 42})
RETURN r.req_id, r.text

// Find all code impacted by a requirement
MATCH (r:Requirement {req_id: "req8"})-[:RELATED_TO]->(f:File)
RETURN f.name, f.path

// Find all commits related to a requirement
MATCH (r:Requirement)-[:SATISFIES]->(c:Commit)
WHERE r.req_id = "REQ-01"
RETURN c.sha, c.commit.message
``` 