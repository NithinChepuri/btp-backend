# Requirement Traceability Analysis for Ada Code

This project implements a traceability system for Ada code that connects requirements to code files, following the approach from the "Requirements2Code" paper.

## Analyze Single Requirement

We've successfully analyzed a single authentication requirement against the Ada AWA codebase. The analysis uses a combination of semantic similarity (70% weight) and keyword matching (30% weight) to find the most relevant code files.

### Authentication Requirement

The analyzed requirement:

```
The system must provide a secure authentication mechanism for users to log in with their credentials. The authentication process should validate user credentials against stored data and enforce password policies including minimum length and complexity requirements. Failed login attempts should be recorded and after a configurable number of failed attempts, the account should be temporarily locked.
```

### Top 10 Most Relevant Files

The analysis identified the following files as most relevant to the authentication requirement:

1. `awa-sysadmin-beans.adb` (Score: 0.3227)
2. `awa-users-services-tests.adb` (Score: 0.3119)
3. `awa-users-servlets.ads` (Score: 0.3083)
4. `awa-users-services.ads` (Score: 0.3075)
5. `awa-users-tests.adb` (Score: 0.2983)
6. `awa-users-filters.ads` (Score: 0.2914)
7. `security-auth-fake.ads` (Score: 0.2875)
8. `awa-users-beans.adb` (Score: 0.2785)
9. `awa-sysadmin-filters.ads` (Score: 0.2686)
10. `awa-tests-helpers-users.adb` (Score: 0.2595)

## Neo4j Visualization

The analysis results have been loaded into Neo4j for visualization. Neo4j provides a graph view of the requirement, related files, and code elements.

### Accessing Neo4j Browser

1. Ensure Neo4j is running
2. Open a web browser and navigate to `http://localhost:7474/`
3. Log in with the credentials (usually neo4j/password or as configured in your constants.py)

### Useful Cypher Queries

Once logged into Neo4j Browser, you can run the following queries to explore the traceability data:

1. View the requirement and related files:
   ```cypher
   MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File) 
   RETURN r, rel, f 
   ORDER BY rel.rank
   ```

2. View the requirement and all traced code elements:
   ```cypher
   MATCH (r:Requirement)-[rel:TRACED_TO]->(n:AdaNode) 
   RETURN r, rel, n 
   ORDER BY rel.score DESC
   ```

3. View files and their code elements:
   ```cypher
   MATCH (f:File)-[:CONTAINS]->(n:AdaNode) 
   RETURN f, n 
   LIMIT 100
   ```

4. View code hierarchy:
   ```cypher
   MATCH p=(n:AdaNode)-[:CONTAINS*]->(c) 
   RETURN p 
   LIMIT 100
   ```

5. View code references:
   ```cypher
   MATCH p=(n:AdaNode)-[:REFERENCES]->(c) 
   RETURN p 
   LIMIT 100
   ```

6. Find specific types of Ada nodes (e.g., procedures related to authentication):
   ```cypher
   MATCH (r:Requirement {req_id: "AUTH-01"})-[:TRACED_TO]->(n:AdaNode)
   WHERE n.type = "PROCEDURE" AND toLower(n.name) CONTAINS "auth"
   RETURN r, n
   ```

## Running Your Own Analysis

To analyze a different requirement:

1. Create a text file with your requirement (e.g., `my_requirement.txt`)
2. Run the analysis:
   ```bash
   python analyze_single_req.py --req-file my_requirement.txt --req-id MY-REQ-01
   ```

3. Load the results into Neo4j:
   ```bash
   python load_req_to_neo4j.py MY-REQ-01
   ```

## Understanding the Authentication Code

The authentication system implemented in the AWA codebase provides:

1. **User credential validation**: The service verifies email/password combinations against stored data.
2. **Password policy enforcement**: Passwords must meet length and complexity requirements.
3. **Failed login tracking**: The system records login attempts and can lock accounts after configurable number of failures.
4. **Secure password storage**: Passwords are stored using HMAC-SHA1 hashing with salt.

Key components:
- `awa-users-services.ads/adb`: Core authentication service implementation
- `awa-users-servlets.ads/adb`: Web interface for authentication
- `awa-users-beans.adb`: UI components for authentication
- `awa-users-filters.ads`: Request filtering for authentication 