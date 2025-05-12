"""
Module for extracting GitHub issues and commit history.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from pathlib import Path
import subprocess

from constants import (
    GITHUB_API_TOKEN, GITHUB_REPO_OWNER, GITHUB_REPO_NAME,
    ADA_CODE_DIR, OUTPUT_DIR
)

class GitHubIssue:
    """Represents a GitHub issue node in the graph."""
    def __init__(self, issue_id: int, title: str, body: str, 
                 created_at: str, updated_at: str, state: str):
        self.id = f"issue-{issue_id}"
        self.issue_id = issue_id
        self.title = title
        self.body = body if body else ""
        self.created_at = created_at
        self.updated_at = updated_at
        self.state = state
        self.embedding = None
        self.related_nodes = []
    
    def __repr__(self):
        return f"Issue #{self.issue_id}: {self.title}"

class GitCommit:
    """Represents a Git commit node in the graph."""
    def __init__(self, commit_hash: str, author: str, date: str, 
                 message: str, changed_files: List[str]):
        self.id = f"commit-{commit_hash[:8]}"
        self.commit_hash = commit_hash
        self.author = author
        self.date = date
        self.message = message
        self.changed_files = changed_files
        self.embedding = None
        self.related_nodes = []
    
    def __repr__(self):
        return f"Commit {self.commit_hash[:8]}: {self.message.split(chr(10))[0]}"

def fetch_github_issues() -> List[GitHubIssue]:
    """Fetch GitHub issues using the GitHub API."""
    issues = []
    
    # If GitHub API credentials are not set, return empty list
    if not GITHUB_API_TOKEN or not GITHUB_REPO_OWNER or not GITHUB_REPO_NAME:
        print("GitHub API credentials not set, skipping issues fetching")
        return issues
    
    headers = {
        "Authorization": f"token {GITHUB_API_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    page = 1
    while True:
        url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues"
        params = {
            "state": "all",
            "per_page": 100,
            "page": page
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching issues: {response.status_code}")
            break
        
        data = response.json()
        if not data:
            break
        
        for issue_data in data:
            # Skip pull requests
            if "pull_request" in issue_data:
                continue
            
            issue = GitHubIssue(
                issue_id=issue_data["number"],
                title=issue_data["title"],
                body=issue_data.get("body", ""),
                created_at=issue_data["created_at"],
                updated_at=issue_data["updated_at"],
                state=issue_data["state"]
            )
            issues.append(issue)
        
        page += 1
    
    return issues

def fetch_commit_history() -> List[GitCommit]:
    """Fetch Git commit history."""
    commits = []
    
    try:
        # Change to the repository directory
        os.chdir(ADA_CODE_DIR)
        
        # Get git commit log in JSON format
        git_log_cmd = [
            "git", "log", "--pretty=format:{%n  \"commit\": \"%H\",%n  \"author\": \"%an\",%n  \"date\": \"%ad\",%n  \"message\": \"%s\"%n},",
            "--date=iso", "--name-only"
        ]
        
        git_log_output = subprocess.check_output(git_log_cmd, universal_newlines=True)
        
        # Format the output as valid JSON
        git_log_json = f"[{git_log_output.rstrip(',')}]"
        
        try:
            # Parse the JSON
            commit_data = json.loads(git_log_json)
            
            # Process each commit
            for i, commit in enumerate(commit_data):
                # Get changed files
                if i < len(commit_data) - 1:
                    lines = commit_data[i+1]["message"].split('\n')
                    changed_files = [line for line in lines if line.strip()]
                else:
                    changed_files = []
                
                # Create commit node
                commit_node = GitCommit(
                    commit_hash=commit["commit"],
                    author=commit["author"],
                    date=commit["date"],
                    message=commit["message"],
                    changed_files=changed_files
                )
                commits.append(commit_node)
        
        except json.JSONDecodeError as e:
            print(f"Error parsing Git log JSON: {e}")
            # Fallback to simple approach if JSON parsing fails
            commits = fetch_commit_history_simple()
    
    except subprocess.SubprocessError as e:
        print(f"Error running Git log command: {e}")
        # Fallback to simple approach if command fails
        commits = fetch_commit_history_simple()
    
    finally:
        # Return to original directory
        os.chdir(Path(__file__).parent)
    
    return commits

def fetch_commit_history_simple() -> List[GitCommit]:
    """Fetch Git commit history with a simpler approach."""
    commits = []
    
    try:
        # Change to the repository directory
        os.chdir(ADA_CODE_DIR)
        
        # Get git commit log with simpler format
        git_log_cmd = [
            "git", "log", "--pretty=format:%H|%an|%ad|%s", "--date=iso"
        ]
        
        git_log_output = subprocess.check_output(git_log_cmd, universal_newlines=True)
        git_log_lines = git_log_output.strip().split('\n')
        
        for line in git_log_lines:
            if not line:
                continue
            
            parts = line.split('|', 3)
            if len(parts) != 4:
                continue
            
            commit_hash, author, date, message = parts
            
            # Create commit node
            commit_node = GitCommit(
                commit_hash=commit_hash,
                author=author,
                date=date,
                message=message,
                changed_files=[]  # No changed files in simple mode
            )
            commits.append(commit_node)
    
    except subprocess.SubprocessError as e:
        print(f"Error running simple Git log command: {e}")
    
    finally:
        # Return to original directory
        os.chdir(Path(__file__).parent)
    
    return commits

def extract_github_data() -> Tuple[List[GitHubIssue], List[GitCommit]]:
    """Extract GitHub issues and commit history."""
    # First, try to load from local JSON files using convert_repo_artifacts
    try:
        from convert_repo_artifacts import load_artifacts
        issues, commits = load_artifacts()
        
        # If we got data from the files, return it
        if issues and commits:
            print("Using issues and commits from local JSON files")
            return issues, commits
    except (ImportError, Exception) as e:
        print(f"Could not load from JSON files: {e}")
    
    # If loading from files failed, fall back to API and Git methods
    print("Falling back to API and Git methods for issues and commits")
    issues = fetch_github_issues()
    commits = fetch_commit_history()
    
    return issues, commits

if __name__ == "__main__":
    issues, commits = extract_github_data()
    
    print(f"Extracted {len(issues)} GitHub issues")
    for i, issue in enumerate(issues[:5]):
        print(f"{i+1}. {issue}")
    
    print(f"\nExtracted {len(commits)} Git commits")
    for i, commit in enumerate(commits[:5]):
        print(f"{i+1}. {commit}") 