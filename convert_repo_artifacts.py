"""
Script to convert JSON files with issues and commits into the format needed by the system.
Provides detailed status information about the processing of JSON files.
"""
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple

from github_integration import GitHubIssue, GitCommit

def convert_issues_from_file(file_path: str) -> List[GitHubIssue]:
    """Convert issues from a JSON file to GitHubIssue objects."""
    print(f"📝 Converting issues from {file_path}...")
    start_time = time.time()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            issues_data = json.load(f)
        
        file_size_kb = os.path.getsize(file_path) / 1024
        print(f"✅ Loaded {len(issues_data)} issues from {file_path} ({file_size_kb:.2f} KB)")
        
        issues = []
        print(f"🔄 Processing {len(issues_data)} issues...")
        
        for i, issue_data in enumerate(issues_data):
            # Print progress for every 50th issue or first/last issue
            if i % 50 == 0 or i == len(issues_data) - 1:
                print(f"  ⏳ Processing issue {i+1}/{len(issues_data)}")
            
            issue = GitHubIssue(
                issue_id=issue_data.get("number", 0),
                title=issue_data.get("title", ""),
                body=issue_data.get("body", ""),
                created_at=issue_data.get("created_at", ""),
                updated_at=issue_data.get("updated_at", ""),
                state=issue_data.get("state", "")
            )
            issues.append(issue)
        
        end_time = time.time()
        print(f"✅ Converted {len(issues)} issues in {end_time - start_time:.2f} seconds")
        
        # Print sample issues
        if issues:
            print("📋 Sample issues:")
            for i, issue in enumerate(issues[:3]):
                print(f"  - Issue #{issue.issue_id}: {issue.title[:50]}...")
            if len(issues) > 3:
                print(f"  - ... and {len(issues) - 3} more issues")
        
        return issues
    
    except Exception as e:
        print(f"❌ Error converting issues: {str(e)}")
        return []

def convert_commits_from_file(file_path: str) -> List[GitCommit]:
    """Convert commits from a JSON file to GitCommit objects."""
    print(f"📝 Converting commits from {file_path}...")
    start_time = time.time()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            commits_data = json.load(f)
        
        file_size_kb = os.path.getsize(file_path) / 1024
        print(f"✅ Loaded {len(commits_data)} commits from {file_path} ({file_size_kb:.2f} KB)")
        
        commits = []
        print(f"🔄 Processing {len(commits_data)} commits...")
        
        file_changes_total = 0
        for i, commit_data in enumerate(commits_data):
            # Print progress for every 50th commit or first/last commit
            if i % 50 == 0 or i == len(commits_data) - 1:
                print(f"  ⏳ Processing commit {i+1}/{len(commits_data)}")
            
            # Extract changed files
            changed_files = commit_data.get("files", [])
            if isinstance(changed_files, list):
                file_paths = [file.get("filename", "") for file in changed_files]
                file_changes_total += len(file_paths)
            else:
                file_paths = []
            
            commit = GitCommit(
                commit_hash=commit_data.get("sha", ""),
                author=commit_data.get("commit", {}).get("author", {}).get("name", ""),
                date=commit_data.get("commit", {}).get("author", {}).get("date", ""),
                message=commit_data.get("commit", {}).get("message", ""),
                changed_files=file_paths
            )
            commits.append(commit)
        
        end_time = time.time()
        print(f"✅ Converted {len(commits)} commits with {file_changes_total} file changes in {end_time - start_time:.2f} seconds")
        
        # Print sample commits and statistics
        if commits:
            print("📋 Sample commits:")
            for i, commit in enumerate(commits[:3]):
                file_count = len(commit.changed_files)
                print(f"  - Commit {commit.commit_hash[:8]}: {commit.message[:50]}... ({file_count} files)")
            if len(commits) > 3:
                print(f"  - ... and {len(commits) - 3} more commits")
            
            # Calculate average files per commit
            avg_files = file_changes_total / len(commits) if commits else 0
            print(f"📊 Average files per commit: {avg_files:.2f}")
        
        return commits
    
    except Exception as e:
        print(f"❌ Error converting commits: {str(e)}")
        return []

def load_artifacts() -> Tuple[List[GitHubIssue], List[GitCommit]]:
    """Load issues and commits from JSON files."""
    print("🔄 Loading artifacts from JSON files...")
    start_time = time.time()
    
    # Paths to the JSON files
    issues_file = "issues.json"
    commits_file = "commits.json"
    
    # Check if files exist
    issues_exist = os.path.exists(issues_file)
    commits_exist = os.path.exists(commits_file)
    
    # Load issues
    issues = []
    if issues_exist:
        try:
            issues = convert_issues_from_file(issues_file)
            print(f"✅ Loaded {len(issues)} issues from {issues_file}")
        except Exception as e:
            print(f"❌ Error loading issues: {e}")
    else:
        print(f"⚠️ Issues file {issues_file} not found")
    
    # Load commits
    commits = []
    if commits_exist:
        try:
            commits = convert_commits_from_file(commits_file)
            print(f"✅ Loaded {len(commits)} commits from {commits_file}")
        except Exception as e:
            print(f"❌ Error loading commits: {e}")
    else:
        print(f"⚠️ Commits file {commits_file} not found")
    
    end_time = time.time()
    print(f"✅ Loaded {len(issues)} issues and {len(commits)} commits in {end_time - start_time:.2f} seconds")
    
    return issues, commits

if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Starting artifact conversion...")
    
    issues, commits = load_artifacts()
    
    # Print overall statistics
    print("\n📊 Conversion statistics:")
    print(f"  - Issues: {len(issues)}")
    print(f"  - Commits: {len(commits)}")
    
    # Print sample issues
    print("\n📋 Sample Issues:")
    for i, issue in enumerate(issues[:5]):
        print(f"{i+1}. {issue}")
    if len(issues) > 5:
        print(f"... and {len(issues) - 5} more")
    
    # Print sample commits
    print("\n📋 Sample Commits:")
    for i, commit in enumerate(commits[:5]):
        print(f"{i+1}. {commit}")
    if len(commits) > 5:
        print(f"... and {len(commits) - 5} more")
    
    end_time = time.time()
    print(f"\n✅ Conversion completed in {end_time - start_time:.2f} seconds")
