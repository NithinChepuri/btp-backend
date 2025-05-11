import requests
import json

owner = 'stcarrez'
repo = 'ada-awa'

def get_issues():
    url = f'https://api.github.com/repos/{owner}/{repo}/issues'
    params = {'state': 'all', 'per_page': 100}
    response = requests.get(url, params=params)
    return [issue for issue in response.json() if 'pull_request' not in issue]

def get_commits():
    url = f'https://api.github.com/repos/{owner}/{repo}/commits'
    params = {'per_page': 100}
    response = requests.get(url, params=params)
    return response.json()

def save_to_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def main():
    print("📄 Fetching issues...")
    issues = get_issues()
    save_to_file('issues.json', issues)
    print(f"✅ Saved {len(issues)} issues to issues.json")

    print("\n🔨 Fetching commits...")
    commits = get_commits()
    save_to_file('commits.json', commits)
    print(f"✅ Saved {len(commits)} commits to commits.json")

if __name__ == '__main__':
    main()
