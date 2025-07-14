from kaggle_secrets import UserSecretsClient
import os
import subprocess

def push_to_git_from_kaggle(repo_name: str, repo_path: str,commit_message:str=None) -> str:
    original_dir = os.getcwd()
    os.chdir(repo_path)

    try:
        user_secrets = UserSecretsClient()
        username = user_secrets.get_secret("GITHUB_USERNAME")
        token = user_secrets.get_secret("GITHUB_TOKEN")
        remote_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"


        if not commit_message:
            subprocess.run(["git", "commit", "-m", "Commit from Kaggle"], check=False)
        else:
            subprocess.run(["git", "commit", "-m", commit_message], check=False)
        # Rename to main (even if already is)
        subprocess.run(["git", "branch", "-M", "main"], check=False)

        # Check if 'origin' remote exists
        remotes = subprocess.run(["git", "remote"], capture_output=True, text=True, check=True).stdout.split()
        if "origin" in remotes:
            subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)
        else:
            subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)

        # Push to GitHub
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

        return f"✅ Successfully pushed to https://github.com/{username}/{repo_name}"

    except subprocess.CalledProcessError as e:
        return f"❌ Git command failed: {e}"
    except Exception as e:
        return f"❌ Error: {e}"
    finally:
        os.chdir(original_dir)
