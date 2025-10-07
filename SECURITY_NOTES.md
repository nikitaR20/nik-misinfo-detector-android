Security notes and remediation steps

1) The repository contains a hard-coded Google API key in `backend/fact_checker.py`.

2) Immediate actions (recommended):
   - Revoke the exposed API key in Google Cloud Console immediately.
   - Rotate the key and update your environment variables to use the new key.

3) Remove the secret from the repository history:
   - Remove the secret from the current working tree and stop tracking it:
     git rm --cached backend/fact_checker.py
     # Edit the file to remove the secret or replace it with os.getenv
     git add backend/fact_checker.py
     git commit -m "Remove hard-coded credentials"

   - Rewrite history to purge the secret from past commits (use `git filter-repo` or BFG):
     # Example using git filter-repo (recommended over filter-branch):
     pip install git-filter-repo
     git filter-repo --path backend/fact_checker.py --invert-paths

   - If you can't remove the file entirely, use the replace-text feature of BFG or filter-repo to scrub the key.

4) Ensure secrets are stored in `.env` (which is in `.gitignore`) or a secrets manager.

5) Optional follow-ups:
   - Add automated secret scanning (GitHub Advanced Security, pre-commit hooks like detect-secrets)
   - Add a CI/CD step to ensure `.env` is not committed

