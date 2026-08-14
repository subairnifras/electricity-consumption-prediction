# Vercel and GitHub CI/CD setup

## Repository and Vercel root

Push this complete folder to GitHub. When importing the repository into Vercel,
set **Root Directory** to `electricity-firefly-webapp` and keep the detected
Flask preset.

## Required GitHub repository secrets

Open **GitHub repository > Settings > Secrets and variables > Actions** and add:

- `VERCEL_TOKEN`: a permanent Vercel access token created for the team that
  owns this project. Do not use `VERCEL_OIDC_TOKEN` from `.env.local`.
- `VERCEL_ORG_ID`: the Vercel Team ID beginning with `team_`.
- `VERCEL_PROJECT_ID`: the Vercel Project ID beginning with `prj_`.

Do not put quotes around secret values and do not include `NAME=`.

## Pipeline behavior

Pull requests run CI only. A push to `main` runs CI and deploys to production
only after CI succeeds. The authentication step intentionally fails early with
a clear error if `VERCEL_TOKEN` is invalid or scoped to the wrong account.

## Local commands

```powershell
py -m venv venv
venv\Scripts\activate
py -m pip install -r electricity-firefly-webapp\requirements.txt
cd electricity-firefly-webapp
py app.py
```

## Push changes

```powershell
git add .
git commit -m "Configure Vercel CI CD"
git push origin main
```

