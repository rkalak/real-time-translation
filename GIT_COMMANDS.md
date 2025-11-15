# Git Commands Reference

## Initial Setup (First Time)

### 1. Initialize Git Repository
```bash
cd /Users/rahulkalakuntla/starlife
git init
```

### 2. Create Repository on GitHub
- Go to https://github.com/new
- Create a new repository (e.g., `starlife`)
- **DO NOT** initialize with README, .gitignore, or license (we already have files)

### 3. Add Remote Repository
```bash
git remote add origin https://github.com/YOUR_USERNAME/starlife.git
# Replace YOUR_USERNAME with your actual GitHub username
```

### 4. Add All Files
```bash
git add .
```

### 5. Make First Commit
```bash
git commit -m "Initial commit: Real-time voice translation pipeline"
```

### 6. Push to GitHub
```bash
git branch -M main
git push -u origin main
```

---

## Daily Workflow (Making Changes)

### 1. Check Status
```bash
git status
```

### 2. Add Specific Files
```bash
# Add a specific file
git add real-time-voice-middleware/pipeline.py

# Add all changes
git add .

# Add all changes in a directory
git add real-time-voice-middleware/
```

### 3. Commit Changes
```bash
git commit -m "Description of your changes"
```

**Good commit message examples:**
```bash
git commit -m "Fix TTS latency by reducing voice stability"
git commit -m "Add consolidated pipeline file"
git commit -m "Update LLM prompt to prevent extra words"
```

### 4. Push to GitHub
```bash
git push
```

**If pushing for the first time to a branch:**
```bash
git push -u origin main
```

---

## Useful Commands

### View Changes
```bash
# See what files changed
git status

# See detailed changes
git diff

# See changes for a specific file
git diff real-time-voice-middleware/pipeline.py
```

### View History
```bash
# See commit history
git log

# See commit history (one line per commit)
git log --oneline

# See last 5 commits
git log -5
```

### Undo Changes
```bash
# Undo changes to a file (before committing)
git checkout -- real-time-voice-middleware/pipeline.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

### Update from Remote
```bash
# Pull latest changes from GitHub
git pull
```

### Create a New Branch
```bash
# Create and switch to new branch
git checkout -b feature-name

# Switch back to main
git checkout main

# List all branches
git branch
```

---

## Complete Workflow Example

```bash
# 1. Navigate to project
cd /Users/rahulkalakuntla/starlife

# 2. Check what changed
git status

# 3. Add your changes
git add .

# 4. Commit with a message
git commit -m "Add TTS latency diagnostic report"

# 5. Push to GitHub
git push
```

---

## Troubleshooting

### If you get "remote origin already exists"
```bash
# Remove existing remote
git remote remove origin

# Add it again
git remote add origin https://github.com/YOUR_USERNAME/starlife.git
```

### If you get authentication errors
```bash
# Use GitHub CLI (if installed)
gh auth login

# Or use SSH instead of HTTPS
git remote set-url origin git@github.com:YOUR_USERNAME/starlife.git
```

### If you need to force push (be careful!)
```bash
# Only use if you're sure - this overwrites remote history
git push --force
```

---

## Quick Reference Card

| Action | Command |
|--------|---------|
| Check status | `git status` |
| Add all files | `git add .` |
| Commit | `git commit -m "message"` |
| Push | `git push` |
| Pull | `git pull` |
| View history | `git log --oneline` |
| View changes | `git diff` |

