---
title: Setting Git & Virtualenv
date: 2022-05-06
categories:
  - Setting
  - Git
tags: 
  - Git
  - Virtualenv
---

### Put Local folder into git repo

- Make folder ‘example’ and git repo ‘example

```bash
#in local cmd example folder
git init 

#add remote repo
git remote add origin 'repo https'

#bring files in repo to local
git pull origin master 

#bring local files to git repo
git add .
git commit -m 'updated'
git push orgin master 

#check remote
git remote -v

#check current status
git status

#error: failed to push some refs to 'https://github.com/jmj3047/.git'
#force to push 
git push -f origin master

```

### Setting virtual env in window/linux

```powershell
#****use virtual env no matter what****
>python -m venv env_name
>source env_name/Scripts/activate #window
>source env_name/bin/activate #linux

#put all the version of modules in requirements.txt
>pip install -r requirements.txt
```