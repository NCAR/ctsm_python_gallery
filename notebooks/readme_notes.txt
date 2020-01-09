# on the web, fork the ctsm_py repository onto your personal github repo

# from the comand line, clone your fork to your local directory
  1004  11:24   git clone git@github.com:wwieder/ctsm_py.git

# work on a notebook, or copy a file into the notebook directory
  1010  11:26   cp SimpleExample.ipynb ../ctsm_py/notebooks/.
  1011  11:26   cd ../ctsm_py/notebooks/
  1012  11:26   ls

# when you're ready, you can add and commit your notebook
  1013  11:26   git add SimpleExample.ipynb
  1014  11:26   git commit -m "simple file IO, averaging, & plotting"
    # or just   git commit

# have a look at what things look like
                git log --decorate --oneline --graph

  1016  11:27   git remote
  1017  11:27   git remote -v

# link the to your remote master & call it 'upstream'
  1018  11:28   git remote add upstream git@github.com:NCAR/ctsm_py.git

# you can try to fetch the remote master (compined pull and merge)
# pull from the upstream (remote) master
  1019  11:28   git pull upstream master
                git diff upstream/master
# if there are conflicts you can change where the head of the branch is
                git rebase upstream/master
  1020  11:28   git status

# push to the your remote fork, this could be to a branch too
# add -f to force the push, above
  1021  11:29   git push origin master

# then go to the web and submit a pull request


