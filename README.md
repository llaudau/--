# --
eval "$(ssh-agent -s)"

ssh-add ~/.ssh/id_ed25519

ssh -T git@github.com

'The command needed to start agent and connect to the github'

git add -u

'Track all updates in the local repository'

git add .

'when i m vsome new file to repository, it will check the changes'

git commit "name_of_commit"

'admit the change and save the change'

git push 

'i've set the default(use git push -u origin main) to push to branch named "main", so this command push it to the directory'