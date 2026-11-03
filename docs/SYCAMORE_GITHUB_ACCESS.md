# Sycamore terminal access to GitHub

The connected GitHub app is authenticated as **YangzhenyuGao** and can publish to
the Agent3 repository. That app connection does not install a Git credential on
Sycamore. The server's HTTPS credential cache was empty during the access check;
SSH host verification was not yet configured. No credential or private key was
copied, printed or installed during publication.

For future terminal pushes, run these steps yourself on Sycamore:

1. Generate a dedicated key, keeping the existing cluster keys intact. If the
   named file already exists, reuse it or choose another filename instead of
   overwriting it. Enter a passphrase when prompted.

   ```bash
   ssh-keygen -t ed25519 -C "YangzhenyuGao@sycamore" -f ~/.ssh/id_ed25519_github
   cat ~/.ssh/id_ed25519_github.pub
   ```

2. Copy **only the `.pub` file's text** into
   [GitHub → Settings → SSH and GPG keys → New SSH key](https://github.com/settings/ssh/new).
   Use the Authentication Key type and a title such as `Sycamore`.

3. Test the connection:

   ```bash
   ssh -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes -o HostKeyAlgorithms=ssh-ed25519 -T git@github.com
   ```

   On the first connection, verify GitHub's host fingerprint against its
   [official fingerprint page](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)
   before accepting. The currently documented Ed25519 fingerprint is
   `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`.
   A greeting naming `YangzhenyuGao` confirms authentication; GitHub does not
   provide an interactive shell.

4. For normal future Git development, clone the published repository into a
   separate directory, leaving this experimental work area intact:

   ```bash
   GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes' git clone git@github.com:YangzhenyuGao/BIOS774_Agent3_RepresentationLearning.git
   cd BIOS774_Agent3_RepresentationLearning
   git config core.sshCommand 'ssh -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes'
   ```

GitHub's [key-generation guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
describes passphrases and optional ssh-agent use. Never paste a private key or
personal access token into chat or commit one to the repository.
