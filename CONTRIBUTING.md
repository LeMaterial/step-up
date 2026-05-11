# Contributing

Thanks for helping improve `step-up`.

## Development Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/LeMaterial/step-up.git
cd step-up
uv sync --dev
```

Install the pre-commit hook:

```bash
uv run pre-commit install
```

Run the full local check suite:

```bash
uv run pre-commit run --all-files
```

## Common Commands

Format code:

```bash
uv run ruff format
```

Lint code:

```bash
uv run ruff check
```

Fix lint issues where possible:

```bash
uv run ruff check --fix
```

Run tests:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=step_up
```

## Signed Commits

Commits must be signed before they can be merged into protected branches.

To create a GPG key:

```bash
gpg --full-generate-key
```

List your secret keys and copy the key ID:

```bash
gpg --list-secret-keys --keyid-format=long
```

Configure Git to sign commits globally with that key:

```bash
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true
```

Export the public key and add it to GitHub under
`Settings -> SSH and GPG keys -> New GPG key`:

```bash
gpg --armor --export <KEY_ID>
```

After global signing is enabled, normal commits are signed automatically:

```bash
git commit -m "Your commit message"
```

To sign one commit explicitly:

```bash
git commit -S -m "Your commit message"
```

Verify the latest commit signature:

```bash
git log --show-signature -1
```

## Before Opening A Pull Request

Run:

```bash
uv run pre-commit run --all-files
```

If Ruff modifies files, stage the changes and run the command again.

Make sure commits are signed. The `main` branch is expected to require signed
commits and passing status checks before changes can be merged.
