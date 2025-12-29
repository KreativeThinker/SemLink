# Contributing to SemLink

Thank you for your interest in contributing to SemLink! This guide will help you get set up and understand our development workflow.

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Project Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/KreativeThinker/SemLink
   cd SemLink
   ```

2. **Install dependencies using uv (recommended)**
   ```bash
   uv sync --dev
   ```

   Or with pip:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Set up pre-commit hooks**
   ```bash
   uv run pre-commit install
   ```

4. **Verify installation**
   ```bash
   uv run semlink --help
   ```

## 🔧 Development Workflow

### Code Style & Formatting

We use **Ruff** for both linting and formatting with these standards:
- Line length: 88 characters
- Quote style: Double quotes
- Target: Python 3.13+

**Before committing, always run:**
```bash
uv run ruff check .        # Lint code
uv run ruff format .       # Format code
```

Or let pre-commit handle it automatically:
```bash
uv run pre-commit run --all-files
```

### Code Guidelines

- **Type hints**: Use comprehensive type annotations on all functions
- **Imports**: Use absolute imports, explicit imports (no wildcards)
- **Naming**: 
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - Descriptive names that explain purpose
- **Error handling**: Use custom `AppError` base class for application errors
- **Architecture**: Follow single responsibility principle, separate CLI from core logic

## 📝 Commit Messages

We use **Conventional Commits** with [Commitizen](https://commitizen-tools.github.io/commitizen/). 

**Please read the [Commitizen documentation](https://commitizen-tools.github.io/commitizen/commit/) to understand proper commit message format.**

Examples of good commit messages:
- `feat: add semantic note linking functionality`
- `fix: resolve issue with graph visualization rendering`
- `docs: update installation instructions`
- `refactor: simplify core task execution logic`

**Use commitizen to help create proper commits:**
```bash
uv run cz commit
```

## 🐛 Working with Issues

### Finding Work
- Browse [GitHub Issues](../../issues) for available tasks
- Issues are labeled with:
  - **Priority**: `P0` (Highest), `P1`, `P2` (Least)
  - **Size**: `XS`, `S`, `M`, `L`, `XL`
  - **Type**: `bug`, `feature`, `enhancement`, `documentation`, `refactor`, `plan`

### Taking an Issue
1. Comment on the issue to indicate you're working on it
2. Create a new branch: `git checkout -b feature/issue-description` or `git checkout -b fix/issue-description`
3. Read the issue description carefully - it contains objectives and acceptance criteria

### Submitting Changes
1. Ensure all formatting and linting passes
2. Write meaningful commit messages following conventional commits
3. Push your branch and create a Pull Request
4. Link the PR to the relevant issue
5. Request review from maintainers

## 🧪 Testing

Currently, no test framework is configured. When adding tests:
- Follow the established patterns in the codebase
- Ensure tests cover both success and error cases
- Add tests for any new functionality
