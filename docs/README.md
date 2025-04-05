# Trittention-Transformer Documentation

This directory contains the documentation for the Trittention-Transformer project. The documentation is built using [Sphinx](https://www.sphinx-doc.org/).

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

### Build Commands

To build the HTML documentation:

```bash
cd docs
make html
```

The built documentation will be available in the `build/html` directory.

Other build formats:

```bash
# Build PDF documentation (requires LaTeX)
make latexpdf

# Build a single HTML page
make singlehtml

# Build ePub documentation
make epub
```

## Documentation Structure

- `source/`: Source files for the documentation
  - `conf.py`: Sphinx configuration file
  - `index.rst`: Documentation homepage
  - `*.md`: Markdown content files
  - `api_reference/`: API reference documentation

## Updating the Documentation

1. Edit the relevant `.md` or `.rst` files in the `source` directory
2. Add new pages to the appropriate toctree in `index.rst` or other index files
3. Build the documentation to preview your changes
4. Commit your changes to the repository

## API Documentation

API documentation is automatically generated from docstrings in the code. To update the API documentation:

1. Ensure your code has proper docstrings in Google style format
2. Run `make html` to rebuild the documentation

## Documentation Conventions

- Use Markdown (.md) for content pages
- Use reStructuredText (.rst) for index pages and API documentation
- Follow Google-style docstrings in code
- Include examples in docstrings where appropriate
- Use cross-references to link between pages

## Viewing the Documentation

After building, open `build/html/index.html` in your web browser to view the documentation.
