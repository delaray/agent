# UV CLI Commands

## Adding Github Repositories to UV

### CLI Command

    uv add git+ssh://git@github.com/delaray/aitils --rev main

### pyproject.toml entry

    [project]
    dependencies = ["httpx"]

    [tool.uv.sources]
    httpx = { git = "https://github.com/encode/httpx", branch = "main" }
