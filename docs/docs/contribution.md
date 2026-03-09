---
icon: fontawesome/solid/pen
title: Contribution
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Reporting an Issue

Before reporting an issue, please check if a similar issue already exists in our [GitHub Issues](https://github.com/open-mosaic/mosaic/issues). If you find a relevant issue, add your information as a comment rather than creating a duplicate.

When creating a new issue, please include:

- **Clear description** - What problem are you experiencing?
- **Steps to reproduce** - Provide detailed steps to reproduce the issue
- **Expected behavior** - What should happen?
- **Actual behavior** - What actually happens?
- **Environment details** - Include relevant version numbers (Mosaic, NCCL, OpenTelemetry, etc.)
- **Screenshots or logs** - If applicable, attach screenshots or relevant log output

Good issue reports help maintainers understand and fix problems more efficiently.

# Raise a PR

We welcome contributions through pull requests! Here's a high-level overview of our contribution workflow:

## Before You Start

- Check existing issues and PRs to ensure your contribution isn't already being worked on
- For significant changes, open an issue first to discuss your approach

## Development Workflow

- Write clean, well-documented code following the project's coding standards
- Test your changes to ensure they work correctly and don't break existing functionality
- When submitting a PR, provide a clear description of your changes, link related issues, and explain the motivation

!!! tips "Developer Certificate of Origin (DCO)"
    We require that every commit in a PR be signed off under the [Developer Certificate of Origin](https://developercertificate.org/).
    The sign-off attests that you have the right to submit the contribution (e.g., you wrote the code or have permission from the copyright holder) and that you agree to license it under the project's terms.

    **How to comply:** Sign off each commit when you create it:

    ```bash
    git commit -s -m "Your commit message"
    ```

    The `-s` flag adds a `Signed-off-by:` line to your commit message.

## Review Process

- **Automated checks** - Your PR will run through CI/CD pipelines (tests, linting, etc.)
- **Code review** - Maintainers will review your code for quality, correctness, and alignment with project goals
- **Feedback and iteration** - Be open to feedback and ready to make requested changes
- **Approval and merge** - Once approved and all checks pass, a maintainer will merge your PR

## Best Practices

- Keep PRs focused and reasonably sized - smaller PRs are easier to review
- Update documentation if your changes affect user-facing features
- Add tests for new functionality or bug fixes
- Respond promptly to review feedback
- Be patient - maintainers are volunteers and may take time to review

## Contribute Test Cases

When adding new test cases, follow these guidelines:

- **Place tests in the correct location** - Add test files to the appropriate suite under `tests/suites`. Create a new suite if testing a new component or feature.
- **Keep test files focused** - Each test file should only contain test cases. Move fixtures and constants to `conftest.py`.
- **Include required docstrings** - Every test function must have a docstring with `:title:`, `:suite:`, and `:description:` fields.
- **Use descriptive names** - Test function names should clearly indicate what is being tested.
- **Add assertions with messages** - Always include descriptive error messages in assertions to aid debugging.
- **Register custom markers** - If adding new pytest markers, register them in `tests/suites/conftest.py`.
- **Follow the Python style guide** - Follow the PEP 8 coding style.
- **Clean up resources** - Use fixture teardown or context managers to ensure proper cleanup after tests.

# Update Documentation

Mosaic documentation is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Serve docs locally

The documentation requires `python` and `pip`. To set up your local environment and test your changes:

```bash
git clone git@github.com:open-mosaic/mosaic.git
cd docs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve --strict
```

!!! tip "Bypass spellcheck"
    - Update `docs/docs/dictionary.txt` for words that should be ignored by spellcheck
    - Update `docs/mkdocs.yaml` to exclude files entirely

The development server will be available at [http://localhost:8000](http://localhost:8000).

# Other Ways of Contributing

Contributing code and documentation isn't the only way to help! Here are other valuable ways to contribute to Mosaic:

- **Infrastructure contributions** - Contribute GPU runners to our self-hosted runner pool to help accelerate CI/CD pipelines and testing
- **Community support** - Help answer questions in GitHub Discussions, review other contributors' PRs, or mentor newcomers
- **Testing and validation** - Test Mosaic on different hardware configurations, CUDA versions, or distributed setups and report your findings
- **Performance analysis** - Run benchmarks, identify bottlenecks, or share optimization strategies and results
- **Example projects** - Create sample integrations, tutorials, or use-case demonstrations that help others adopt Mosaic
- **Outreach** - Write blog posts, speak at conferences, or share your Mosaic experiences with the community

Every contribution, big or small, helps make Mosaic better for everyone!
