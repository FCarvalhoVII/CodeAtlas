# Security policy

## Supported versions

Security fixes are provided for the latest released minor version.

## Reporting a vulnerability

Do not open a public issue containing an exploit, credential, or private repository data. Use the repository host's private security-advisory channel and include the affected version, reproduction steps, impact, and any suggested mitigation. Maintainers should acknowledge a report within seven days and coordinate disclosure after a fix is available.

## Security boundaries

CodeAtlas treats repository contents as untrusted data. The scanner does not follow symlinks, bounds individual reads, and the Python parser never imports scanned modules. Indexes contain source snippets in plain text and must be protected like the repository. Remote providers receive data only when explicitly configured; see [provider privacy](docs/providers.md) and [known limitations](docs/limitations.md).
