# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in Neurova, please report it responsibly. **Do not open a public issue** for security concerns.

### How to Report

Send security reports to: [Contact email to be added]

Include in your report:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

### What to Expect

- We will acknowledge your report within 48 hours
- We will provide an estimated timeline for a fix
- We will keep you informed of progress
- We will credit you for the discovery (unless you prefer to remain anonymous)

## Supported Versions

Security updates will be provided for:

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1.0 | No        |

## Security Best Practices

### When Using Neurova

- Keep your installation up to date
- Use virtual environments to isolate dependencies
- Validate input data before processing
- Be cautious when loading models or data from untrusted sources
- Review dependency security with `pip audit` or similar tools

### For Contributors

- Never commit credentials, API keys, or sensitive data
- Use environment variables for configuration
- Validate user inputs thoroughly
- Follow secure coding practices
- Run security linters before submitting code

## Known Security Considerations

### File I/O Operations

- Image loading uses well-tested libraries (NumPy, Pillow)
- Always validate file paths and permissions
- Be aware of potential issues with malformed image files

### GPU Operations (CuPy)

- GPU memory is managed by CUDA runtime
- Ensure NVIDIA drivers are up to date
- Be cautious with untrusted CUDA code

### Dependencies

- NumPy, SciPy, Pillow, CuPy are all from trusted sources
- We use only permissive licenses (MIT, BSD)
- Dependencies are minimal to reduce attack surface

## Updates and Patches

Security patches will be released as soon as possible after verification. Check:

- GitHub Security Advisories
- Release notes in CHANGELOG.md
- Project announcements

## Vulnerability Disclosure

After a fix is released:

- We will publish a security advisory
- We will detail the impact and affected versions
- We will credit the reporter (if they agree)

## Questions

For general security questions, open a GitHub discussion or issue (non-sensitive topics only).

Thank you for helping keep Neurova secure.
