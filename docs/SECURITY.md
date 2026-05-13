# ARGUS Security Notes

ARGUS is configured for **local demos only**. Do not expose the demo stack directly to the public internet.

## API Protection

All non-probe API routes require `ARGUS_API_KEY`. Probe endpoints (`/health`, `/ready`) are exempt.

Supported auth mechanisms:

| Method | Usage |
|---|---|
| `X-ARGUS-API-Key: <key>` | Header-based API key (recommended) |
| `Authorization: Bearer <key>` | Bearer token |
| HTTP Basic | Dashboard only; username `argus`, password is `ARGUS_API_KEY` |

### Admin Endpoints

`DELETE /phase3/ueba/risks` requires `ARGUS_ADMIN_API_KEY`, provided via:

- `X-ARGUS-Admin-Key` header (preferred)
- `X-ARGUS-API-Key` header (if it matches the admin key)
- Bearer or Basic auth (if the token matches the admin key)

### Additional Protections

- **CORS**: restricted to configured origins (`ARGUS_CORS_ORIGINS`).
- **Trusted hosts**: request host validation (`ARGUS_TRUSTED_HOSTS`).
- **Rate limiting**: fixed-window per-minute limits, optionally Redis-backed.
- **Request size**: body size cap via `ARGUS_MAX_REQUEST_BYTES` (default 1 MB).
- **Constant-time comparison**: API key validation uses `hmac.compare_digest`.

## Local Infrastructure Security

The Docker Compose stack disables or omits production-grade security for convenience:

| Service | Security Status |
|---|---|
| Elasticsearch | Security disabled (`xpack.security.enabled=false`) |
| Kafka | Plaintext listeners, no SASL/TLS |
| Redis | No auth, no TLS |
| MLflow | Local volume storage, no auth |

Use this compose stack only on a trusted local machine.

## Browser Security

The dashboard renders API data using `textContent` and sanitized CSS class names. No dynamic HTML injection is used. Alert fields, user IDs, host IDs, and session IDs are rendered as text nodes only.

## Secrets Management

- `.env` is gitignored. Do not commit real API keys.
- `.env.example` contains safe placeholder values only.
- Model bundles, tokenized datasets, and session parquets are gitignored.

Before a public push:

```bash
git status --short
git grep -n "sk-" -- .
```

Also run a dedicated secret scanner (Gitleaks, GitHub secret scanning, or `trufflehog`).

## Dependency Audit

`pip-audit` is included in the dev toolchain and CI. It is configured to report but not fail CI due to known dependency-policy conflicts:

- `drain3==0.9.11` pins `cachetools==4.2.1`, which blocks newer MLflow versions.
- MLflow 2.x caps `pyarrow<16`, which blocks audited PyArrow 17+.
- The current Transformers advisory points to a 5.0 release candidate not yet adopted.

The production path forward is to split parser/training/registry tooling into separate dependency groups or service images.

## Responsible Disclosure

For a portfolio release, document security issues in GitHub Issues only if they do not expose live credentials or private infrastructure details.
