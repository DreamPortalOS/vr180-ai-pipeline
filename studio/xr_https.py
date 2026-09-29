"""Self-signed TLS certificate for the Quest dome viewer's Wi-Fi access.

WebXR requires a secure context.  The USB tunnel (``adb reverse tcp:8787
tcp:8787``) gives one for free — ``localhost`` is a secure context — so the
viewer works there with plain http.  Wi-Fi access instead needs
``https://<computer-IP>:8787/xr`` and therefore a cert.  Git for Windows
ships openssl, so this module generates a self-signed cert/key pair into the
studio work_root with **no new Python dependency** (CLAUDE.md boundary lock:
no new deps for a card).  openssl is invoked as a subprocess **list** — never
``shell=True``.

When openssl is missing or fails, :class:`CertError` carries a message that
names the USB alternative, so the Studio log tells the operator exactly what
to do instead of an opaque traceback.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

#: Subject + SAN days.  365d keeps the Quest's "continue to site" exception
#: valid across restarts — regenerate only by deleting the files.
CERT_DAYS = 365


class CertError(RuntimeError):
    """Raised when openssl is unavailable or cert generation failed."""


def _openssl_bin() -> str | None:
    """Locate openssl on PATH (None if absent)."""
    return shutil.which("openssl")


def _hostname() -> str:
    """Best-effort local hostname for the cert's subjectAltName."""
    try:
        return socket.gethostname() or "localhost"
    except OSError:
        return "localhost"


def _build_cmd(openssl: str, key: Path, cert: Path, *, with_addext: bool) -> list[str]:
    """The openssl argv that produces a self-signed rsa:2048 cert+key pair.

    Kept as a helper so a test can assert the exact list form (no shell, no
    string concatenation) and so the ``-addext`` fallback can drop the two
    related args together.
    """
    cmd = [
        openssl,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-days",
        str(CERT_DAYS),
        "-nodes",
        "-keyout",
        str(key),
        "-out",
        str(cert),
        "-subj",
        "/CN=vr180-studio-dome-xr",
    ]
    if with_addext:
        cmd += [
            "-addext",
            f"subjectAltName=DNS:localhost,DNS:{_hostname()},IP:127.0.0.1",
        ]
    return cmd


def _generate(openssl: str, key: Path, cert: Path) -> tuple[Path, Path]:
    """Run openssl until a cert+key pair lands; retry without ``-addext``.

    OpenSSL 3.x (Git for Windows) supports ``-addext``; older/LibreSSL builds
    reject it.  A single retry without it still yields a working cert (the
    Quest shows a "not trusted" continue-prompt either way for a self-signed
    cert; the SAN only silences desktop Chrome's red page).
    """
    last_err = ""
    for with_addext in (True, False):
        cmd = _build_cmd(openssl, key, cert, with_addext=with_addext)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            last_err = f"openssl invocation failed: {exc}"
            continue
        if proc.returncode == 0 and cert.is_file() and key.is_file():
            return cert, key
        last_err = (proc.stderr or proc.stdout or "").strip()[-400:]
    raise CertError(
        "openssl self-signed cert generation failed. Use the USB access method "
        "instead: `adb reverse tcp:8787 tcp:8787` then open "
        "http://localhost:8787/xr on the Quest (localhost is a secure context, "
        "so WebXR works without https). Last openssl error:\n" + last_err
    )


def ensure_self_signed_cert(directory: str | Path) -> tuple[Path, Path]:
    """Return ``(cert_path, key_path)``, generating the pair if absent.

    Reuses an existing pair in ``directory`` so repeated ``--https`` starts do
    not regenerate (and so do not invalidate the Quest's stored exception).
    Raises :class:`CertError` with a USB-method hint when openssl is missing.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cert = root / "dome_xr_cert.pem"
    key = root / "dome_xr_key.pem"
    if cert.is_file() and key.is_file():
        return cert, key
    openssl = _openssl_bin()
    if openssl is None:
        raise CertError(
            "openssl not found on PATH — cannot generate a self-signed TLS "
            "certificate for --https. Use the USB access method instead: "
            "`adb reverse tcp:8787 tcp:8787` then open http://localhost:8787/xr "
            "on the Quest (localhost is a secure context, so WebXR works "
            "without https)."
        )
    return _generate(openssl, key, cert)
