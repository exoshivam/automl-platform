"""Outgoing mail. Dev/test: written to MAIL_SINK_DIR as .eml files (listed at /_dev/mail).
Prod: an SES/Postmark sender is the M5 task; until then prod mail is only logged."""

from email.message import EmailMessage

import structlog

from app.db import now, uuid7
from app.settings import get_settings

log = structlog.get_logger()


def send_mail(to: str, subject: str, body: str) -> None:
    s = get_settings()
    msg = EmailMessage()
    msg["From"] = "AutoML <no-reply@automl.local>"
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg.set_content(body)
    if s.is_prod:
        log.warning("mail_not_sent_no_provider", to=to, subject=subject)
        return
    s.mail_sink_dir.mkdir(parents=True, exist_ok=True)
    path = s.mail_sink_dir / f"{now().strftime('%Y%m%dT%H%M%S%f')}-{uuid7().hex[:8]}.eml"
    path.write_bytes(bytes(msg))
    log.info("mail_written", to=to, subject=subject, path=str(path))


def list_sink(limit: int = 50) -> list[dict]:
    s = get_settings()
    if not s.mail_sink_dir.exists():
        return []
    from email import message_from_bytes, policy

    out = []
    for p in sorted(s.mail_sink_dir.glob("*.eml"), reverse=True)[:limit]:
        msg = message_from_bytes(p.read_bytes(), policy=policy.default)
        out.append({"file": p.name, "to": msg["To"], "subject": msg["Subject"], "date": msg["Date"],
                    "body": msg.get_content()})
    return out
