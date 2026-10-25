"""No image may write its own gitleaks egress config (alpha-engine-config-I8058).

krepis.session_dlp resolves its ruleset directory as $KREPIS_GITLEAKS_DIR,
then /opt/{llm,groom,drain}-llm-routing, then the ruleset krepis ships as
package data. An operator directory OUTRANKS the packaged copy. So a Dockerfile
that writes even a minimal ``gitleaks-egress.toml`` replaces the fleet's real
ruleset (stock plus the nousergon-specific custom rules) with whatever it
wrote. A narrower ruleset fails QUIET: the scan runs, returns clean, and the
call proceeds. That is how a two-line ``[extend] useDefault = true`` stub
stood in for the real config in three Lambda images.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A write of the egress config, or a stock-only `[extend]` stanza, in any
# non-comment Dockerfile line.
_STUB = re.compile(r"gitleaks-egress\.toml|useDefault")


def _dockerfiles() -> list[Path]:
    return sorted({*ROOT.glob("Dockerfile*"), *ROOT.glob("*/Dockerfile*")})


def _code_lines(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
    )


def test_there_is_a_dockerfile_to_check() -> None:
    assert _dockerfiles(), "no Dockerfile found; the guard below would pass vacuously"


def test_no_dockerfile_writes_its_own_gitleaks_egress_config() -> None:
    offenders = [
        str(p.relative_to(ROOT)) for p in _dockerfiles() if _STUB.search(_code_lines(p))
    ]
    assert not offenders, (
        f"{offenders} write a gitleaks egress config into the image. Any file there "
        "outranks the ruleset krepis ships as package data, so the image scans with "
        "whatever was written instead of the fleet's real rules. Delete the write "
        "and let krepis.session_dlp resolve its packaged dlp_config/."
    )


def test_the_guard_catches_the_stub_it_exists_for(tmp_path: Path) -> None:
    sample = tmp_path / "Dockerfile"
    sample.write_text(
        "# a comment naming gitleaks-egress.toml is fine\n"
        "RUN printf '[extend]\\nuseDefault = true\\n' > /opt/llm-routing/gitleaks-egress.toml\n"
    )
    assert _STUB.search(_code_lines(sample))
    sample.write_text("# only a comment: gitleaks-egress.toml\nRUN gitleaks version\n")
    assert not _STUB.search(_code_lines(sample))


def test_every_image_that_ships_gitleaks_asserts_the_packaged_ruleset() -> None:
    """Removing the stub is half the fix. The build must also PROVE that the
    ruleset the installed krepis resolves is the packaged fleet ruleset, because
    a narrow config and the real one are indistinguishable at the call site."""
    missing = [
        str(p.relative_to(ROOT))
        for p in _dockerfiles()
        if "gitleaks version" in _code_lines(p)
        and "ruleset_matches_packaged: True" not in _code_lines(p)
    ]
    assert not missing, (
        f"{missing} install gitleaks but never assert at build time that krepis "
        "resolved its packaged ruleset; add the `python -m krepis.session_dlp "
        "preflight` check grepping `ruleset_matches_packaged: True` after the pip install."
    )
