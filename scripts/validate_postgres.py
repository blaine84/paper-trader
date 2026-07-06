"""Post-migration validation script for Postgres.

Runs a suite of health checks against the live Postgres database and
optionally the Web API and Pi Proxy endpoints.

Usage:
    python scripts/validate_postgres.py
    python scripts/validate_postgres.py --skip-http
    python scripts/validate_postgres.py --proxy-url https://your-proxy.ts.net
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

# Expected tables — imported from migration script for consistency.
from scripts.migrate_to_postgres import TABLE_ORDER


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    status: str  # "PASS" | "FAIL" | "WARNING" | "SKIP"
    detail: str
    expected: str = ""
    actual: str = ""


@dataclass
class ValidationReport:
    """Aggregated results from all validation checks."""

    status: str = "PASS"
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        """Add a check result and update overall status."""
        self.checks.append(result)
        if result.status == "FAIL":
            self.status = "FAIL"


def validate_row_counts(engine) -> CheckResult:
    """Check that all expected tables have > 0 rows (Requirement 5.1)."""
    empty_tables: list[str] = []
    table_counts: dict[str, int] = {}

    with engine.connect() as conn:
        for table in TABLE_ORDER:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            table_counts[table] = count or 0
            if count == 0:
                empty_tables.append(table)

    total_rows = sum(table_counts.values())

    if empty_tables:
        return CheckResult(
            name="row_counts",
            status="FAIL",
            detail=f"{len(empty_tables)} table(s) have 0 rows: {', '.join(empty_tables)}",
            expected="All tables have > 0 rows",
            actual=f"{len(empty_tables)} empty tables out of {len(TABLE_ORDER)} total ({total_rows:,} total rows)",
        )

    return CheckResult(
        name="row_counts",
        status="PASS",
        detail=f"All {len(TABLE_ORDER)} tables have rows ({total_rows:,} total rows)",
        expected="All tables have > 0 rows",
        actual=f"{len(TABLE_ORDER)} tables, {total_rows:,} total rows",
    )


def validate_stale_reservations(engine) -> CheckResult:
    """Check no candidates in RESERVED state (Requirement 5.2).

    After migration with services stopped, there should be zero
    RESERVED candidates. During normal operation a small number
    may exist transiently during an active cycle.
    """
    with engine.connect() as conn:
        stale_count = conn.execute(
            text("SELECT COUNT(*) FROM pm_candidates WHERE state = 'RESERVED'")
        ).scalar() or 0

    if stale_count > 0:
        return CheckResult(
            name="stale_reservations",
            status="WARNING",
            detail=(
                f"{stale_count} candidate(s) in RESERVED state. "
                "Expected 0 when services are stopped post-migration."
            ),
            expected="0",
            actual=str(stale_count),
        )

    return CheckResult(
        name="stale_reservations",
        status="PASS",
        detail="No stale reservations found.",
        expected="0",
        actual="0",
    )


def validate_candidate_uniqueness(engine) -> CheckResult:
    """Check candidate_id uniqueness in pm_candidates (Requirement 5.3)."""
    with engine.connect() as conn:
        duplicate_count = conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT candidate_id FROM pm_candidates
                GROUP BY candidate_id HAVING COUNT(*) > 1
            ) dups
        """)).scalar() or 0

    if duplicate_count > 0:
        return CheckResult(
            name="candidate_id_uniqueness",
            status="FAIL",
            detail=f"{duplicate_count} duplicate candidate_id group(s) found.",
            expected="0 duplicates",
            actual=f"{duplicate_count} duplicate groups",
        )

    return CheckResult(
        name="candidate_id_uniqueness",
        status="PASS",
        detail="All candidate_ids are unique.",
        expected="0 duplicates",
        actual="0 duplicates",
    )


def validate_fk_integrity(engine) -> CheckResult:
    """Check trade_events FK references are valid (Requirement 5.4).

    Verifies:
    - trade_events.trade_id references an existing trades.id
    - trade_events.candidate_lineage_id references an existing pm_candidates.candidate_id
    """
    errors: list[str] = []

    with engine.connect() as conn:
        # Check trade_events → trades (trade_id)
        orphan_trades = conn.execute(text("""
            SELECT COUNT(*) FROM trade_events te
            WHERE te.trade_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM trades t WHERE t.id = te.trade_id)
        """)).scalar() or 0

        if orphan_trades > 0:
            errors.append(
                f"trade_events: {orphan_trades} row(s) reference non-existent trades"
            )

        # Check trade_events → pm_candidates (candidate_lineage_id)
        orphan_candidates = conn.execute(text("""
            SELECT COUNT(*) FROM trade_events te
            WHERE te.candidate_lineage_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM pm_candidates pc
                  WHERE pc.candidate_id = te.candidate_lineage_id
              )
        """)).scalar() or 0

        if orphan_candidates > 0:
            errors.append(
                f"trade_events: {orphan_candidates} row(s) reference non-existent pm_candidates"
            )

    if errors:
        return CheckResult(
            name="fk_integrity",
            status="FAIL",
            detail="; ".join(errors),
            expected="0 orphaned FK references",
            actual="; ".join(errors),
        )

    return CheckResult(
        name="fk_integrity",
        status="PASS",
        detail="All FK references in trade_events are valid.",
        expected="0 orphaned FK references",
        actual="0 orphaned FK references",
    )


def validate_web_health(base_url: str, timeout: int = 10) -> CheckResult:
    """Check Web API /api/data returns HTTP 200 (Requirement 5.5)."""
    import requests

    url = f"{base_url.rstrip('/')}/api/data"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return CheckResult(
                name="web_api_health",
                status="PASS",
                detail=f"GET {url} returned HTTP 200.",
                expected="HTTP 200",
                actual=f"HTTP {resp.status_code}",
            )
        else:
            return CheckResult(
                name="web_api_health",
                status="FAIL",
                detail=f"GET {url} returned HTTP {resp.status_code}.",
                expected="HTTP 200",
                actual=f"HTTP {resp.status_code}",
            )
    except requests.exceptions.Timeout:
        return CheckResult(
            name="web_api_health",
            status="FAIL",
            detail=f"GET {url} timed out after {timeout}s.",
            expected="HTTP 200 within {timeout}s",
            actual="Timeout",
        )
    except requests.exceptions.ConnectionError as e:
        return CheckResult(
            name="web_api_health",
            status="FAIL",
            detail=f"GET {url} connection failed: {e}",
            expected="HTTP 200",
            actual="Connection error",
        )
    except Exception as e:
        return CheckResult(
            name="web_api_health",
            status="FAIL",
            detail=f"GET {url} failed: {e}",
            expected="HTTP 200",
            actual=f"Error: {e}",
        )


def validate_proxy_health(proxy_url: str, timeout: int = 15) -> CheckResult:
    """Check Pi Proxy public URL returns HTTP 200 (Requirement 5.6)."""
    import requests

    try:
        resp = requests.get(proxy_url, timeout=timeout)
        if resp.status_code == 200 and len(resp.text) > 0:
            return CheckResult(
                name="proxy_health",
                status="PASS",
                detail=f"GET {proxy_url} returned HTTP 200 with non-empty body.",
                expected="HTTP 200 with non-empty body",
                actual=f"HTTP {resp.status_code}, {len(resp.text)} bytes",
            )
        elif resp.status_code == 200:
            return CheckResult(
                name="proxy_health",
                status="FAIL",
                detail=f"GET {proxy_url} returned HTTP 200 but empty body.",
                expected="HTTP 200 with non-empty body",
                actual="HTTP 200, empty body",
            )
        else:
            return CheckResult(
                name="proxy_health",
                status="FAIL",
                detail=f"GET {proxy_url} returned HTTP {resp.status_code}.",
                expected="HTTP 200",
                actual=f"HTTP {resp.status_code}",
            )
    except requests.exceptions.Timeout:
        return CheckResult(
            name="proxy_health",
            status="FAIL",
            detail=f"GET {proxy_url} timed out after {timeout}s.",
            expected="HTTP 200 within {timeout}s",
            actual="Timeout",
        )
    except requests.exceptions.ConnectionError as e:
        return CheckResult(
            name="proxy_health",
            status="FAIL",
            detail=f"GET {proxy_url} connection failed: {e}",
            expected="HTTP 200",
            actual="Connection error",
        )
    except Exception as e:
        return CheckResult(
            name="proxy_health",
            status="FAIL",
            detail=f"GET {proxy_url} failed: {e}",
            expected="HTTP 200",
            actual=f"Error: {e}",
        )


def run_validation(
    database_url: str,
    skip_http: bool = False,
    web_url: str = "http://127.0.0.1:5000",
    proxy_url: str | None = None,
    web_timeout: int = 10,
    proxy_timeout: int = 15,
) -> ValidationReport:
    """Run all validation checks and return a structured report.

    Args:
        database_url: Postgres connection URL.
        skip_http: If True, skip Web API and proxy health checks.
        web_url: Base URL for the Web API health check.
        proxy_url: Public URL for the Pi Proxy health check.
            If None, reads PI_PROXY_URL from environment.
        web_timeout: Timeout in seconds for the Web API health check.
        proxy_timeout: Timeout in seconds for the proxy health check.

    Returns:
        ValidationReport with overall status and individual check results.
    """
    report = ValidationReport()
    engine = create_engine(database_url, pool_pre_ping=True)

    try:
        # Database checks
        log.info("Running row count validation...")
        report.add(validate_row_counts(engine))

        log.info("Running stale reservation check...")
        report.add(validate_stale_reservations(engine))

        log.info("Running candidate_id uniqueness check...")
        report.add(validate_candidate_uniqueness(engine))

        log.info("Running FK integrity check...")
        report.add(validate_fk_integrity(engine))

        # HTTP checks (optional)
        if skip_http:
            report.add(CheckResult(
                name="web_api_health",
                status="SKIP",
                detail="HTTP checks skipped via --skip-http flag.",
            ))
            report.add(CheckResult(
                name="proxy_health",
                status="SKIP",
                detail="HTTP checks skipped via --skip-http flag.",
            ))
        else:
            log.info("Running Web API health check...")
            report.add(validate_web_health(web_url, timeout=web_timeout))

            resolved_proxy_url = proxy_url or os.environ.get("PI_PROXY_URL", "").strip()
            if resolved_proxy_url:
                log.info("Running Pi Proxy health check...")
                report.add(validate_proxy_health(resolved_proxy_url, timeout=proxy_timeout))
            else:
                report.add(CheckResult(
                    name="proxy_health",
                    status="SKIP",
                    detail="No proxy URL provided (set PI_PROXY_URL or use --proxy-url).",
                ))
    finally:
        engine.dispose()

    return report


def _print_report(report: ValidationReport) -> None:
    """Print a human-readable validation report to stdout (Requirement 5.8)."""
    print(f"\n{'='*60}")
    print(f"  Post-Migration Validation Report: {report.status}")
    print(f"{'='*60}")

    for check in report.checks:
        if check.status == "PASS":
            symbol = "\u2713"
        elif check.status == "WARNING":
            symbol = "\u26a0"
        elif check.status == "SKIP":
            symbol = "\u2014"
        else:
            symbol = "\u2717"

        print(f"\n  {symbol} [{check.status}] {check.name}")
        print(f"    {check.detail}")
        if check.expected and check.status == "FAIL":
            print(f"    Expected: {check.expected}")
            print(f"    Actual:   {check.actual}")

    print(f"\n{'='*60}")
    print(f"  Overall: {report.status}")
    print(f"{'='*60}\n")


def main() -> int:
    """CLI entry point. Returns 0 for PASS, 1 for FAIL."""
    parser = argparse.ArgumentParser(
        description="Post-migration validation for Postgres"
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres connection URL (default: reads DATABASE_URL from environment)",
    )
    parser.add_argument(
        "--skip-http",
        action="store_true",
        help="Skip Web API and proxy HTTP health checks",
    )
    parser.add_argument(
        "--web-url",
        default="http://127.0.0.1:5000",
        help="Base URL for Web API health check (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="Pi Proxy public URL for health check (default: reads PI_PROXY_URL from env)",
    )
    parser.add_argument(
        "--web-timeout",
        type=int,
        default=10,
        help="Timeout in seconds for Web API health check (default: 10)",
    )
    parser.add_argument(
        "--proxy-timeout",
        type=int,
        default=15,
        help="Timeout in seconds for proxy health check (default: 15)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve database URL.
    database_url = args.database_url or os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("ERROR: No database URL. Set DATABASE_URL or pass --database-url.")
        return 1

    # Run validation.
    report = run_validation(
        database_url=database_url,
        skip_http=args.skip_http,
        web_url=args.web_url,
        proxy_url=args.proxy_url,
        web_timeout=args.web_timeout,
        proxy_timeout=args.proxy_timeout,
    )

    # Print report.
    _print_report(report)

    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
