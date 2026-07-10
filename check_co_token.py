#!/usr/bin/env python3
"""One-off diagnostic: exercise the Code Ocean API token end to end.

TEMPORARY -- delete after confirming the "Code Ocean API Credentials" secret works.

Uses the exact same auth scheme as code/metadata/write_metadata.py (HTTP Basic with
API_KEY as the username, empty password), hits the real CO REST API, and prints the
capsule slug/URL and the run's release version. It distinguishes a 401 (token present
but rejected -- bad/expired/wrong value) from other failures, and reports exactly which
env vars are present, so you can tell "secret not attached" apart from "token invalid".

Run it inside the capsule (a Reproducible Run or a cloud-workstation terminal) where the
secret is attached:

    python check_co_token.py

CO_CAPSULE_ID / CO_COMPUTATION_ID are injected automatically during a Reproducible Run.
In an interactive terminal CO_COMPUTATION_ID may be unset; the capsule lookup (which is
what actually exercises the token) still runs, and you can pass ids explicitly:

    python check_co_token.py --capsule-id <id> --computation-id <id>
"""

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

CO_API_BASE = "https://codeocean.allenneuraldynamics.org/api/v1"
CO_WEB_BASE = "https://codeocean.allenneuraldynamics.org/capsule"


def _mask(value: str) -> str:
    """Show only enough of a secret to confirm it is set, without leaking it."""
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return f"<{len(value)} chars>"
    return f"{value[:4]}...{value[-4:]} ({len(value)} chars)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capsule-id", default=os.environ.get("CO_CAPSULE_ID"))
    ap.add_argument("--computation-id", default=os.environ.get("CO_COMPUTATION_ID"))
    args = ap.parse_args()

    api_key = os.environ.get("API_KEY")
    capsule_id = args.capsule_id
    computation_id = args.computation_id

    print("=== Code Ocean API token check ===")
    print(f"  API_KEY:            {_mask(api_key)}")
    print(f"  CO_CAPSULE_ID:      {capsule_id or '<unset>'}")
    print(f"  CO_COMPUTATION_ID:  {computation_id or '<unset>'}")
    print(f"  API base:           {CO_API_BASE}")
    print()

    if not api_key:
        print("FAIL: API_KEY is not set. The 'Code Ocean API Credentials' secret is not "
              "attached to this capsule (Capsule Settings -> Credentials).")
        return 2
    if not capsule_id:
        print("FAIL: no capsule id. Run inside a capsule, or pass --capsule-id.")
        return 2

    auth = base64.b64encode(f"{api_key}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}

    def _get(path: str) -> dict:
        req = urllib.request.Request(f"{CO_API_BASE}{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    # 1. Capsule lookup -- this is the call that actually exercises the token.
    try:
        print(f"GET /capsules/{capsule_id} ...")
        capsule = _get(f"/capsules/{capsule_id}")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("FAIL (401 Unauthorized): the token was sent but Code Ocean rejected it. "
                  "The secret is attached, but the API_KEY value is invalid/expired/revoked "
                  "or belongs to an account without API access. Regenerate the token "
                  "(Account -> User API credentials) and update the secret.")
        else:
            print(f"FAIL (HTTP {e.code} {e.reason}): {e.read().decode(errors='replace')[:500]}")
        return 1
    except urllib.error.URLError as e:
        print(f"FAIL (network/URL error, not an auth problem): {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"FAIL (could not parse response as JSON): {e}")
        return 1

    slug = capsule.get("slug")
    capsule_url = f"{CO_WEB_BASE}/{slug}/tree" if slug else "<no slug in response>"
    print("  OK -- token accepted.")
    print(f"     name:  {capsule.get('name')}")
    print(f"     slug:  {slug}")
    print(f"     url:   {capsule_url}")
    print()

    # 2. Computation lookup -- yields the release version (needs a computation id).
    if not computation_id:
        print("NOTE: CO_COMPUTATION_ID unset, so the release version can't be looked up "
              "(it is auto-set during a Reproducible Run). The token itself is confirmed "
              "working by the capsule lookup above.")
        return 0

    try:
        print(f"GET /computations/{computation_id} ...")
        computation = _get(f"/computations/{computation_id}")
    except urllib.error.HTTPError as e:
        print(f"FAIL on computation lookup (HTTP {e.code} {e.reason}). The token works "
              f"(capsule lookup succeeded); this id may be wrong: {computation_id}")
        return 1
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"FAIL on computation lookup: {e}")
        return 1

    version = (f"v{computation['version']}.0" if "version" in computation
               else "from non-release editable capsule")
    print("  OK.")
    print(f"     version: {version}")
    print()
    print("SUCCESS: token works and provenance (capsule URL + version) resolves. "
          "processing.json will be written on the next Reproducible Run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
