"""Download raw Hemibrain SWCs using a credential already configured on this host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import Request, urlopen

ENDPOINT = "https://neuprint.janelia.org/api/skeletons/skeleton/hemibrain:v1.2.1/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".config/neuprint/token")
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if token.startswith("{"):
        payload = json.loads(token)
        token = next(payload[k] for k in ("token", "access_token", "auth_token") if k in payload)
    token = token.strip('"\x27')
    if token.lower().startswith("bearer "):
        token = token[7:]
    if not token or any(c.isspace() for c in token):
        raise ValueError("Invalid token file format")
    rows = json.loads(args.manifest.read_text())["records"]
    bodies = sorted({str(int(r["body_id"])) for r in rows})
    args.output.mkdir(parents=True, exist_ok=True)

    def fetch(body):
        path = args.output / f"{body}.swc"
        if path.exists():
            return body, None
        request = Request(f"{ENDPOINT}{body}?format=swc", headers={"Authorization": f"Bearer {token}"})
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read()
            nodes = [line for line in data.splitlines() if line.strip() and not line.lstrip().startswith(b"#")]
            if not nodes or any(len(line.split()) != 7 for line in nodes):
                raise ValueError("Response is not an SWC skeleton")
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
            return body, None
        except Exception as exc:
            return body, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = {body: error for body, error in pool.map(fetch, bodies) if error is not None}
    summary = {"dataset": "hemibrain:v1.2.1", "kind": "raw_neuprint_swc",
               "requested": len(bodies), "downloaded": len(bodies) - len(errors), "errors": errors}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['downloaded']}/{len(bodies)} skeletons downloaded", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
