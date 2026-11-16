#!/usr/bin/env python3
"""Serve the FISH2 review dashboard over a private Unix socket."""

import argparse
import json
import os
import socket
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from build_compartment_review import LABEL, LABELS, write_json

LOCK = threading.Lock()
FILES = ("compartment_edges.parquet", "compartment_bridges.parquet")
HOSTS = {"127.0.0.1:18765", "localhost:18765"}


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


class ReviewServer(ThreadingHTTPServer):
    address_family = socket.AF_UNIX

    def server_bind(self):
        self.socket.bind(self.server_address)
        os.chmod(self.server_address, 0o600)
        self.server_name = "localhost"
        self.server_port = 0


class Handler(BaseHTTPRequestHandler):
    def address_string(self):
        return "local"

    def send_bytes(self, data, kind, status=200, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def reply(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def folders(self, body):
        root = self.server.root
        return (root / "automatic" / body, root / "corrected" / body, root / "drafts" / f"{body}.json")

    def state(self, body):
        auto, corrected, draft = self.folders(body)
        saved = load(corrected / "review.json")
        pending = load(draft)
        if pending and saved and pending["revision"] <= saved["revision"]:
            pending = None
        return auto, corrected, draft, saved, pending

    def panel(self):
        rows = load(self.server.root / "panel.json", [])
        for row in rows:
            body = str(row["body_id"])
            auto, _, _, saved, pending = self.state(body)
            batch = load(self.server.root / "batch_status" / f"{body}.json", {})
            detail = load(auto / "candidate.json", {})
            row.update(
                state=("corrected_with_draft" if saved and pending else "draft" if pending
                       else "corrected" if saved else "ready" if auto.exists() else batch.get("state", "pending")),
                ambiguous=(saved or {}).get("ambiguous", detail.get("edge_counts", {}).get("ambiguous", 0)
                                            + detail.get("bridge_counts", {}).get("ambiguous", 0)),
                main_pair=detail.get("step01_arbors", {}).get("pair_confidence"),
                error=batch.get("error"),
            )
        return rows

    def body_data(self, body):
        auto, _, _, saved, pending = self.state(body)
        detail = load(auto / "candidate.json")
        if detail is None:
            raise FileNotFoundError("This candidate is not ready")
        nodes = pd.read_parquet(auto / "skeleton_nodes.parquet")
        points = (nodes[["x_em", "y_em", "z_em"]].to_numpy() * [.008, .008, .030]).round(4)
        edges = pd.read_parquet(auto / FILES[0])
        bridges = pd.read_parquet(auto / FILES[1])
        n = len(edges)
        links = edges[["u", "v"]].to_numpy(int).tolist()
        links += bridges[["u_node_index", "v_node_index"]].to_numpy(int).tolist()
        synapses = pd.read_parquet(auto / "synapse_attachments.parquet")
        attached = synapses[synapses.attachment_status.eq("attached")]
        syn = {}
        for role in ("pre", "post"):
            xyz = attached.loc[attached.role.eq(role), ["x_em_nm", "y_em_nm", "z_em_nm"]].drop_duplicates()
            syn[role] = (xyz.to_numpy() / 1000).round(4).tolist()
        current = pending or saved or {}
        return dict(detail=detail, xyz=points.tolist(), edges=links, labels=edges[LABEL].tolist() + bridges[LABEL].tolist(),
                    original_edges=n, synapses=syn, changes=current.get("changes", []),
                    revision=current.get("revision", 0), corrected=bool(saved), draft=bool(pending))

    def save(self, body, data):
        with LOCK:
            auto, corrected, draft, saved, pending = self.state(body)
            revision = (pending or saved or {}).get("revision", 0)
            if type(data.get("revision")) is not int or data["revision"] != revision:
                self.reply({"error": "This neuron changed in another tab. Reload it before saving."}, 409)
                return
            action = data.get("action")
            if action not in {"draft", "corrected"}:
                raise ValueError("Invalid review action")
            frames = [pd.read_parquet(auto / name) for name in FILES]
            original = [label for frame in frames for label in frame[LABEL].tolist()]
            changes = {}
            if not isinstance(data.get("changes"), list):
                raise ValueError("Expected a list of edge labels")
            for edge, label in data["changes"]:
                if type(edge) is not int or not 0 <= edge < len(original) or label not in LABELS or edge in changes:
                    raise ValueError("Invalid edge label")
                changes[edge] = label
            changes = {edge: label for edge, label in changes.items() if original[edge] != label}
            ambiguous = sum(changes.get(i, lab) == "ambiguous" for i, lab in enumerate(original))
            record = dict(body_id=int(body), revision=revision + 1, state=action,
                          updated_at=datetime.now(timezone.utc).isoformat(),
                          ambiguous=ambiguous, changes=sorted(changes.items()))
            if action == "draft":
                write_json(draft, record)
            else:
                target = corrected / "revisions" / str(record["revision"])
                target.mkdir(parents=True, exist_ok=True)
                offset = 0
                for name, frame in zip(FILES, frames):
                    frame[LABEL] = [changes.get(offset + i, lab) for i, lab in enumerate(frame[LABEL])]
                    frame["manual_override"] = [offset + i in changes for i in range(len(frame))]
                    frame.to_parquet(target / name, index=False)
                    offset += len(frame)
                record["files"] = {name: str(target / name) for name in FILES}
                write_json(corrected / "review.json", record)
                draft.unlink(missing_ok=True)
            self.reply(record)

    def do_GET(self):
        if self.headers.get("Host") not in HOSTS:
            self.reply({"error": "Invalid host"}, 403)
            return
        route = urlsplit(self.path).path
        try:
            if route == "/":
                self.send_bytes(Path(__file__).with_name("compartment_review.html").read_bytes(),
                                "text/html; charset=utf-8")
            elif route == "/api/panel":
                self.reply(self.panel())
            elif route.startswith("/api/body/") and route[10:].isdigit():
                self.reply(self.body_data(route[10:]))
            else:
                self.reply({"error": "Not found"}, 404)
        except FileNotFoundError as exc:
            self.reply({"error": str(exc)}, 404)

    def do_POST(self):
        route = urlsplit(self.path).path
        origin = self.headers.get("Origin")
        expected = "http://" + self.headers.get("Host", "")
        if (self.headers.get("Host") not in HOSTS or origin != expected
                or self.headers.get_content_type() != "application/json"):
            self.reply({"error": "Invalid request origin or content type"}, 403)
            return
        if not (route.startswith("/api/body/") and route[10:].isdigit()):
            self.reply({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 5_000_000:
                raise ValueError("Invalid payload size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Invalid payload")
            self.save(route[10:], payload)
        except (ValueError, TypeError, KeyError) as exc:
            self.reply({"error": str(exc)}, 400)
        except FileNotFoundError as exc:
            self.reply({"error": str(exc)}, 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("socket", type=Path)
    args = parser.parse_args()
    with ReviewServer(str(args.socket), Handler) as server:
        server.root = args.output.resolve()
        try:
            server.serve_forever()
        finally:
            args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
