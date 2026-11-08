"""Terminal progress reporting, ASCII summary table, and JSON export for batch jobs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, TextIO, Union

from subsync.batch.runner import BatchJobResult, BatchSummary


def format_offset(offset_seconds: float) -> str:
    """Format temporal offset `b` in seconds with explicit sign (e.g. ``+1.250s``)."""
    return f"{offset_seconds:+.3f}s"


def format_speed_factor(speed_factor: float) -> str:
    """Format linear drift / speed factor `a` with 6 decimal places (e.g. ``1.000000``)."""
    return f"{speed_factor:.6f}"


def format_correlation(correlation: float) -> str:
    """Format correlation score (e.g. ``0.950``)."""
    return f"{correlation:.3f}"


def _format_episode_label(result: BatchJobResult, max_width: int = 40) -> str:
    """Create a concise human-readable episode/file label for a job row."""
    ep_prefix = ""
    if result.season is not None and result.episode is not None:
        ep_prefix = f"S{result.season:02d}E{result.episode:02d} "
    elif result.episode is not None:
        ep_prefix = f"E{result.episode:02d} "

    name = result.video_path.name
    if ep_prefix and ep_prefix.strip().lower() not in name.lower():
        label = f"{ep_prefix}{name}"
    else:
        label = name

    if len(label) > max_width and max_width > 6:
        return label[: max_width - 3] + "..."
    return label


class BatchProgressReporter:
    """Callable progress reporter for unattended CLI batch execution."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        verbose: int = 1,
    ) -> None:
        self.stream: TextIO = stream if stream is not None else sys.stdout
        self.verbose = verbose

    def __call__(
        self,
        result: BatchJobResult,
        completed: int,
        total: int,
    ) -> None:
        if self.verbose <= 0:
            return

        ep_label = _format_episode_label(result, max_width=50)
        status = result.status

        if status == "OK":
            offset_str = format_offset(result.offset_seconds)
            factor_str = format_speed_factor(result.speed_factor)
            score_str = format_correlation(result.correlation)
            out_name = result.out_path.name if result.out_path else "-"
            line = (
                f"[{completed}/{total}] [{status}] {ep_label} -> {out_name} "
                f"(b={offset_str}, a={factor_str}, score={score_str}, {result.elapsed_seconds:.2f}s)\n"
            )
        elif status == "FAILED":
            err = result.error or "Unknown error"
            line = (
                f"[{completed}/{total}] [{status}] {ep_label} -> ERROR: {err} "
                f"({result.elapsed_seconds:.2f}s)\n"
            )
        elif status == "SKIPPED":
            out_name = result.out_path.name if result.out_path else "-"
            line = f"[{completed}/{total}] [{status}] {ep_label} -> {out_name} (already exists)\n"
        elif status == "DRY_RUN":
            out_name = str(result.out_path) if result.out_path else "-"
            line = f"[{completed}/{total}] [{status}] {ep_label} -> {out_name}\n"
        else:
            line = f"[{completed}/{total}] [{status}] {ep_label}\n"

        self.stream.write(line)
        self.stream.flush()


def create_progress_callback(
    stream: Optional[TextIO] = None,
    verbose: int = 1,
) -> Callable[[BatchJobResult, int, int], None]:
    """Create a terminal progress callback function for `run_batch`."""
    return BatchProgressReporter(stream=stream, verbose=verbose)


def format_summary_table(summary: BatchSummary) -> str:
    """Format a structured ASCII summary table for `summary`.

    Displays each episode/file, status, recap handling, multi-act offset evolution,
    IA correlation score, and 1st synchronized subtitle line.
    """
    headers = [
        "Épisode / Fichier",
        "Statut",
        "Récap « Précédemment »",
        "Évolution Décalage (Début -> Milieu -> Fin)",
        "Corr. IA",
        "1re réplique synchronisée",
    ]

    rows: List[List[str]] = []
    for res in summary.results:
        file_col = _format_episode_label(res, max_width=36)
        status_col = res.status
        det = res.details or {}

        if res.status == "OK":
            recap_col = str(det.get("recap_status") or "Non")
            offset_col = str(det.get("offset_evolution") or f"{format_offset(res.offset_seconds)} (constant)")
            score_col = f"{res.correlation * 100.0:.1f}%"
            ft = det.get("first_sub_time_s")
            fx = det.get("first_sub_text")
            if ft is not None and fx:
                out_col = f"{float(ft):.2f}s : « {str(fx)[:48]} »"
            else:
                out_col = res.out_path.name if res.out_path else "-"
        elif res.status == "FAILED":
            recap_col = "-"
            offset_col = "-"
            score_col = "-"
            out_col = f"ERROR: {res.error}" if res.error else "ERROR"
        else:
            recap_col = "-"
            offset_col = "-"
            score_col = "-"
            out_col = res.out_path.name if res.out_path else "-"

        rows.append(
            [
                file_col,
                status_col,
                recap_col,
                offset_col,
                score_col,
                out_col,
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > col_widths[i]:
                col_widths[i] = len(cell)

    def _sep(char: str = "-") -> str:
        return "+" + "+".join(char * (w + 2) for w in col_widths) + "+"

    def _fmt_row(cells: List[str]) -> str:
        parts: List[str] = []
        for i, cell in enumerate(cells):
            if i == 4:
                parts.append(f" {cell:>{col_widths[i]}} ")
            else:
                parts.append(f" {cell:<{col_widths[i]}} ")
        return "|" + "|".join(parts) + "|"

    lines: List[str] = [
        "",
        "=== RÉCAPITULATIF SYNCHRONISATION BATCH HYBRIDE (MATH + IA) ===",
        _sep("-"),
        _fmt_row(headers),
        _sep("="),
    ]
    if rows:
        for r in rows:
            lines.append(_fmt_row(r))
    else:
        empty_msg = " No matched media/subtitle pairs to process "
        inner_width = sum(w + 3 for w in col_widths) - 1
        lines.append("|" + f"{empty_msg:<{inner_width}}"[:inner_width] + "|")
    lines.append(_sep("-"))

    footer = (
        f"Summary: total={summary.total} | OK={summary.succeeded} | "
        f"FAILED={summary.failed} | SKIPPED={summary.skipped} | "
        f"DRY_RUN={summary.dry_run_count} | elapsed={summary.total_elapsed_seconds:.2f}s"
    )
    lines.append(footer)

    if summary.unmatched_videos:
        lines.append(f"Unmatched videos ({len(summary.unmatched_videos)}): " + ", ".join(p.name for p in summary.unmatched_videos))
    if summary.unmatched_subtitles:
        lines.append(f"Unmatched subtitles ({len(summary.unmatched_subtitles)}): " + ", ".join(p.name for p in summary.unmatched_subtitles))

    return "\n".join(lines)


def print_summary_table(
    summary: BatchSummary,
    stream: Optional[TextIO] = None,
) -> str:
    """Print the ASCII summary table to `stream` (default stdout) and return it."""
    target = stream if stream is not None else sys.stdout
    table_str = format_summary_table(summary)
    target.write(table_str + "\n")
    target.flush()
    return table_str


# Convenient aliases
format_summary = format_summary_table
print_summary = print_summary_table


def export_summary_json(
    summary: BatchSummary,
    json_path: Union[str, Path],
) -> Path:
    """Write a machine-readable JSON report of `summary` to `json_path`."""
    out_file = Path(json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return out_file
