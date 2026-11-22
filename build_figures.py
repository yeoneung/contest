#!/usr/bin/env python3
"""Build the article's three figures from their saved numerical inputs.

Run from any directory. Recompute grids separately with the experiment scripts.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FIGURES = {
    1: ("fig_timeline.pdf", "fig_timeline.tex"),
    2: ("fig_stanford_decision.pdf", "plot_stanford_awards.py"),
    3: ("fig_budget_regions.pdf", "budget_region_map.py"),
}


def tex_engine(configured):
    candidate = configured or os.environ.get("TECTONIC")
    if candidate:
        found = shutil.which(candidate)
        if found:
            return found
        raise SystemExit("TeX executable not found: " + candidate)
    for name in ("tectonic", "pdflatex"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("Install Tectonic or pdfLaTeX, or set TECTONIC to its executable.")


def compile_tex(source, engine, env):
    with tempfile.TemporaryDirectory(prefix="contest_figure_tex_") as folder:
        if "tectonic" in Path(engine).name.lower():
            command = [engine, "--outdir", folder, source]
            if os.environ.get("TECTONIC_OFFLINE") == "1":
                command.insert(1, "-C")
            passes = 1
        elif "pdflatex" in Path(engine).name.lower():
            command = [engine, "-interaction=nonstopmode", "-halt-on-error",
                       "-output-directory=" + folder, source]
            passes = 2
        else:
            raise SystemExit("Supported TeX engines are Tectonic and pdfLaTeX.")
        for _ in range(passes):
            result = subprocess.run(command, cwd=ROOT / "figures", env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True)
            if result.returncode:
                raise SystemExit(result.stdout)
        pdf = Path(folder) / Path(source).with_suffix(".pdf").name
        shutil.copy2(pdf, ROOT / "figures" / pdf.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=int, nargs="+", choices=sorted(FIGURES),
                        default=sorted(FIGURES), metavar="FIGURE_NUMBER",
                        help="Select figures 1–3; default is all three.")
    parser.add_argument("--tex-engine", help="Tectonic or pdfLaTeX executable.")
    args = parser.parse_args()
    engine = tex_engine(args.tex_engine) if 1 in args.only else None
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for number in dict.fromkeys(args.only):
        output, source = FIGURES[number]
        print(f"Building Figure {number}: {output}", flush=True)
        if source.endswith(".tex"):
            compile_tex(source, engine, env)
        else:
            subprocess.run([sys.executable, str(ROOT / "experiments" / source),
                            "--reuse-grid"], cwd=ROOT, env=env, check=True)
        print("Saved " + str(ROOT / "figures" / output), flush=True)


if __name__ == "__main__":
    main()
