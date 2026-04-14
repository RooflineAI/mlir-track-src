# Copyright 2026 RooflineAI GmbH
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import typer

from mlir_track_src.track_src import track_src

app = typer.Typer(
    help="Tracking operations using source locations in MLIR",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

def main():
    typer.run(track_src)

if __name__ == "__main__":
    main()
