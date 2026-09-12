{
  description = "IMWUT 2026 FishSense Lite — uv-managed dev shell (FHS env with C + Rust toolchain for fishsense-core)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Libraries the prebuilt pip/uv wheels link against, plus the C + Rust
        # toolchain needed to build `fishsense-core` (a maturin/pyo3 Rust sdist
        # pulled in transitively via fishsense-meta). Without `cc` the maturin
        # build fails with `linker 'cc' not found`.
        fhsLibs = pkgs: (with pkgs; [
          # uv + python toolchain (repo pins 3.13 via .python-version)
          uv
          python313

          # C/C++ build toolchain — BLAS/LAPACK source builds (numpy/scipy) and
          # the linker maturin needs to link the Rust extension.
          stdenv.cc
          stdenv.cc.cc.lib
          gcc
          gnumake
          cmake
          ninja
          pkg-config
          gfortran

          # Rust toolchain for fishsense-core (pyo3). Provided explicitly so the
          # build is reproducible instead of relying on maturin's network
          # auto-download of a temporary rustup toolchain.
          cargo
          rustc

          # Core runtime libs (matplotlib / scikit-image / numba wheels)
          zlib
          glib
          freetype
          fontconfig
          libGL

          # CLI niceties inside the sandbox
          git
          bashInteractive
          which
        ]);

        fhs = pkgs.buildFHSEnv {
          name = "fishsense-imwut-dev";
          targetPkgs = fhsLibs;
          profile = ''
            export UV_PYTHON=3.13
            echo "fishsense-imwut-dev (FHS + C/Rust toolchain). Run 'uv sync' to set up .venv, then 'uv run jupyter lab'."
          '';
          runScript = "bash";
        };
      in
      {
        devShells.default = fhs.env;
        packages.default = fhs;
        apps.default = {
          type = "app";
          program = "${fhs}/bin/fishsense-imwut-dev";
        };
      });
}
